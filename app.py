from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score

import plotly.graph_objects as go
import plotly.io as pio

app = Flask(__name__)

# -----------------------------
# Data + feature engineering
# -----------------------------

def _validate_csv_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError("CSV harus punya kolom minimal: Date dan Close")

    df = df.rename(columns={cols["date"]: "Date", cols["close"]: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Close"])
    df = df.set_index("Date")
    return df[["Close"]]

def _fetch_data_from_yfinance(ticker: str, period: str = "5y") -> pd.DataFrame:
    # Works for global tickers and Indonesia tickers like BBCA.JK
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise ValueError("Data kosong. Cek ticker (contoh global: AAPL, MSFT; Indonesia: BBCA.JK, TLKM.JK).")
    data = data.dropna(subset=["Close"]).copy()
    data.index = pd.to_datetime(data.index)
    return data[["Close", "Open", "High", "Low", "Volume"]]

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _make_features(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()

    # Ensure expected columns exist
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    close = df["Close"]

    # Core features
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)

    df["ma_5"] = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["std_20"] = close.rolling(20).std()

    df["mom_10"] = close - close.shift(10)
    df["rsi_14"] = _rsi(close, 14)

    # Targets
    df["y_next_close"] = close.shift(-1)
    df["y_next_dir"] = (df["y_next_close"] > close).astype(int)
    df["y_next_ret"] = (df["y_next_close"] / close) - 1.0

    df = df.dropna()
    return df

# -----------------------------
# Walk-forward evaluation
# -----------------------------

def _walk_forward_eval_reg(model, X: pd.DataFrame, y: pd.Series, start_frac: float = 0.7) -> float:
    start = int(len(X) * start_frac)
    preds, trues = [], []
    for i in range(start, len(X)):
        model.fit(X.iloc[:i], y.iloc[:i])
        preds.append(float(model.predict(X.iloc[i:i+1])[0]))
        trues.append(float(y.iloc[i]))
    return float(mean_absolute_error(trues, preds)) if preds else float("nan")

def _walk_forward_eval_clf(model, X: pd.DataFrame, y: pd.Series, start_frac: float = 0.7) -> float:
    start = int(len(X) * start_frac)
    preds, trues = [], []
    for i in range(start, len(X)):
        model.fit(X.iloc[:i], y.iloc[:i])
        preds.append(int(model.predict(X.iloc[i:i+1])[0]))
        trues.append(int(y.iloc[i]))
    return float(accuracy_score(trues, preds)) if preds else float("nan")

# -----------------------------
# Forecasting
# -----------------------------

@dataclass
class ForecastPack:
    hist: pd.DataFrame
    forecast: pd.DataFrame
    mae_price: float
    acc_dir: float
    fig_html: str

def _signal_from_pred(pred_ret: float, rsi14: float, threshold: float = 0.005) -> str:
    if pred_ret > threshold and rsi14 < 70:
        return "BUY"
    if pred_ret < -threshold and rsi14 > 30:
        return "SELL"
    return "HOLD"

def _train_and_forecast(features: pd.DataFrame, horizon_days: int, signal_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    # Use same feature set for both tasks
    feature_cols = [c for c in features.columns if c not in ["y_next_close", "y_next_dir", "y_next_ret"]]
    X = features[feature_cols]

    y_price = features["y_next_close"]
    y_dir = features["y_next_dir"]
    y_ret = features["y_next_ret"]

    reg = RandomForestRegressor(
        n_estimators=600,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    clf = RandomForestClassifier(
        n_estimators=600,
        random_state=42,
        min_samples_leaf=2,
        n_jobs=-1
    )

    # Walk-forward metrics (more realistic than 80/20 once)
    mae_price = _walk_forward_eval_reg(reg, X, y_price, start_frac=0.75)
    acc_dir = _walk_forward_eval_clf(clf, X, y_dir, start_frac=0.75)

    # Fit final on all data
    reg.fit(X, y_price)
    clf.fit(X, y_dir)

    # Iterative forecast: we extend Close by predicted close (business days)
    close_series = features["Close"].copy()
    forecast_rows = []

    for _ in range(horizon_days):
        temp_df = pd.DataFrame({"Close": close_series})
        temp_feat = _make_features(temp_df)

        latest = temp_feat.iloc[-1:]
        latest_X = latest[feature_cols]

        pred_close = float(reg.predict(latest_X)[0])
        pred_dir = int(clf.predict(latest_X)[0])
        pred_ret = float(pred_close / float(latest["Close"].iloc[0]) - 1.0)
        rsi14 = float(latest["rsi_14"].iloc[0])

        next_date = (close_series.index[-1] + pd.tseries.offsets.BDay(1))
        signal = _signal_from_pred(pred_ret, rsi14, threshold=signal_threshold)

        forecast_rows.append({
            "Date": next_date,
            "PredictedClose": pred_close,
            "PredictedReturn": pred_ret,
            "PredictedDirection": "UP" if pred_dir == 1 else "DOWN",
            "Signal": signal,
            "RSI14": rsi14
        })

        close_series.loc[next_date] = pred_close

    forecast_df = pd.DataFrame(forecast_rows).set_index("Date")
    return features, forecast_df, mae_price, acc_dir

def _build_plot(hist_df: pd.DataFrame, forecast_df: pd.DataFrame, ticker_label: str) -> str:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist_df.index, y=hist_df["Close"],
        mode="lines", name="Historical Close"
    ))
    fig.add_trace(go.Scatter(
        x=forecast_df.index, y=forecast_df["PredictedClose"],
        mode="lines+markers", name="Predicted Close"
    ))

    # add signal markers (text)
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["PredictedClose"],
        mode="text",
        text=forecast_df["Signal"],
        textposition="top center",
        name="Signal"
    ))

    fig.update_layout(
        title=f"Forecast (Demo) | {ticker_label}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=560
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")

# -----------------------------
# Web routes
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    pack: Optional[ForecastPack] = None

    if request.method == "POST":
        ticker = (request.form.get("ticker") or "").strip()
        period = (request.form.get("period") or "5y").strip()
        horizon = int(request.form.get("horizon") or "10")
        threshold = float(request.form.get("threshold") or "0.005")

        csv_file = request.files.get("csvfile")

        try:
            if csv_file and csv_file.filename:
                raw = csv_file.read()
                df = pd.read_csv(io.BytesIO(raw))
                base_df = _validate_csv_columns(df)
                ticker_label = "CSV Upload"
            else:
                if not ticker:
                    raise ValueError("Masukkan ticker (global: AAPL, MSFT | Indonesia: BBCA.JK, TLKM.JK) atau upload CSV.")
                base_df = _fetch_data_from_yfinance(ticker, period=period)
                ticker_label = ticker.upper()

            feats = _make_features(base_df)
            hist, forecast, mae_price, acc_dir = _train_and_forecast(feats, horizon_days=horizon, signal_threshold=threshold)
            fig_html = _build_plot(hist, forecast, ticker_label)

            pack = ForecastPack(
                hist=hist,
                forecast=forecast,
                mae_price=mae_price,
                acc_dir=acc_dir,
                fig_html=fig_html
            )

        except Exception as e:
            error = str(e)

    return render_template("index.html", error=error, pack=pack)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    JSON API for integration.
    Body example:
    {
      "ticker": "BBCA.JK",
      "period": "5y",
      "horizon": 10,
      "threshold": 0.005
    }
    """
    body: Dict[str, Any] = request.get_json(force=True) or {}
    ticker = (body.get("ticker") or "").strip()
    period = (body.get("period") or "5y").strip()
    horizon = int(body.get("horizon") or 10)
    threshold = float(body.get("threshold") or 0.005)

    if not ticker:
        return jsonify({"error": "ticker required"}), 400

    try:
        base_df = _fetch_data_from_yfinance(ticker, period=period)
        feats = _make_features(base_df)
        hist, forecast, mae_price, acc_dir = _train_and_forecast(feats, horizon_days=horizon, signal_threshold=threshold)

        out = forecast.reset_index()
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

        return jsonify({
            "ticker": ticker.upper(),
            "period": period,
            "horizon": horizon,
            "threshold": threshold,
            "metrics": {"mae_price": mae_price, "acc_direction": acc_dir},
            "forecast": out.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# For production: gunicorn app:app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
