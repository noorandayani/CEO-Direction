from __future__ import annotations

import io
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, accuracy_score


# -----------------------------
# Helpers
# -----------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-9)
    return 100 - (100 / (1 + rs))


def validate_csv(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols or "close" not in cols:
        raise ValueError("CSV harus punya kolom minimal: Date dan Close (case-insensitive).")

    df = df.rename(columns={cols["date"]: "Date", cols["close"]: "Close"})
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").dropna(subset=["Close"]).set_index("Date")
    return df[["Close"]]


@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_yf(ticker: str, period: str) -> pd.DataFrame:
    data = yf.download(ticker, period=period, auto_adjust=False, progress=False)
    if data is None or data.empty:
        raise ValueError("Data kosong. Cek ticker. Contoh: AAPL, MSFT, BBCA.JK, TLKM.JK.")
    data = data.dropna(subset=["Close"]).copy()
    data.index = pd.to_datetime(data.index)
    return data[["Close", "Open", "High", "Low", "Volume"]]


def make_features(base_df: pd.DataFrame) -> pd.DataFrame:
    df = base_df.copy()
    for col in ["Open", "High", "Low", "Volume"]:
        if col not in df.columns:
            df[col] = np.nan

    close = df["Close"]
    df["ret_1"] = close.pct_change(1)
    df["ret_5"] = close.pct_change(5)
    df["ma_5"] = close.rolling(5).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["std_20"] = close.rolling(20).std()
    df["mom_10"] = close - close.shift(10)
    df["rsi_14"] = rsi(close, 14)

    df["y_next_close"] = close.shift(-1)
    df["y_next_dir"] = (df["y_next_close"] > close).astype(int)
    df["y_next_ret"] = (df["y_next_close"] / close) - 1.0

    df = df.dropna()
    return df


def walk_forward_mae(model, X: pd.DataFrame, y: pd.Series, start_frac: float = 0.75) -> float:
    start = int(len(X) * start_frac)
    preds, trues = [], []
    for i in range(start, len(X)):
        model.fit(X.iloc[:i], y.iloc[:i])
        preds.append(float(model.predict(X.iloc[i:i+1])[0]))
        trues.append(float(y.iloc[i]))
    return float(mean_absolute_error(trues, preds)) if preds else float("nan")


def walk_forward_acc(model, X: pd.DataFrame, y: pd.Series, start_frac: float = 0.75) -> float:
    start = int(len(X) * start_frac)
    preds, trues = [], []
    for i in range(start, len(X)):
        model.fit(X.iloc[:i], y.iloc[:i])
        preds.append(int(model.predict(X.iloc[i:i+1])[0]))
        trues.append(int(y.iloc[i]))
    return float(accuracy_score(trues, preds)) if preds else float("nan")


def signal_from_pred(pred_ret: float, rsi14: float, threshold: float) -> str:
    # Rule sederhana: return pred + filter RSI
    if pred_ret > threshold and rsi14 < 70:
        return "BUY"
    if pred_ret < -threshold and rsi14 > 30:
        return "SELL"
    return "HOLD"


def train_and_forecast(features: pd.DataFrame, horizon: int, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, float, float]:
    target_cols = ["y_next_close", "y_next_dir", "y_next_ret"]
    feature_cols = [c for c in features.columns if c not in target_cols]

    X = features[feature_cols]
    y_price = features["y_next_close"]
    y_dir = features["y_next_dir"]

    reg = RandomForestRegressor(
        n_estimators=700, random_state=42, min_samples_leaf=2, n_jobs=-1
    )
    clf = RandomForestClassifier(
        n_estimators=700, random_state=42, min_samples_leaf=2, n_jobs=-1
    )

    mae = walk_forward_mae(reg, X, y_price, start_frac=0.75)
    acc = walk_forward_acc(clf, X, y_dir, start_frac=0.75)

    # Fit full data
    reg.fit(X, y_price)
    clf.fit(X, y_dir)

    close_series = features["Close"].copy()
    rows = []

    for _ in range(horizon):
        temp_df = pd.DataFrame({"Close": close_series})
        temp_feat = make_features(temp_df)
        latest = temp_feat.iloc[-1:]
        latest_X = latest[feature_cols]

        pred_close = float(reg.predict(latest_X)[0])
        pred_dir = int(clf.predict(latest_X)[0])

        last_close = float(latest["Close"].iloc[0])
        pred_ret = float(pred_close / last_close - 1.0)
        rsi14 = float(latest["rsi_14"].iloc[0])

        next_date = (close_series.index[-1] + pd.tseries.offsets.BDay(1))
        sig = signal_from_pred(pred_ret, rsi14, threshold)

        rows.append({
            "Date": next_date,
            "PredictedClose": pred_close,
            "PredictedReturn": pred_ret,
            "PredictedDirection": "UP" if pred_dir == 1 else "DOWN",
            "Signal": sig,
            "RSI14": rsi14
        })

        close_series.loc[next_date] = pred_close

    forecast_df = pd.DataFrame(rows).set_index("Date")
    return features, forecast_df, mae, acc


def build_plot(hist: pd.DataFrame, forecast: pd.DataFrame, title: str):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist["Close"], mode="lines", name="Historical Close"
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["PredictedClose"], mode="lines+markers", name="Predicted Close"
    ))
    fig.add_trace(go.Scatter(
        x=forecast.index, y=forecast["PredictedClose"], mode="text",
        text=forecast["Signal"], textposition="top center", name="Signal"
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white",
        height=560
    )
    return fig


# -----------------------------
# UI
# -----------------------------

st.set_page_config(page_title="Stock Predictor Pro (Streamlit)", page_icon="📈", layout="wide")

st.title("📈 Stock Predictor Pro (Streamlit)")
st.caption("Support saham Indonesia (.JK) + global. Output: prediksi harga, arah UP/DOWN, sinyal BUY/HOLD/SELL. Edukasi, bukan rekomendasi investasi.")

with st.sidebar:
    st.subheader("Input")
    mode = st.radio("Sumber data", ["Ticker (yfinance)", "Upload CSV"], index=0)

    period = st.selectbox("Periode data", ["1y", "3y", "5y", "10y", "max"], index=2)
    horizon = st.slider("Horizon prediksi (hari bursa)", min_value=1, max_value=60, value=10)
    threshold = st.number_input("Threshold sinyal (return). 0.005 = 0.5%", min_value=0.001, max_value=0.05, value=0.005, step=0.001)

    st.divider()
    st.markdown("**Contoh ticker**")
    st.markdown("- Global: `AAPL`, `MSFT`, `TSLA`")
    st.markdown("- Indonesia: `BBCA.JK`, `TLKM.JK`, `ASII.JK`")
    st.markdown("Jika input `BBCA` tanpa `.JK`, aktifkan auto-suffix di bawah.")
    auto_jk = st.checkbox("Auto tambah .JK jika tidak ada suffix", value=True)

ticker = ""
uploaded = None

colA, colB = st.columns([2, 3])

with colA:
    if mode == "Ticker (yfinance)":
        ticker = st.text_input("Ticker", value="BBCA.JK")
    else:
        uploaded = st.file_uploader("Upload CSV (Date, Close)", type=["csv"])

    run_btn = st.button("🚀 Generate Forecast", type="primary", use_container_width=True)

with colB:
    st.info(
        "Tips cepat:\n"
        "- Untuk Indonesia, ticker biasanya pakai `.JK`.\n"
        "- Sinyal dibuat dari prediksi return + filter RSI (sederhana).\n"
        "- MAE & akurasi dihitung walk-forward supaya lebih realistis."
    )

if run_btn:
    try:
        with st.spinner("Ngambil data dan melatih model..."):
            if mode == "Upload CSV":
                if uploaded is None:
                    st.error("Upload CSV dulu ya.")
                    st.stop()
                df = pd.read_csv(uploaded)
                base = validate_csv(df)
                label = "CSV Upload"
            else:
                t = (ticker or "").strip().upper()
                if not t:
                    st.error("Masukkan ticker.")
                    st.stop()
                if auto_jk and "." not in t and t.isalnum():
                    # heuristic: if user types BBCA, assume BBCA.JK
                    t = f"{t}.JK"
                base = fetch_yf(t, period)
                label = t

            feats = make_features(base)
            if len(feats) < 120:
                st.warning("Data terlalu sedikit untuk hasil yang stabil. Coba pilih period lebih panjang.")
            hist, forecast, mae, acc = train_and_forecast(feats, horizon=horizon, threshold=threshold)

        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("MAE (walk-forward) - Harga", f"{mae:.4f}")
        m2.metric("Akurasi (walk-forward) - Arah", f"{acc:.3f}")
        m3.metric("Forecast horizon", f"{horizon} hari bursa")

        # Plot
        fig = build_plot(hist, forecast, f"Forecast (Demo) | {label}")
        st.plotly_chart(fig, use_container_width=True)

        # Table
        show = forecast.copy()
        show["PredictedReturn"] = (show["PredictedReturn"] * 100.0)
        show = show.rename(columns={"PredictedReturn": "PredictedReturn(%)"})
        st.subheader("Tabel Forecast")
        st.dataframe(show, use_container_width=True)

        # Download forecast CSV
        out = forecast.reset_index()
        out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download forecast CSV",
            data=csv_bytes,
            file_name=f"forecast_{label.replace('.', '_')}.csv",
            mime="text/csv"
        )

        # Show last signal summary
        last_row = forecast.iloc[-1]
        st.success(
            f"Ringkasan hari terakhir: **{forecast.index[-1].strftime('%Y-%m-%d')}** | "
            f"Pred Close: **{last_row['PredictedClose']:.4f}** | "
            f"Arah: **{last_row['PredictedDirection']}** | "
            f"Sinyal: **{last_row['Signal']}**"
        )

    except Exception as e:
        st.error(f"Error: {e}")
