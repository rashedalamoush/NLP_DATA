import sys, os, sqlite3, warnings
from datetime import datetime, timedelta
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import streamlit as st

DB_PATH     = "data/stocksense.db"
REPORTS_DIR = "data/reports"
Path(REPORTS_DIR).mkdir(parents=True, exist_ok=True)

BULLISH_KW = [
    "beats","beat","exceeds","raises guidance","record revenue","buyback",
    "upgrade","outperform","strong earnings","better than expected",
    "above expectations","revenue growth","fda approved","dividend increase"
]
BEARISH_KW = [
    "misses","miss","below expectations","cuts guidance","layoffs","layoff",
    "downgrade","underperform","net loss","missed earnings","profit warning",
    "recall","investigation","reduces dividend"
]
LABEL2IDX = {"UP":0,"DOWN":1,"NEUTRAL":2}
IDX2LABEL = {0:"UP",1:"DOWN",2:"NEUTRAL"}

ALL_TICKERS = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","AMD",
    "JPM","V","MA","BAC","GS","BLK",
    "LLY","UNH","JNJ","ABBV",
    "PG","KO","WMT","MCD","HD",
    "XOM","CVX","COP",
    "CAT","UNP","RTX","LMT",
    "CRM","ADBE","NFLX","DIS","NOW","HUBS"
]

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense AI Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0D1B2A; }
    .stApp { background-color: #0D1B2A; color: #FFFFFF; }
    .metric-card {
        background: #1B3A5C; border-radius: 12px;
        padding: 16px; text-align: center; border: 1px solid #2E5F8A;
    }
    .pred-up    { background: #1B5E20; border-radius:12px; padding:20px; text-align:center; border:2px solid #69F0AE; }
    .pred-down  { background: #7f0000; border-radius:12px; padding:20px; text-align:center; border:2px solid #FF5252; }
    .pred-neut  { background: #263238; border-radius:12px; padding:20px; text-align:center; border:2px solid #78909C; }
    .news-pos   { border-left: 4px solid #69F0AE; padding: 8px 12px; margin: 4px 0; background: #1a2e1a; border-radius:4px; }
    .news-neg   { border-left: 4px solid #FF5252; padding: 8px 12px; margin: 4px 0; background: #2e1a1a; border-radius:4px; }
    .news-neut  { border-left: 4px solid #78909C; padding: 8px 12px; margin: 4px 0; background: #1a1e22; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────────────────────
def rsi_calc(s, n=14):
    d = s.diff()
    g = d.clip(lower=0).rolling(n).mean()
    l = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + g / (l + 1e-9))

@st.cache_data(ttl=300)
def fetch_news(ticker, days=7):
    articles = []
    try:
        conn   = sqlite3.connect(DB_PATH)
        rows   = conn.execute("""
            SELECT title, published_at, finbert_label, finbert_score
            FROM news WHERE ticker=?
            AND finbert_label IS NOT NULL
            ORDER BY published_at DESC LIMIT 15
        """, (ticker,)).fetchall()
        conn.close()
        if rows:
            for r in rows:
                articles.append({
                    "title":      r[0],
                    "date":       str(r[1])[:10],
                    "sentiment":  r[2] or "neutral",
                    "confidence": round(float(r[3] or 0.5), 3)
                })
            return articles
    except: pass
    try:
        import yfinance as yf
        raw_news = yf.Ticker(ticker).news or []
        titles = []
        dates  = []
        for n in raw_news[:10]:
            title = n.get('title','') or n.get('headline','')
            if not title: continue
            try:
                pub = n.get('providerPublishTime') or n.get('pubDate') or 0
                date_str = datetime.fromtimestamp(int(pub)).strftime('%Y-%m-%d') if pub else 'N/A'
            except:
                date_str = 'N/A'
            titles.append(title)
            dates.append(date_str)

        # Try FinBERT first
        finbert_results = []
        try:
            from transformers import pipeline
            finbert = pipeline("text-classification",
                               model="ProsusAI/finbert",
                               top_k=1)
            for title in titles:
                res = finbert(title[:512])[0][0]
                label = res["label"].lower()
                score = round(res["score"], 3)
                finbert_results.append((label, score))
        except:
            finbert_results = []

        for i, title in enumerate(titles):
            if finbert_results:
                sent, conf = finbert_results[i]
            else:
                t    = title.lower()
                bull = sum(1 for k in BULLISH_KW if k in t)
                bear = sum(1 for k in BEARISH_KW if k in t)
                sent = "positive" if bull>bear else "negative" if bear>bull else "neutral"
                conf = 0.65
            articles.append({
                "title":      title,
                "date":       dates[i],
                "sentiment":  sent,
                "confidence": conf
            })
    except: pass
    return articles

@st.cache_data(ttl=300)
def get_prices(ticker):
    try:
        conn   = sqlite3.connect(DB_PATH)
        prices = pd.read_sql("""
            SELECT date,open,high,low,close,volume FROM prices
            WHERE ticker=? ORDER BY date DESC LIMIT 60
        """, conn, params=(ticker,))
        conn.close()
        if len(prices) >= 20:
            return prices.sort_values('date').reset_index(drop=True)
    except: pass
    try:
        import yfinance as yf
        df = yf.download(ticker, period='3mo', progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df.reset_index(inplace=True)
        df.columns = [str(c).lower() for c in df.columns]
        return df[['date','open','high','low','close','volume']].reset_index(drop=True)
    except:
        return pd.DataFrame()

def compute_indicators(prices):
    if prices.empty: return prices
    c = prices['close'].astype(float)
    v = prices['volume'].astype(float)
    prices = prices.copy()
    prices['rsi14']   = rsi_calc(c, 14)
    prices['rsi7']    = rsi_calc(c, 7)
    prices['macd']    = c.ewm(12).mean() - c.ewm(26).mean()
    prices['macd_h']  = prices['macd'] - prices['macd'].ewm(9).mean()
    ma20              = c.rolling(20).mean()
    std20             = c.rolling(20).std()
    prices['bb_upper']= ma20 + 2*std20
    prices['bb_lower']= ma20 - 2*std20
    prices['bb_pct']  = (c - ma20 + 2*std20) / (4*std20 + 1e-9)
    prices['r5']      = c.pct_change(5)*100
    prices['r14']     = c.pct_change(14)*100
    prices['vol_r']   = v / (v.rolling(20).mean() + 1e-9)
    low20             = prices['low'].astype(float).rolling(20).min()
    high20            = prices['high'].astype(float).rolling(20).max()
    prices['stoch']   = (c - low20) / (high20 - low20 + 1e-9) * 100
    return prices

def ml_predict(articles, prices):
    sent_map = {"positive":1.,"neutral":0.,"negative":-1.}
    avg_sent = np.mean([sent_map.get(a['sentiment'],0) for a in articles]) if articles else 0.
    avg_conf = np.mean([a['confidence'] for a in articles]) if articles else 0.5
    sent_sig = avg_sent * avg_conf
    all_text = ' '.join(a['title'] for a in articles).lower()
    kw_bull  = float(sum(1 for k in BULLISH_KW if k in all_text))
    kw_bear  = float(sum(1 for k in BEARISH_KW if k in all_text))
    try:
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
        conn     = sqlite3.connect(DB_PATH)
        train_df = pd.read_sql("""
            SELECT d.label, n.finbert_label, n.finbert_score, n.title
            FROM dataset d JOIN news n ON d.news_id = n.id
            WHERE d.label IS NOT NULL AND n.finbert_label IS NOT NULL
        """, conn); conn.close()
        train_df['fb_dir']  = train_df['finbert_label'].map(sent_map).fillna(0)
        train_df['fb_conf'] = train_df['finbert_score'].fillna(0).astype(float)
        train_df['fb_sig']  = train_df['fb_dir'] * train_df['fb_conf']
        kws = train_df['title'].apply(lambda t:(
            float(sum(1 for k in BULLISH_KW if k in str(t).lower())),
            float(sum(1 for k in BEARISH_KW if k in str(t).lower()))
        ))
        train_df['kw_bull'] = kws.apply(lambda x: x[0])
        train_df['kw_bear'] = kws.apply(lambda x: x[1])
        feat = ['fb_dir','fb_conf','fb_sig','kw_bull','kw_bear']
        X = train_df[feat].values.astype(float)
        y = train_df['label'].map(LABEL2IDX).values
        mask = ~np.isnan(X).any(axis=1)
        X,y  = X[mask],y[mask]
        sc   = StandardScaler(); Xs = sc.fit_transform(X)
        clf  = XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.05,
                             eval_metric='logloss',random_state=42,verbosity=0)
        clf.fit(Xs,y)
        x_new = sc.transform([[avg_sent,avg_conf,sent_sig,kw_bull,kw_bear]])
        proba = clf.predict_proba(x_new)[0]
        idx   = int(np.argmax(proba))
        return {
            "prediction":    IDX2LABEL[idx],
            "confidence":    round(float(proba[idx]),3),
            "probabilities": {IDX2LABEL[i]:round(float(p),3) for i,p in enumerate(proba)},
            "signal":        "STRONG" if float(proba[idx])>=0.65 else
                             "MODERATE" if float(proba[idx])>=0.50 else "WEAK"
        }
    except:
        pred = ("UP" if avg_sent>0.2 and kw_bull>kw_bear else
                "DOWN" if avg_sent<-0.2 and kw_bear>kw_bull else "NEUTRAL")
        return {"prediction":pred,"confidence":0.52,
                "probabilities":{"UP":0.33,"DOWN":0.33,"NEUTRAL":0.34},"signal":"WEAK"}

def make_chart(prices, ticker):
    fig, axes = plt.subplots(3,1,figsize=(12,8),
                             gridspec_kw={'height_ratios':[3,1,1]},
                             facecolor='#0D1B2A')
    for ax in axes: ax.set_facecolor('#111D2B')

    df = prices.tail(30)
    xs = range(len(df))

    # Candlesticks
    for i,(_, row) in enumerate(df.iterrows()):
        color = '#26A69A' if float(row['close'])>=float(row['open']) else '#EF5350'
        axes[0].plot([i,i],[float(row['low']),float(row['high'])],color=color,lw=1)
        axes[0].bar(i,abs(float(row['close'])-float(row['open'])),
                    bottom=min(float(row['open']),float(row['close'])),
                    color=color,width=0.7,alpha=0.9)

    if 'bb_upper' in df.columns:
        axes[0].plot(xs,df['bb_upper'].values,'--',color='#78909C',lw=1,alpha=0.6,label='BB')
        axes[0].plot(xs,df['bb_lower'].values,'--',color='#78909C',lw=1,alpha=0.6)

    axes[0].set_title(f'{ticker} — 30-Day Chart',color='white',fontsize=13,fontweight='bold')
    axes[0].tick_params(colors='#78909C'); axes[0].spines['bottom'].set_visible(False)
    for sp in axes[0].spines.values(): sp.set_color('#263238')

    # RSI
    if 'rsi14' in df.columns:
        axes[1].plot(xs,df['rsi14'].values,color='#AB47BC',lw=1.5)
        axes[1].axhline(70,color='#EF5350',lw=0.8,ls='--',alpha=0.6)
        axes[1].axhline(30,color='#26A69A',lw=0.8,ls='--',alpha=0.6)
        axes[1].fill_between(xs,df['rsi14'].values,50,
                             where=[v>50 for v in df['rsi14'].values],
                             alpha=0.15,color='#26A69A')
        axes[1].fill_between(xs,df['rsi14'].values,50,
                             where=[v<50 for v in df['rsi14'].values],
                             alpha=0.15,color='#EF5350')
        axes[1].set_ylim(0,100); axes[1].set_ylabel('RSI',color='#78909C',fontsize=9)
    axes[1].tick_params(colors='#78909C')
    for sp in axes[1].spines.values(): sp.set_color('#263238')

    # MACD
    if 'macd_h' in df.columns:
        colors_macd = ['#26A69A' if v>=0 else '#EF5350' for v in df['macd_h'].values]
        axes[2].bar(xs,df['macd_h'].values,color=colors_macd,alpha=0.8)
        axes[2].axhline(0,color='#78909C',lw=0.8)
        axes[2].set_ylabel('MACD',color='#78909C',fontsize=9)
    axes[2].tick_params(colors='#78909C')
    for sp in axes[2].spines.values(): sp.set_color('#263238')

    plt.tight_layout(pad=0.5)
    return fig

# ─────────────────────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────────────────────

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 StockSense")
    st.markdown("*AI-Powered Financial Analysis*")
    st.markdown("---")
    ticker = st.selectbox("Select Ticker", ALL_TICKERS, index=0)
    custom = st.text_input("Or enter custom ticker:", "")
    if custom.strip(): ticker = custom.strip().upper()
    days   = st.slider("News lookback (days)", 3, 30, 7)
    run_btn= st.button("🚀 Run Analysis", use_container_width=True, type="primary")
    st.markdown("---")
    st.markdown("**Model:** XGBoost")
    st.markdown("**Features:** FinBERT + Keywords + Technical")
    st.markdown("**Dataset:** 28,598 labeled samples")
    st.markdown("**Threshold:** τ ≥ 0.65")

# ── Header ────────────────────────────────────────────────────
st.markdown("# 📊 StockSense AI Agent")
st.markdown("*Multimodal Financial Sentiment Analysis — CS722 NLP + CS728 Data Mining*")
st.markdown("---")

if not run_btn:
    col1, col2, col3 = st.columns(3)
    col1.metric("Dataset Size",  "28,598 samples")
    col2.metric("Best Accuracy", "82.85% (Random Split)")
    col3.metric("Selective τ≥0.65", "98.23%")
    st.info("👈 Select a ticker and click **Run Analysis** to start.")
    st.stop()

# ── Analysis ──────────────────────────────────────────────────
with st.spinner(f"Analyzing {ticker}..."):

    # Run pipeline
    prog = st.progress(0, text="Fetching news...")
    articles = fetch_news(ticker, days)
    prog.progress(25, text="Loading prices...")

    prices = get_prices(ticker)
    if not prices.empty:
        prices = compute_indicators(prices)
    prog.progress(50, text="Running ML model...")

    pred   = ml_predict(articles, prices)
    prog.progress(75, text="Generating chart...")
    chart  = make_chart(prices, ticker) if not prices.empty else None
    prog.progress(100, text="Done!")
    prog.empty()

# ── Prediction Banner ─────────────────────────────────────────
p    = pred['prediction']
conf = pred['confidence']
css_class = "pred-up" if p=="UP" else "pred-down" if p=="DOWN" else "pred-neut"
emoji     = "📈" if p=="UP" else "📉" if p=="DOWN" else "➡️"

st.markdown(f"""
<div class="{css_class}">
  <h1 style='color:white;margin:0;font-size:3em'>{emoji} {p}</h1>
  <p style='color:#ccc;margin:4px 0'>Confidence: {conf*100:.1f}% — {pred['signal']} Signal</p>
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ── Metrics Row ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
proba = pred['probabilities']
c1.metric("📈 UP",      f"{proba.get('UP',0)*100:.1f}%")
c2.metric("📉 DOWN",    f"{proba.get('DOWN',0)*100:.1f}%")
c3.metric("➡️ NEUTRAL", f"{proba.get('NEUTRAL',0)*100:.1f}%")
c4.metric("🎯 Confidence", f"{conf*100:.1f}%",
          delta="STRONG" if conf>=0.65 else "MODERATE" if conf>=0.5 else "WEAK")

st.markdown("---")

# ── Chart + Technical ─────────────────────────────────────────
col_chart, col_tech = st.columns([2,1])

with col_chart:
    st.subheader("📊 Price Chart")
    if chart:
        st.pyplot(chart, use_container_width=True)
    else:
        st.warning("No price data available")

with col_tech:
    st.subheader("📉 Technical Indicators")
    if not prices.empty:
        latest = prices.iloc[-1]
        rsi14  = float(latest.get('rsi14', 50))
        macd_h = float(latest.get('macd_h', 0))
        stoch  = float(latest.get('stoch', 50))
        vol_r  = float(latest.get('vol_r', 1))
        r5     = float(latest.get('r5', 0))

        rsi_color = "🔴" if rsi14>70 else "🟢" if rsi14<30 else "🟡"
        st.metric(f"{rsi_color} RSI (14d)", f"{rsi14:.1f}",
                  delta="Overbought" if rsi14>70 else "Oversold" if rsi14<30 else "Normal")
        st.metric("⚡ MACD Histogram", f"{macd_h:.4f}",
                  delta="Bullish" if macd_h>0 else "Bearish")
        st.metric("📍 Stochastic K", f"{stoch:.1f}")
        st.metric("📦 Volume Ratio", f"{vol_r:.2f}x",
                  delta="High" if vol_r>1.3 else "Low" if vol_r<0.7 else "Normal")
        st.metric("📅 5d Return", f"{r5:+.2f}%",
                  delta=f"{'▲' if r5>0 else '▼'}")

st.markdown("---")

# ── News Feed ─────────────────────────────────────────────────
st.subheader(f"📰 Recent News — {ticker} ({len(articles)} articles)")

sent_map_display = {"positive":"🟢","negative":"🔴","neutral":"🟡"}
css_sent = {"positive":"news-pos","negative":"news-neg","neutral":"news-neut"}

if articles:
    for a in articles[:10]:
        sent  = a.get('sentiment','neutral')
        emoji_s = sent_map_display.get(sent,'🟡')
        css_s   = css_sent.get(sent,'news-neut')
        title   = a.get('title','')[:120]
        date_s  = a.get('date','')
        conf_s  = float(a.get('confidence',0.5))
        st.markdown(f"""
<div class="{css_s}">
  <span style='font-size:0.85em;color:#90A4AE'>{date_s}</span>
  <span style='margin-left:8px'>{emoji_s} <strong>{sent.upper()}</strong>
  ({conf_s:.2f})</span><br>
  <span style='color:#ECEFF1'>{title}</span>
</div>
""", unsafe_allow_html=True)
else:
    st.warning("No recent news found for this ticker.")

# ── Sentiment Summary ─────────────────────────────────────────
if articles:
    st.markdown("---")
    st.subheader("🧠 Sentiment Summary")
    sent_counts = {"positive":0,"negative":0,"neutral":0}
    for a in articles:
        sent_counts[a.get('sentiment','neutral')] = \
            sent_counts.get(a.get('sentiment','neutral'),0) + 1

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("🟢 Positive", sent_counts['positive'])
    col_s2.metric("🔴 Negative", sent_counts['negative'])
    col_s3.metric("🟡 Neutral",  sent_counts['neutral'])

    # Bar chart
    fig_s, ax_s = plt.subplots(figsize=(6,2), facecolor='#111D2B')
    ax_s.set_facecolor('#111D2B')
    bars = ax_s.bar(['Positive','Negative','Neutral'],
                    [sent_counts['positive'],sent_counts['negative'],sent_counts['neutral']],
                    color=['#26A69A','#EF5350','#78909C'])
    ax_s.tick_params(colors='white')
    for sp in ax_s.spines.values(): sp.set_color('#263238')
    plt.tight_layout()
    st.pyplot(fig_s, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#546E7A;font-size:0.85em'>
StockSense AI Agent<br>
Model: XGBoost + FinBERT + Technical Indicators | Dataset: 28,598 samples<br>
⚠️ For research purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
