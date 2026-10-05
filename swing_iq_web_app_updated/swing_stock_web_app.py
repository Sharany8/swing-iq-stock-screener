
from flask import Flask, jsonify, request, render_template_string
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import math
import time

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "LT.NS", "BHARTIARTL.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "HINDUNILVR.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TATAMOTORS.NS", "ADANIENT.NS", "ADANIPORTS.NS", "NTPC.NS",
    "POWERGRID.NS", "ONGC.NS", "TATASTEEL.NS", "JSWSTEEL.NS",
    "BEL.NS", "HAL.NS", "COALINDIA.NS", "IOC.NS", "BPCL.NS",
    "ZOMATO.NS", "TRENT.NS"
]

# In a production version, replace DEFAULT_SYMBOLS with a complete
# NSE symbol provider/file. The pipeline itself is designed to handle
# the full universe.
UNIVERSE = DEFAULT_SYMBOLS

CACHE = {
    "basic": {},
    "history": {},
    "news": {},
    "timestamp": {}
}

CACHE_TTL = 15 * 60


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def cached(key, symbol):
    item = CACHE[key].get(symbol)
    stamp = CACHE["timestamp"].get((key, symbol), 0)
    if item is not None and time.time() - stamp < CACHE_TTL:
        return item
    return None


def save_cache(key, symbol, value):
    CACHE[key][symbol] = value
    CACHE["timestamp"][(key, symbol)] = time.time()
    return value


def clean_number(x, default=0.0):
    try:
        if x is None or pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def stock_name(symbol):
    return symbol.replace(".NS", "")


# ============================================================
# MARKET DATA
# ============================================================

def get_basic_data(symbol):
    old = cached("basic", symbol)
    if old:
        return old

    try:
        t = yf.Ticker(symbol)
        info = t.fast_info

        price = clean_number(getattr(info, "last_price", 0))
        market_cap = clean_number(getattr(info, "market_cap", 0)) / 1e7
        volume = clean_number(getattr(info, "last_volume", 0))

        if price <= 0:
            hist = t.history(period="5d", auto_adjust=False)
            if not hist.empty:
                price = clean_number(hist["Close"].iloc[-1])
                volume = clean_number(hist["Volume"].iloc[-1])

        result = {
            "symbol": symbol,
            "name": stock_name(symbol),
            "price": price,
            "market_cap": market_cap,
            "volume": volume
        }
        return save_cache("basic", symbol, result)
    except Exception:
        return {
            "symbol": symbol,
            "name": stock_name(symbol),
            "price": 0,
            "market_cap": 0,
            "volume": 0
        }


def get_history(symbol):
    old = cached("history", symbol)
    if old is not None:
        return old

    try:
        df = yf.download(
            symbol,
            period="6mo",
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = ["Open", "High", "Low", "Close", "Volume"]
        df = df[[x for x in required if x in df.columns]].dropna()

        return save_cache("history", symbol, df)
    except Exception:
        return pd.DataFrame()


# ============================================================
# TECHNICAL INDICATORS
# ============================================================

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - (100 / (1 + rs))
    return value.fillna(50)


def macd(series):
    fast = ema(series, 12)
    slow = ema(series, 26)
    line = fast - slow
    signal = ema(line, 9)
    return line, signal


def atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def technical_analysis(df):
    if df.empty or len(df) < 60:
        return None

    close = df["Close"]
    volume = df["Volume"]

    e20 = ema(close, 20)
    e50 = ema(close, 50)
    r = rsi(close)
    macd_line, macd_signal = macd(close)
    average_volume = volume.rolling(20).mean()
    a = atr(df)

    price = clean_number(close.iloc[-1])
    rsi_value = clean_number(r.iloc[-1], 50)
    ema20 = clean_number(e20.iloc[-1])
    ema50 = clean_number(e50.iloc[-1])
    macd_value = clean_number(macd_line.iloc[-1])
    macd_sig = clean_number(macd_signal.iloc[-1])
    volume_ratio = clean_number(
        volume.iloc[-1] / average_volume.iloc[-1]
        if average_volume.iloc[-1] else 1
    )
    atr_pct = clean_number(a.iloc[-1] / price * 100 if price else 0)

    score = 0

    # Trend: 25
    if price > ema20:
        score += 10
    if ema20 > ema50:
        score += 15

    # Momentum: 25
    if 50 <= rsi_value <= 70:
        score += 15
    elif 45 <= rsi_value < 50:
        score += 7

    if macd_value > macd_sig:
        score += 10

    # Volume: 15
    if volume_ratio >= 1.5:
        score += 15
    elif volume_ratio >= 1.1:
        score += 8

    # Volatility/risk: 10
    if 1 <= atr_pct <= 5:
        score += 10
    elif atr_pct <= 7:
        score += 5

    # Short-term return: 25
    ret5 = clean_number((price / close.iloc[-6] - 1) * 100)
    ret20 = clean_number((price / close.iloc[-21] - 1) * 100)

    if ret5 > 0:
        score += 10
    if ret20 > 0:
        score += 15

    return {
        "technical_score": min(score, 100),
        "rsi": round(rsi_value, 2),
        "ema20": round(ema20, 2),
        "ema50": round(ema50, 2),
        "macd": round(macd_value, 2),
        "volume_ratio": round(volume_ratio, 2),
        "atr_pct": round(atr_pct, 2),
        "return_5d": round(ret5, 2),
        "return_20d": round(ret20, 2)
    }


# ============================================================
# NEWS
# ============================================================

def get_news_score(symbol):
    old = cached("news", symbol)
    if old is not None:
        return old

    try:
        news = yf.Ticker(symbol).get_news(count=8)

        if not news:
            return save_cache("news", symbol, {
                "score": 50,
                "count": 0,
                "label": "No recent news"
            })

        positive_words = [
            "profit", "growth", "surge", "strong", "upgrade", "order",
            "deal", "approval", "record", "expansion", "partnership",
            "buy", "bullish", "positive"
        ]
        negative_words = [
            "loss", "fall", "drop", "downgrade", "fraud", "weak",
            "decline", "warning", "debt", "negative", "sell", "lawsuit"
        ]

        score = 50
        counted = 0

        for item in news:
            title = str(item.get("title", "")).lower()
            pos = sum(w in title for w in positive_words)
            neg = sum(w in title for w in negative_words)
            score += (pos - neg) * 4
            counted += 1

        score = max(0, min(100, score))

        label = "Positive" if score >= 60 else "Negative" if score <= 40 else "Neutral"

        return save_cache("news", symbol, {
            "score": round(score, 1),
            "count": counted,
            "label": label
        })
    except Exception:
        return save_cache("news", symbol, {
            "score": 50,
            "count": 0,
            "label": "Unavailable"
        })


# ============================================================
# PROGRESSIVE SCREENING
# ============================================================

def cheap_screen(symbols, min_cap, max_cap, min_price, max_price, min_volume):
    survivors = []

    for symbol in symbols:
        basic = get_basic_data(symbol)

        if not (min_cap <= basic["market_cap"] <= max_cap):
            continue

        if not (min_price <= basic["price"] <= max_price):
            continue

        if basic["volume"] < min_volume:
            continue

        # Cheap pre-screen using recent price/volume data.
        hist = get_history(symbol)

        if hist.empty or len(hist) < 25:
            continue

        close = hist["Close"]
        vol = hist["Volume"]

        recent_return = clean_number(
            (close.iloc[-1] / close.iloc[-6] - 1) * 100
        )

        avg20 = clean_number(vol.tail(20).mean())
        volume_ratio = clean_number(
            vol.iloc[-1] / avg20 if avg20 else 0
        )

        # This is deliberately inexpensive compared with full analysis.
        prescore = 50
        if recent_return > 0:
            prescore += min(20, recent_return * 4)
        if volume_ratio > 1:
            prescore += min(20, (volume_ratio - 1) * 10)

        survivors.append({
            **basic,
            "recent_return": round(recent_return, 2),
            "volume_ratio": round(volume_ratio, 2),
            "prescore": round(prescore, 2)
        })

    return sorted(survivors, key=lambda x: x["prescore"], reverse=True)


def analyze_candidates(candidates, technical_limit, use_news):
    results = []

    for candidate in candidates[:technical_limit]:
        symbol = candidate["symbol"]
        hist = get_history(symbol)
        tech = technical_analysis(hist)

        if not tech:
            continue

        investor_score = 50  # Adapter placeholder; see note below.
        news = get_news_score(symbol) if use_news else {
            "score": 50, "count": 0, "label": "Disabled"
        }

        # Transparent composite score.
        final_score = (
            tech["technical_score"] * 0.65 +
            investor_score * 0.10 +
            news["score"] * 0.15 +
            min(candidate["prescore"], 100) * 0.10
        )

        final_score = round(min(100, max(0, final_score)), 1)

        if final_score >= 80:
            signal = "Strong Setup"
        elif final_score >= 65:
            signal = "Potential"
        elif final_score >= 50:
            signal = "Watch"
        else:
            signal = "Low Priority"

        results.append({
            **candidate,
            **tech,
            "investor_score": investor_score,
            "news_score": news["score"],
            "news_label": news["label"],
            "news_count": news["count"],
            "swing_score": final_score,
            "signal": signal
        })

    return sorted(results, key=lambda x: x["swing_score"], reverse=True)


# ============================================================
# STOCK DETAILS
# ============================================================

def get_stock_details(symbol):
    hist = get_history(symbol)
    if hist.empty or len(hist) < 30:
        return None

    tech = technical_analysis(hist) or {}
    close = hist["Close"]
    e20 = ema(close, 20)
    e50 = ema(close, 50)
    r = rsi(close)

    chart = []
    for idx, row in hist.tail(120).iterrows():
        chart.append({
            "date": idx.strftime("%d %b"),
            "timestamp": int(idx.timestamp() * 1000),
            "open": round(clean_number(row.get("Open")), 2),
            "high": round(clean_number(row.get("High")), 2),
            "low": round(clean_number(row.get("Low")), 2),
            "close": round(clean_number(row.get("Close")), 2),
            "volume": int(clean_number(row.get("Volume"))),
            "ema20": round(clean_number(e20.loc[idx]), 2),
            "ema50": round(clean_number(e50.loc[idx]), 2),
            "rsi": round(clean_number(r.loc[idx], 50), 2)
        })

    news = get_news_score(symbol)
    basic = get_basic_data(symbol)

    return {
        "symbol": symbol,
        "name": stock_name(symbol),
        "price": basic.get("price", tech.get("price", 0)),
        "market_cap": basic.get("market_cap", 0),
        "volume": basic.get("volume", 0),
        "technical_score": tech.get("technical_score", 0),
        "rsi": tech.get("rsi", 50),
        "ema20": tech.get("ema20", 0),
        "ema50": tech.get("ema50", 0),
        "macd": tech.get("macd", 0),
        "volume_ratio": tech.get("volume_ratio", 0),
        "atr_pct": tech.get("atr_pct", 0),
        "return_5d": tech.get("return_5d", 0),
        "return_20d": tech.get("return_20d", 0),
        "news": news,
        "chart": chart
    }


# ============================================================
# PIPELINE
# ============================================================

def run_screening(params):
    symbols = UNIVERSE

    # Stage 1: whole-universe lightweight scan
    stage1 = len(symbols)

    # Stage 2: cheap filters
    cheap = cheap_screen(
        symbols,
        params["min_cap"],
        params["max_cap"],
        params["min_price"],
        params["max_price"],
        params["min_volume"]
    )

    stage2 = len(cheap)

    # Stage 3: pre-screen rank
    prescreen_limit = params["prescreen_limit"]
    ranked = cheap[:prescreen_limit]

    # Stage 4: technical + optional news
    detailed = analyze_candidates(
        ranked,
        params["technical_limit"],
        params["use_news"]
    )

    final = detailed[:params["final_limit"]]

    for i, item in enumerate(final, 1):
        item["rank"] = i

    return {
        "universe_count": stage1,
        "cheap_count": stage2,
        "prescreen_count": len(ranked),
        "technical_count": len(detailed),
        "final_count": len(final),
        "results": final,
        "timestamp": datetime.now().strftime("%d %b %Y, %H:%M")
    }


# ============================================================
# API
# ============================================================

@app.route("/api/screen", methods=["POST"])
def api_screen():
    data = request.get_json(force=True)

    try:
        min_cap = float(data.get("min_cap", 0))
        max_cap = float(data.get("max_cap", 900000))
        min_price = float(data.get("min_price", 0))
        max_price = float(data.get("max_price", 100000))
        min_volume = float(data.get("min_volume", 100000))
        use_news = bool(data.get("use_news", True))
        mode = data.get("mode", "Balanced")

        limits = {
            "Fast": (100, 50, 10),
            "Balanced": (300, 100, 10),
            "Detailed": (500, 200, 15)
        }

        prescreen_limit, technical_limit, final_limit = limits.get(
            mode, limits["Balanced"]
        )

        result = run_screening({
            "min_cap": min_cap,
            "max_cap": max_cap,
            "min_price": min_price,
            "max_price": max_price,
            "min_volume": min_volume,
            "use_news": use_news,
            "prescreen_limit": prescreen_limit,
            "technical_limit": technical_limit,
            "final_limit": final_limit
        })

        return jsonify({"ok": True, **result})

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/stock/<path:symbol>", methods=["GET"])
def api_stock(symbol):
    try:
        symbol = symbol.upper()
        details = get_stock_details(symbol)
        if not details:
            return jsonify({"ok": False, "error": "Not enough market data for this stock."}), 404
        return jsonify({"ok": True, **details})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ============================================================
# FRONTEND
# ============================================================

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SwingIQ — Swing Stock Research</title>
<style>
:root{
  --bg:#0b0f14;--panel:#111720;--panel2:#0e141b;--line:#202a35;--text:#edf2f7;--muted:#8592a3;
  --green:#35d07f;--red:#f06a7a;--amber:#f0bf63;--blue:#6da8ff;--white:#fff;
}
*{box-sizing:border-box} html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
button,input,select{font:inherit}button{cursor:pointer}
.app{max-width:1320px;margin:auto;padding:0 24px 50px}
.topbar{height:72px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line)}
.brand{display:flex;gap:12px;align-items:center}.mark{width:34px;height:34px;border:1px solid #2e3a47;border-radius:10px;display:grid;place-items:center;font-weight:900;color:var(--green);background:#10171f}.brand-name{font-weight:850;letter-spacing:-.4px}.brand-name span{color:var(--green)}
.toplinks{display:flex;gap:8px;align-items:center}.toplink{border:0;background:transparent;color:var(--muted);padding:8px 10px;border-radius:8px}.toplink:hover{color:var(--text);background:#121922}.market-dot{display:flex;align-items:center;gap:7px;padding:7px 10px;border:1px solid var(--line);border-radius:999px;font-size:11px;color:var(--muted)}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(53,208,127,.1)}
.hero{padding:46px 0 30px;display:grid;grid-template-columns:1.25fr .75fr;gap:28px;align-items:end}.eyebrow{font-size:11px;letter-spacing:1.1px;text-transform:uppercase;color:var(--green);font-weight:800;margin-bottom:13px}.hero h1{font-size:clamp(38px,5vw,64px);line-height:1.02;letter-spacing:-2.8px;margin:0 0 16px;max-width:780px}.hero p{max-width:720px;color:var(--muted);line-height:1.75;margin:0;font-size:15px}.hero-note{justify-self:end;max-width:330px;border-left:2px solid #2b3540;padding-left:18px;color:#aab5c2;font-size:13px;line-height:1.65}.hero-note strong{display:block;color:var(--text);margin-bottom:5px}
.searchbar{display:grid;grid-template-columns:1.15fr .85fr auto;gap:10px;background:var(--panel);border:1px solid var(--line);padding:12px;border-radius:16px}.searchbox{display:flex;align-items:center;gap:10px;padding:0 12px;background:#0c1219;border:1px solid var(--line);border-radius:11px}.searchbox svg{color:#6f7c8b;flex:0 0 auto}.searchbox input{width:100%;background:transparent;border:0;outline:0;color:var(--text);padding:11px 0}.searchbox input::placeholder{color:#657180}.primary{border:0;background:var(--green);color:#06120b;font-weight:850;border-radius:11px;padding:0 20px}.primary:hover{filter:brightness(1.06)}.primary:disabled{opacity:.5;cursor:wait}
.filters{margin-top:10px;background:var(--panel);border:1px solid var(--line);border-radius:16px;overflow:hidden}.filter-head{display:flex;align-items:center;justify-content:space-between;padding:13px 15px}.filter-head button{border:0;background:transparent;color:var(--muted);font-size:12px}.filter-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:10px;padding:0 15px 15px}.filter-field label{display:block;color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.6px;margin-bottom:6px}.filter-field input,.filter-field select{width:100%;background:#0b1118;color:var(--text);border:1px solid var(--line);border-radius:9px;padding:10px 11px;outline:none}.filter-field input:focus,.filter-field select:focus{border-color:#405063}.check{display:flex;align-items:center;gap:7px;height:39px;border:1px solid var(--line);padding:0 11px;border-radius:9px;color:#abb6c3;font-size:12px;background:#0b1118}.check input{accent-color:var(--green)}
.notice{display:flex;gap:10px;align-items:flex-start;margin-top:10px;background:#10161e;border:1px solid var(--line);border-radius:12px;padding:11px 13px;color:#8f9aaa;font-size:12px;line-height:1.55}.notice b{color:#cbd4de}
.statusline{min-height:22px;display:flex;align-items:center;justify-content:space-between;margin:18px 0 8px}.status{font-size:12px;color:var(--muted)}.status strong{color:var(--text)}.spinner{width:13px;height:13px;border:2px solid #26313d;border-top-color:var(--green);border-radius:50%;display:inline-block;animation:spin .8s linear infinite;margin-right:7px;vertical-align:-2px}@keyframes spin{to{transform:rotate(360deg)}}
.results-shell{display:grid;grid-template-columns:1fr;gap:14px}.funnel{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}.funnel-card{padding:13px 14px;background:var(--panel);border:1px solid var(--line);border-radius:12px}.funnel-card b{font-size:18px}.funnel-card span{display:block;margin-top:4px;font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.5px}
.toolbar{display:flex;justify-content:space-between;align-items:center;gap:12px;margin:8px 0}.toolbar h2{font-size:18px;margin:0}.toolbar p{margin:4px 0 0;color:var(--muted);font-size:12px}.toolbar-right{display:flex;gap:8px}.sort{background:#10161e;border:1px solid var(--line);color:#b8c2cd;border-radius:9px;padding:8px 10px;font-size:12px}
.results{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}.stock-card{background:var(--panel);border:1px solid var(--line);border-radius:15px;padding:15px;transition:transform .16s,border-color .16s,background .16s}.stock-card:hover{transform:translateY(-2px);border-color:#344252;background:#121922}.stock-top{display:flex;justify-content:space-between;gap:12px}.stock-title{display:flex;gap:11px}.ticker{width:36px;height:36px;border-radius:10px;background:#0d141b;border:1px solid #26313c;display:grid;place-items:center;font-size:11px;font-weight:900;color:#cdd7e1}.stock-card h3{margin:1px 0 4px;font-size:15px}.price{font-weight:800}.price small{font-weight:500;color:var(--muted);font-size:10px}.scorebox{text-align:right}.score{font-size:24px;font-weight:900;line-height:1}.score-label{font-size:9px;text-transform:uppercase;color:var(--muted);letter-spacing:.7px;margin-top:4px}.good{color:var(--green)}.mid{color:var(--amber)}.bad{color:var(--red)}.meta{display:flex;flex-wrap:wrap;gap:6px;margin:13px 0}.chip{border:1px solid var(--line);background:#0d141b;padding:5px 7px;border-radius:7px;font-size:10px;color:#9eabba}.chip.good-chip{color:#7ee8ad;border-color:#254b37;background:#0d1a13}.chip.red-chip{color:#f694a0;border-color:#4a2930;background:#1a1013}.card-bottom{display:flex;justify-content:space-between;align-items:center;border-top:1px solid var(--line);padding-top:11px}.signal{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:.6px}.view-btn{border:0;background:transparent;color:#aab6c4;font-size:11px;font-weight:750;padding:6px 0}.view-btn:hover{color:var(--white)}
.empty{padding:46px;text-align:center;color:var(--muted);background:var(--panel);border:1px solid var(--line);border-radius:14px}.empty h3{color:var(--text);margin:0 0 7px}
.backdrop{position:fixed;inset:0;background:rgba(0,0,0,.58);opacity:0;pointer-events:none;transition:opacity .2s;z-index:20}.backdrop.show{opacity:1;pointer-events:auto}.drawer{position:fixed;top:0;right:0;width:min(640px,96vw);height:100vh;background:#0b1118;border-left:1px solid var(--line);transform:translateX(102%);transition:transform .22s ease;z-index:21;overflow:auto}.drawer.show{transform:translateX(0)}.drawer-inner{padding:22px}.drawer-head{display:flex;justify-content:space-between;align-items:flex-start;gap:15px}.close{border:1px solid var(--line);background:#10171f;color:#aab5c2;width:34px;height:34px;border-radius:9px}.detail-symbol{font-size:26px;font-weight:900;letter-spacing:-1px}.detail-name{color:var(--muted);font-size:12px;margin-top:3px}.detail-price{font-size:22px;font-weight:900;margin-top:14px}.detail-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:16px 0}.mini{background:#10171f;border:1px solid var(--line);border-radius:10px;padding:11px}.mini span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.5px;margin-bottom:5px}.mini b{font-size:13px}.chart-box{background:#0e141b;border:1px solid var(--line);border-radius:13px;padding:13px}.chart-head{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}.tabs{display:flex;gap:4px}.tab{border:1px solid var(--line);background:#111820;color:#8592a3;border-radius:7px;padding:5px 8px;font-size:10px}.tab.active{color:#e9eef3;background:#17212c}.chart-wrap{height:300px;position:relative}.chart-wrap canvas{width:100%;height:100%;display:block}.tooltip{position:absolute;display:none;pointer-events:none;background:#101820;border:1px solid #2b3744;border-radius:8px;padding:7px 9px;font-size:10px;color:#cbd4de;white-space:nowrap;box-shadow:0 8px 26px rgba(0,0,0,.32)}.legend{display:flex;gap:12px;color:var(--muted);font-size:9px;margin-top:8px}.legend i{display:inline-block;width:7px;height:7px;border-radius:2px;margin-right:4px}.teal{background:var(--green)}.blue{background:var(--blue)}
.section-title{font-size:12px;font-weight:850;margin:18px 0 9px}.breakdown{display:grid;grid-template-columns:1fr 1fr;gap:8px}.factor{background:#10171f;border:1px solid var(--line);border-radius:10px;padding:10px}.factor-top{display:flex;justify-content:space-between;gap:10px;font-size:10px}.bar{height:5px;background:#202a35;border-radius:99px;overflow:hidden;margin-top:8px}.bar span{display:block;height:100%;background:var(--green);border-radius:99px}.newsbox{background:#10171f;border:1px solid var(--line);border-radius:10px;padding:11px;color:#9da9b6;font-size:11px;line-height:1.6}.newsbox strong{color:#dce4ec}.disclaimer{font-size:10px;line-height:1.6;color:#667382;margin-top:16px}
@media(max-width:900px){.hero{grid-template-columns:1fr}.hero-note{justify-self:start}.filter-grid{grid-template-columns:repeat(2,1fr)}.funnel{grid-template-columns:repeat(3,1fr)}.results{grid-template-columns:1fr}}
@media(max-width:600px){.app{padding:0 14px 36px}.topbar{height:62px}.toplinks .toplink{display:none}.hero{padding:34px 0 22px}.hero h1{letter-spacing:-1.8px}.searchbar{grid-template-columns:1fr}.primary{height:44px}.filter-grid{grid-template-columns:1fr}.funnel{grid-template-columns:repeat(2,1fr)}.detail-grid{grid-template-columns:repeat(2,1fr)}.chart-wrap{height:240px}.breakdown{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="app">
<header class="topbar">
  <div class="brand"><div class="mark">S</div><div class="brand-name">Swing<span>IQ</span></div></div>
  <div class="toplinks"><button class="toplink" onclick="scrollToSection('finder')">Finder</button><button class="toplink" onclick="document.getElementById('about').scrollIntoView()">How it works</button><div class="market-dot"><span class="dot"></span>NSE research workspace</div></div>
</header>

<section class="hero" id="finder">
  <div><div class="eyebrow">Research before the move</div><h1>Find a swing setup worth a closer look.</h1><p>SwingIQ narrows a broad stock universe into a short, explainable watchlist using price action, momentum, volume and recent news.</p></div>
  <div class="hero-note"><strong>Not an AI prediction engine.</strong>Every score is built from visible factors, so you can open a stock and see why it made the list.</div>
</section>

<section class="searchbar">
  <div class="searchbox"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="7"></circle><path d="m20 20-4-4"></path></svg><input id="quickSearch" placeholder="Search a stock to inspect it, e.g. RELIANCE"></div>
  <select class="sort" id="quickRange"><option value="" selected>Use the filters below</option><option value="under50">Under ₹50</option><option value="under200">Under ₹200</option><option value="smallcap">Small cap</option></select>
  <button class="primary" id="run" onclick="runScreen()">Find Swing Stocks</button>
</section>

<section class="filters" id="advancedFilters">
  <div class="filter-head"><div><strong>Screening filters</strong><div class="muted">Start broad, then let the ranking do the heavy lifting.</div></div><button onclick="toggleFilters()" id="toggleFiltersBtn">Hide filters</button></div>
  <div class="filter-grid" id="filterGrid">
    <div class="filter-field"><label>Market cap from (₹ Cr)</label><input id="minCap" type="number" value="0" min="0"></div>
    <div class="filter-field"><label>Market cap to (₹ Cr)</label><input id="maxCap" type="number" value="900000" min="0"></div>
    <div class="filter-field"><label>Price from (₹)</label><input id="minPrice" type="number" value="0" min="0"></div>
    <div class="filter-field"><label>Price to (₹)</label><input id="maxPrice" type="number" value="100000" min="0"></div>
    <div class="filter-field"><label>Daily volume above</label><input id="minVolume" type="number" value="100000" min="0"></div>
    <div class="filter-field"><label>Screening depth</label><select id="mode"><option>Fast</option><option selected>Balanced</option><option>Detailed</option></select></div>
    <div class="filter-field"><label>Swing horizon</label><select id="horizon"><option>Next day</option><option selected>3–7 trading days</option><option>1–2 weeks</option></select></div>
    <div class="filter-field"><label>Recent news</label><div class="check"><input id="news" type="checkbox" checked> Include in final score</div></div>
    <div class="filter-field"><label>Quick price preset</label><select id="preset"><option value="">No preset</option><option value="50">Under ₹50</option><option value="100">Under ₹100</option><option value="500">Under ₹500</option></select></div>
    <div class="filter-field"><label>Results</label><div class="check">Top-ranked setups first</div></div>
  </div>
</section>

<div class="notice"><div>◌</div><div><b>Broad search:</b> A very wide market-cap range can cover most listed stocks. The app progressively ranks candidates so detailed calculations are reserved for the stronger setups.</div></div>

<div class="statusline"><div class="status" id="status">Ready for your first screen.</div><div id="loading" style="display:none"><span class="spinner"></span><span class="status">Screening…</span></div></div>

<main id="output"></main>

<section id="about" style="margin-top:46px;border-top:1px solid var(--line);padding-top:25px"><div class="muted" style="font-size:11px;text-transform:uppercase;letter-spacing:.8px">How it works</div><div style="margin-top:8px;color:#a4afbd;font-size:12px;line-height:1.7;max-width:760px">The current prototype scans its configured universe, applies inexpensive filters, ranks candidates, performs technical analysis on a smaller shortlist, then blends that with recent-news context. The next stage can replace the demo universe with a complete NSE stock source and live investor-activity data.</div></section>
<footer class="disclaimer">Educational project only. SwingIQ is decision-support software, not investment advice. Data availability and freshness depend on the market-data provider.</footer>
</div>

<div class="backdrop" id="backdrop" onclick="closeDrawer()"></div>
<aside class="drawer" id="drawer"><div class="drawer-inner" id="drawerInner"></div></aside>

<script>
let lastResults=[];
let chartState={data:[],range:90,showEma:true};
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[c]));}
function num(v,d=0){let n=Number(v);return Number.isFinite(n)?n:d}
function scoreClass(s){return s>=65?'good':s>=50?'mid':'bad'}
function toggleFilters(){const g=document.getElementById('filterGrid');const b=document.getElementById('toggleFiltersBtn');const hidden=g.style.display==='none';g.style.display=hidden?'grid':'none';b.textContent=hidden?'Hide filters':'Show filters'}
function scrollToSection(id){document.getElementById(id).scrollIntoView({behavior:'smooth'})}

document.getElementById('preset').addEventListener('change',e=>{if(e.target.value){document.getElementById('maxPrice').value=e.target.value}})
document.getElementById('quickRange').addEventListener('change',e=>{const v=e.target.value;if(v==='under50')document.getElementById('maxPrice').value=50; if(v==='under200')document.getElementById('maxPrice').value=200; if(v==='smallcap'){document.getElementById('maxCap').value=10000} if(v)runScreen()})
document.getElementById('quickSearch').addEventListener('keydown',e=>{if(e.key==='Enter'){const q=e.target.value.trim().toUpperCase().replace('.NS','');const hit=lastResults.find(x=>x.name===q||x.symbol===q+'.NS');if(hit)openStock(hit.symbol);else if(q)openStock(q+'.NS')}})

async function runScreen(){
 const b=document.getElementById('run'),status=document.getElementById('status'),load=document.getElementById('loading'),output=document.getElementById('output');
 b.disabled=true;load.style.display='flex';status.textContent='Building your shortlist…';output.innerHTML='';
 const payload={min_cap:num(document.getElementById('minCap').value),max_cap:num(document.getElementById('maxCap').value),min_price:num(document.getElementById('minPrice').value),max_price:num(document.getElementById('maxPrice').value),min_volume:num(document.getElementById('minVolume').value),mode:document.getElementById('mode').value,horizon:document.getElementById('horizon').value,use_news:document.getElementById('news').checked};
 try{const r=await fetch('/api/screen',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await r.json();if(!data.ok)throw Error(data.error||'Screening failed');lastResults=data.results||[];status.textContent='Updated '+data.timestamp;renderResults(data)}catch(e){status.textContent='Screening failed';output.innerHTML='<div class="empty"><h3>Something went wrong</h3><div>'+esc(e.message)+'</div></div>'}finally{b.disabled=false;load.style.display='none'}
}

function renderResults(data){
 const output=document.getElementById('output');let cards='';
 data.results.forEach((s,i)=>{const ret=num(s.return_5d),vol=num(s.volume_ratio);cards+=`<article class="stock-card"><div class="stock-top"><div class="stock-title"><div class="ticker">${esc(s.name.slice(0,3))}</div><div><h3>${esc(s.name)}</h3><div class="muted">${esc(s.symbol)} · ₹${num(s.market_cap).toLocaleString('en-IN',{maximumFractionDigits:0})} Cr</div></div></div><div class="scorebox"><div class="score ${scoreClass(s.swing_score)}">${num(s.swing_score).toFixed(1)}</div><div class="score-label">Swing score</div></div></div><div class="price" style="margin-top:13px">₹${num(s.price).toLocaleString('en-IN',{maximumFractionDigits:2})} <small>· ${ret>=0?'+':''}${ret.toFixed(2)}% 5D</small></div><div class="meta"><span class="chip ${num(s.rsi)>=50&&num(s.rsi)<=70?'good-chip':''}">RSI ${num(s.rsi).toFixed(1)}</span><span class="chip">Volume ${vol.toFixed(2)}×</span><span class="chip">EMA20 ${num(s.ema20)>=num(s.ema50)?'above':'below'} EMA50</span><span class="chip">News ${esc(s.news_label)}</span></div><div class="card-bottom"><span class="signal ${scoreClass(s.swing_score)}">${esc(s.signal)}</span><button class="view-btn" onclick="openStock('${esc(s.symbol)}')">View analysis →</button></div></article>`});
 if(!cards)cards='<div class="empty"><h3>No stocks matched</h3><div>Try widening the price, market-cap or volume filters.</div></div>';
 output.innerHTML=`<section class="results-shell"><div class="funnel"><div class="funnel-card"><b>${data.universe_count}</b><span>Universe</span></div><div class="funnel-card"><b>${data.cheap_count}</b><span>Passed filters</span></div><div class="funnel-card"><b>${data.prescreen_count}</b><span>Pre-screen</span></div><div class="funnel-card"><b>${data.technical_count}</b><span>Analysed</span></div><div class="funnel-card"><b>${data.final_count}</b><span>Watchlist</span></div></div><div class="toolbar"><div><h2>Today's shortlist</h2><p>Open any card for the full setup, price chart and score breakdown.</p></div><div class="toolbar-right"><select class="sort" id="sortResults" onchange="sortResults(this.value)"><option value="score">Sort: Swing score</option><option value="return">Sort: 5D return</option><option value="volume">Sort: Volume</option><option value="rsi">Sort: RSI</option></select></div></div><div class="results" id="resultsGrid">${cards}</div></section>`;
}
function sortResults(by){const map={score:'swing_score',return:'return_5d',volume:'volume_ratio',rsi:'rsi'};lastResults=[...lastResults].sort((a,b)=>num(b[map[by]])-num(a[map[by]]));renderResults({results:lastResults,universe_count:lastResults.length,cheap_count:lastResults.length,prescreen_count:lastResults.length,technical_count:lastResults.length,final_count:lastResults.length})}

async function openStock(symbol){
 const drawer=document.getElementById('drawer'),backdrop=document.getElementById('backdrop'),inner=document.getElementById('drawerInner');
 drawer.classList.add('show');backdrop.classList.add('show');inner.innerHTML='<div class="empty" style="margin-top:30px"><span class="spinner"></span> Loading stock analysis…</div>';
 try{const r=await fetch('/api/stock/'+encodeURIComponent(symbol));const data=await r.json();if(!data.ok)throw Error(data.error||'Unable to load stock');renderDetail(data)}catch(e){inner.innerHTML='<div class="empty" style="margin-top:30px"><h3>Could not load this stock</h3><div>'+esc(e.message)+'</div></div>'}
}
function closeDrawer(){document.getElementById('drawer').classList.remove('show');document.getElementById('backdrop').classList.remove('show')}
function renderDetail(d){
 chartState.data=d.chart||[];chartState.range=Math.min(chartState.data.length,90);
 const newsLabel=d.news?.label||'Neutral';
 document.getElementById('drawerInner').innerHTML=`<div class="drawer-head"><div><div class="detail-symbol">${esc(d.name)}</div><div class="detail-name">${esc(d.symbol)} · ${num(d.market_cap).toLocaleString('en-IN',{maximumFractionDigits:0})} Cr market cap</div><div class="detail-price">₹${num(d.price).toLocaleString('en-IN',{maximumFractionDigits:2})}</div></div><button class="close" onclick="closeDrawer()">×</button></div><div class="detail-grid"><div class="mini"><span>RSI</span><b>${num(d.rsi).toFixed(1)}</b></div><div class="mini"><span>5D return</span><b class="${num(d.return_5d)>=0?'good':'bad'}">${num(d.return_5d)>=0?'+':''}${num(d.return_5d).toFixed(2)}%</b></div><div class="mini"><span>Volume</span><b>${num(d.volume_ratio).toFixed(2)}×</b></div><div class="mini"><span>ATR risk</span><b>${num(d.atr_pct).toFixed(2)}%</b></div></div><div class="chart-box"><div class="chart-head"><strong>Price action</strong><div class="tabs"><button class="tab" onclick="setChartRange(30,this)">1M</button><button class="tab active" onclick="setChartRange(90,this)">3M</button><button class="tab" onclick="setChartRange(120,this)">6M</button></div></div><div class="chart-wrap" id="chartWrap"><canvas id="priceCanvas"></canvas><div class="tooltip" id="chartTip"></div></div><div class="legend"><span><i class="teal"></i>Price</span><span><i class="blue"></i>EMA20</span><span><i style="background:#7f8c9b"></i>EMA50</span></div></div><div class="section-title">What is driving the setup</div><div class="breakdown"><div class="factor"><div class="factor-top"><span>Technical</span><b>${num(d.technical_score).toFixed(0)}/100</b></div><div class="bar"><span style="width:${Math.max(0,Math.min(100,num(d.technical_score)))}%"></span></div></div><div class="factor"><div class="factor-top"><span>News context</span><b>${num(d.news?.score,50).toFixed(0)}/100</b></div><div class="bar"><span style="width:${num(d.news?.score,50)}%"></span></div></div><div class="factor"><div class="factor-top"><span>Trend</span><b>${num(d.price)>=num(d.ema20)?'Above EMA20':'Below EMA20'}</b></div><div class="bar"><span style="width:${num(d.price)>=num(d.ema20)?78:38}%"></span></div></div><div class="factor"><div class="factor-top"><span>Momentum</span><b>${num(d.macd)>=0?'Positive MACD':'Negative MACD'}</b></div><div class="bar"><span style="width:${num(d.macd)>=0?72:34}%"></span></div></div></div><div class="section-title">Recent news</div><div class="newsbox"><strong>${esc(newsLabel)}</strong> · ${num(d.news?.count)} recent headlines were available to the prototype scorer. <div style="margin-top:5px">News is treated as context, not as a standalone buy signal.</div></div><div class="disclaimer">The chart is historical market data. The score is an experimental research signal and is not a guarantee of future performance.</div>`;
 requestAnimationFrame(drawChart);window.addEventListener('resize',drawChart,{once:true});
}
function setChartRange(n,btn){chartState.range=Math.min(n,chartState.data.length);document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));btn.classList.add('active');drawChart()}
function drawChart(){
 const c=document.getElementById('priceCanvas'),wrap=document.getElementById('chartWrap');if(!c||!wrap||!chartState.data.length)return;const dpr=window.devicePixelRatio||1,w=wrap.clientWidth,h=wrap.clientHeight;c.width=w*dpr;c.height=h*dpr;const ctx=c.getContext('2d');ctx.scale(dpr,dpr);ctx.clearRect(0,0,w,h);
 const data=chartState.data.slice(-chartState.range);const vals=data.flatMap(x=>[x.close,x.ema20,x.ema50]).filter(Number.isFinite);let min=Math.min(...vals),max=Math.max(...vals),pad=(max-min)*.08||1;min-=pad;max+=pad;const L=8,R=8,T=12,B=22;const px=i=>L+i*((w-L-R)/(data.length-1||1));const py=v=>T+(max-v)*(h-T-B)/(max-min);
 ctx.strokeStyle='#1b2530';ctx.lineWidth=1;for(let i=0;i<4;i++){let y=T+i*(h-T-B)/3;ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke()}
 ctx.fillStyle='#718092';ctx.font='9px sans-serif';for(let i=0;i<4;i++){let y=T+i*(h-T-B)/3;let val=max-i*(max-min)/3;ctx.fillText('₹'+val.toFixed(0),w-40,y-3)}
 function line(key,stroke,width){ctx.beginPath();data.forEach((p,i)=>{let x=px(i),y=py(p[key]);if(i===0)ctx.moveTo(x,y);else ctx.lineTo(x,y)});ctx.strokeStyle=stroke;ctx.lineWidth=width;ctx.stroke()}
 line('ema50','#7f8c9b',1);line('ema20','#6da8ff',1.2);line('close','#35d07f',2);
 ctx.fillStyle='#718092';ctx.font='9px sans-serif';[0,Math.floor(data.length/2),data.length-1].forEach(i=>{if(i>=0){ctx.fillText(data[i].date,px(i)-12,h-5)}});
 const tip=document.getElementById('chartTip');c.onmousemove=e=>{const rect=c.getBoundingClientRect();const idx=Math.max(0,Math.min(data.length-1,Math.round(((e.clientX-rect.left)-L)/(w-L-R)*(data.length-1))));const p=data[idx];tip.style.display='block';tip.style.left=Math.min(w-125,Math.max(0,e.clientX-rect.left+10))+'px';tip.style.top=Math.max(4,e.clientY-rect.top-52)+'px';tip.innerHTML=`<b>${esc(p.date)}</b><br>Close ₹${num(p.close).toFixed(2)} · EMA20 ₹${num(p.ema20).toFixed(2)}`};c.onmouseleave=()=>tip.style.display='none';
}
window.addEventListener('keydown',e=>{if(e.key==='Escape')closeDrawer()});
</script>
</body></html>
"""



@app.route("/")
def home():
    return render_template_string(HTML)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)