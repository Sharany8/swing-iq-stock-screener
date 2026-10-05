# SwingIQ — Interactive Swing Stock Research Web App

## Run in VS Code

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python swing_stock_web_app.py
```

Open: http://127.0.0.1:5000

## Main interactions
- Broad stock screening with filters and quick presets
- Result cards with sorting
- Click **View analysis** to open a stock detail drawer
- Interactive historical price chart with 1M / 3M / 6M controls
- Hover the chart to inspect price and EMA20 values
- Technical score, RSI, 5D return, volume ratio and ATR shown in the detail view
- Recent-news context shown for the selected stock

## Important
The current prototype still uses the configured demo NSE universe in the Python file. Replace that universe with a complete NSE symbol source before presenting it as a complete-market screener. Investor activity is still a placeholder in this prototype.
