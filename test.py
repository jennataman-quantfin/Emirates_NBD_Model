import yfinance as yf

data = yf.download("EMIRATESNBD.AE", start="2020-01-01", end="2025-01-01")
print(data.head(10))
print("Total rows:", len(data))