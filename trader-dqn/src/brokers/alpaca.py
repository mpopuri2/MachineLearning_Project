import os
import time
import requests

class AlpacaPaper:
    def __init__(self, base_url=None):
        self.key = os.environ.get("ALPACA_API_KEY")
        self.secret = os.environ.get("ALPACA_API_SECRET")
        self.base = base_url or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.h = {"APCA-API-KEY-ID": self.key, "APCA-API-SECRET-KEY": self.secret}
        self.trading = f"{self.base}/v2"
        self.market = "https://data.alpaca.markets/v2"

    def account(self):
        r = requests.get(f"{self.trading}/account", headers=self.h)
        r.raise_for_status()
        return r.json()

    def get_quote(self, symbol: str):
        r = requests.get(f"{self.market}/stocks/{symbol}/quotes/latest", headers=self.h)
        r.raise_for_status()
        return r.json()["quote"]["ap"]

    def submit_order(self, symbol: str, qty: int, side: str, type_="market", tif="day"):
        payload = {"symbol": symbol, "qty": qty, "side": side, "type": type_, "time_in_force": tif}
        r = requests.post(f"{self.trading}/orders", headers=self.h, json=payload)
        r.raise_for_status()
        return r.json()