"""Align the downloaded Binance CSVs into one 1-minute panel: data/panel.parquet.

    python scripts/build_panel.py

Columns: s_* spot BTCUSDT, p_* USDT-M perp, pi_* perp premium index, ix_* multi-exchange
index (research only), e_* ETHUSDT spot, fund_rate. Index = bar OPEN time (UTC).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from btcpred.data import build_panel  # noqa: E402

if __name__ == "__main__":
    P = build_panel()
    print(P.shape, P.index[0], P.index[-1])
    print("missing spot minutes:", int(P["s_missing"].sum()), "| missing perp minutes:", int(P["p_missing"].sum()))
