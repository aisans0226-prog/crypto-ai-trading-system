"""
trading/arbitrage_engine.py — Cross-exchange funding-rate arbitrage.

Strategy: go LONG on exchange with lower price and SHORT on the other
          using perpetual funding-rate differential as edge.

NOTE: True cross-exchange arb requires simultaneous execution
      and is inherently capital-intensive.  This module identifies
      and alerts on opportunities but keeps execution optional.
"""
import asyncio
from dataclasses import dataclass
from typing import List, Optional
from loguru import logger

from data_engine.market_data import MarketDataEngine


@dataclass
class ArbitrageOpportunity:
    symbol: str
    binance_price: float
    bybit_price: float
    spread_pct: float
    binance_funding: float
    favourable: bool


class ArbitrageEngine:
    MIN_SPREAD_PCT = 0.15     # minimum 0.15 % spread to flag opportunity

    def __init__(self, engine: MarketDataEngine) -> None:
        self._engine = engine

    async def scan(self, symbols: List[str]) -> List[ArbitrageOpportunity]:
        opportunities: List[ArbitrageOpportunity] = []
        semaphore = asyncio.Semaphore(30)

        async def check(sym: str) -> Optional[ArbitrageOpportunity]:
            async with semaphore:
                try:
                    # Binance
                    bn_ticker = await self._engine.get_ticker_24h_binance(sym)
                    bn_price = float(bn_ticker["lastPrice"])
                    bn_funding = await self._engine.get_funding_rate_binance(sym)

                    # Bybit (use their ticker endpoint)
                    by_data = await self._engine._get(
                        "https://api.bybit.com/v5/market/tickers",
                        params={"category": "linear", "symbol": sym},
                    )
                    by_price = float(
                        by_data["result"]["list"][0]["lastPrice"]
                    )

                    spread_pct = abs(bn_price - by_price) / min(bn_price, by_price) * 100
                    if spread_pct >= self.MIN_SPREAD_PCT:
                        return ArbitrageOpportunity(
                            symbol=sym,
                            binance_price=bn_price,
                            bybit_price=by_price,
                            spread_pct=round(spread_pct, 4),
                            binance_funding=bn_funding,
                            favourable=bn_funding < -0.001,  # negative funding bonus
                        )
                except Exception as exc:
                    logger.debug("Arb check {} failed: {}", sym, exc)
                return None

        tasks = [check(s) for s in symbols[:100]]   # limit arb scan to top 100
        raw = await asyncio.gather(*tasks)
        opportunities = [o for o in raw if o is not None]
        opportunities.sort(key=lambda x: x.spread_pct, reverse=True)
        logger.info("Arb scan: {} opportunities found", len(opportunities))
        return opportunities
