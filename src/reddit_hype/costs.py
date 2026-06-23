"""Transaction-cost model: commission + half-spread + slippage, plus a
participation cap for capacity analysis. All inputs are basis points and live in
``configs/strategy_params.yaml`` so cost assumptions are explicit and sweepable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import Settings, load_settings


@dataclass
class CostModel:
    commission_bps: float = 1.0
    spread_bps: float = 10.0
    slippage_bps: float = 5.0
    borrow_bps_annual: float = 300.0
    max_participation: float = 0.10

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> "CostModel":
        settings = settings or load_settings()
        c = settings.strategy_params.get("costs", {})
        return cls(
            commission_bps=float(c.get("commission_bps", 1.0)),
            spread_bps=float(c.get("spread_bps", 10.0)),
            slippage_bps=float(c.get("slippage_bps", 5.0)),
            borrow_bps_annual=float(c.get("borrow_bps_annual", 300.0)),
            max_participation=float(c.get("max_participation", 0.10)),
        )

    @property
    def one_way_rate(self) -> float:
        """Cost of trading one unit of notional, one side, as a return drag."""
        return (self.commission_bps + self.spread_bps + self.slippage_bps) / 1e4

    def round_trip_rate(self) -> float:
        return 2.0 * self.one_way_rate

    def borrow_drag(self, holding_days: int) -> float:
        """Borrow cost for a short leg held ``holding_days`` (research-only)."""
        return self.borrow_bps_annual / 1e4 * (holding_days / 252.0)

    def capacity_per_name(self, dollar_volume: float) -> float:
        """Max $ tradable in a name per day at the participation cap."""
        return self.max_participation * float(dollar_volume or 0.0)
