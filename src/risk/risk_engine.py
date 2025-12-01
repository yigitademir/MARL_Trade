# src/risk/risk_engine.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class RiskConfig:
    """
    Configuration for RiskEngine v1.

    All values are fractions (0.02 = 2%).
    """
    risk_per_trade_pct: float = 0.02       # Equity risk per trade
    atr_multiplier: float = 2.0            # SL distance = ATR * 2
    rr_take_profit: float = 2.0            # TP distance = SL * 2
    max_drawdown_pct: float = 0.30         # 30% kill-switch
    min_atr: float = 1e-6                  # Guard against zero ATR


class RiskEngine:
    """
    RiskEngine v1 — deterministic, rule-based risk management.

    Responsibilities:
    -----------------
    • Apply fixed % risk-per-trade
    • Compute ATR-based SL/TP levels
    • Perform position sizing
    • Track drawdown and kill-switch
    • Check entry/exit conditions

    This engine is independent of PPO and works both for:
      • backtesting
      • later integration into the RL environment (optional)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.config = config or RiskConfig()

        # Drawdown state
        self.peak_equity: Optional[float] = None
        self.current_drawdown: float = 0.0
        self.kill_switch_triggered: bool = False

    # --------------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------------

    def reset(self, initial_equity: float) -> None:
        """
        Reset internal state at the beginning of a backtest or episode.
        """
        self.peak_equity = float(initial_equity)
        self.current_drawdown = 0.0
        self.kill_switch_triggered = False

    def update_equity(self, equity: float) -> float:
        """
        Update equity and drawdown tracking. Trigger kill-switch if needed.

        Returns
        -------
        float : current drawdown fraction (0.25 = 25%)
        """
        if self.peak_equity is None:
            self.peak_equity = float(equity)

        if equity > self.peak_equity:
            self.peak_equity = float(equity)

        self.current_drawdown = (
            (self.peak_equity - equity) / max(self.peak_equity, 1e-8)
        )

        if self.current_drawdown >= self.config.max_drawdown_pct:
            self.kill_switch_triggered = True

        return self.current_drawdown

    # --------------------------------------------------------------
    # Position size and levels
    # --------------------------------------------------------------

    def compute_position_size(self, equity: float, atr: float) -> float:
        """
        Compute size from risk value and ATR distance.

        Size = max_risk_value / stop_distance
        """
        atr = float(atr)

        if atr < self.config.min_atr:
            return 0.0

        max_risk_value = equity * self.config.risk_per_trade_pct
        stop_distance = atr * self.config.atr_multiplier

        if stop_distance <= 0:
            return 0.0

        size = max_risk_value / stop_distance
        return float(max(size, 0.0))

    def compute_levels(self, direction: int, entry_price: float, atr: float) -> Dict[str, float]:
        """
        Generate ATR-based stop-loss and take-profit.

        Long:
            SL = entry - ATR * k
            TP = entry + ATR * k * rr
        Short:
            SL = entry + ATR * k
            TP = entry - ATR * k * rr
        """
        atr = float(atr)
        entry_price = float(entry_price)
        k = self.config.atr_multiplier
        rr = self.config.rr_take_profit

        if direction == 1:
            sl = entry_price - atr * k
            tp = entry_price + atr * k * rr
        elif direction == -1:
            sl = entry_price + atr * k
            tp = entry_price - atr * k * rr
        else:
            raise ValueError(f"Invalid direction {direction}, expected 1 or -1")

        return {"stop_loss": float(sl), "take_profit": float(tp)}

    # --------------------------------------------------------------
    # Exit logic
    # --------------------------------------------------------------

    def check_exit_conditions(
        self,
        price: float,
        direction: int,
        position: Dict[str, Any],
    ) -> Optional[str]:
        """
        Check:
        • stop-loss hit
        • take-profit hit
        • agent direction flip

        Returns "SL", "TP", "SIGNAL_FLIP", or None.
        """
        pos_dir = position["direction"]
        sl = position["stop_loss"]
        tp = position["take_profit"]

        # Stop-loss
        if pos_dir == 1 and price <= sl:
            return "SL"
        if pos_dir == -1 and price >= sl:
            return "SL"

        # Take-profit
        if pos_dir == 1 and price >= tp:
            return "TP"
        if pos_dir == -1 and price <= tp:
            return "TP"

        # Signal flip
        if direction != pos_dir:
            return "SIGNAL_FLIP"

        return None

    # --------------------------------------------------------------
    # Main interface
    # --------------------------------------------------------------

    def process_signal(
        self,
        direction: int,
        price: float,
        equity: float,
        atr: float,
        position: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Main decision function.

        Receives:
            agent signal, price, equity, ATR, open position

        Returns:
            decision dictionary for the backtester
        """
        dd = self.update_equity(equity)

        # Kill-switch → force exit, block entry
        if self.kill_switch_triggered:
            return {
                "should_enter": False,
                "should_exit": position is not None,
                "new_position": None,
                "exit_reason": "KILL_SWITCH" if position else None,
                "kill_switch": True,
                "current_drawdown": dd,
            }

        # Existing position: check for exit
        if position is not None:
            reason = self.check_exit_conditions(price, direction, position)
            if reason is not None:
                return {
                    "should_enter": False,
                    "should_exit": True,
                    "new_position": None,
                    "exit_reason": reason,
                    "kill_switch": False,
                    "current_drawdown": dd,
                }
            return {
                "should_enter": False,
                "should_exit": False,
                "new_position": None,
                "exit_reason": None,
                "kill_switch": False,
                "current_drawdown": dd,
            }

        # No open position → agent wants flat
        if direction == 0:
            return {
                "should_enter": False,
                "should_exit": False,
                "new_position": None,
                "exit_reason": None,
                "kill_switch": False,
                "current_drawdown": dd,
            }

        # Compute size for new entry
        size = self.compute_position_size(equity, atr)
        if size <= 0.0:
            return {
                "should_enter": False,
                "should_exit": False,
                "new_position": None,
                "exit_reason": "SIZE_TOO_SMALL",
                "kill_switch": False,
                "current_drawdown": dd,
            }

        # Build new position object
        levels = self.compute_levels(direction, price, atr)
        new_position = {
            "direction": int(direction),
            "entry_price": price,
            "size": size,
            "stop_loss": levels["stop_loss"],
            "take_profit": levels["take_profit"],
        }

        return {
            "should_enter": True,
            "should_exit": False,
            "new_position": new_position,
            "exit_reason": None,
            "kill_switch": False,
            "current_drawdown": dd,
        }