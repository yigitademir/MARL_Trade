import numpy as np
from dataclasses import dataclass


@dataclass
class RiskConfig:
    max_risk_per_trade: float = 0.01        # 1% of equity
    max_leverage: float = 5.0
    max_drawdown_limit: float = 0.30        # 30% kill-switch
    atr_volatility_cap: float = 3.0        # upper bound for ATR_pct_norm
    atr_entry_sensitivity: float = 1.0     # lower = more trades, higher = fewer trades


class RiskEngine:
    """
    Updated Risk Engine for use with ATR_pct_norm.

    Main ideas:
    - ATR_pct_norm is already stable → no absolute thresholds.
    - Volatility affects position sizing and when to allow entries.
    - Signal overrides must be smooth, not binary.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.initial_equity = 0.0
        self.kill_switch_triggered = False

    # ============================================================
    # RESET
    # ============================================================

    def reset(self, initial_equity: float):
        self.initial_equity = initial_equity
        self.kill_switch_triggered = False

    # ============================================================
    # MAIN PROCESSOR
    # ============================================================

    def process_signal(
        self,
        direction: int,
        price: float,
        equity: float,
        atr: float,
        position: dict | None,
    ):
        """
        direction: +1 long, -1 short, 0 hold
        atr: ATR_pct_norm (normalized)
        """

        # Ensure ATR sanity: clip extreme feature values
        atr = float(np.clip(atr, -self.config.atr_volatility_cap, self.config.atr_volatility_cap))

        # 1) Check kill-switch (deep drawdowns)
        if equity < self.initial_equity * (1.0 - self.config.max_drawdown_limit):
            self.kill_switch_triggered = True
            return {
                "kill_switch": True,
                "should_enter": False,
                "should_exit": position is not None,
                "new_position": None,
            }

        # 2) If already in a position → manage exit logic
        if position:
            should_exit = self._should_exit(direction, position)
            return {
                "kill_switch": False,
                "should_enter": False,
                "should_exit": should_exit,
                "new_position": None,
            }

        # 3) If flat → check entry conditions
        if position is None and direction != 0:
            allow = self._entry_condition_from_atr(direction, atr)
            if allow:
                new_pos = self._construct_position(direction, price, equity, atr)
                return {
                    "kill_switch": False,
                    "should_enter": True,
                    "should_exit": False,
                    "new_position": new_pos,
                }

        # Default → no action
        return {
            "kill_switch": False,
            "should_enter": False,
            "should_exit": False,
            "new_position": None,
        }

    # ============================================================
    # ENTRY LOGIC
    # ============================================================

    def _entry_condition_from_atr(self, direction: int, atr: float) -> bool:
        """
        ATR_pct_norm already behaves like a volatility z-score.
        Low ATR → trending market → easier entries
        High ATR → choppy → restrict entries

        Formula:
            permit = exp(-|atr| * sensitivity)
        """

        sensitivity = self.config.atr_entry_sensitivity

        # Exponential decay gating
        entry_prob = float(np.exp(-abs(atr) * sensitivity))

        # If volatility is extreme → reject
        if abs(atr) > self.config.atr_volatility_cap:
            return False

        # Threshold: permit entries when probability is reasonable
        return entry_prob > 0.2      # Very permissive threshold (tweakable)

    # ============================================================
    # EXIT LOGIC
    # ============================================================

    def _should_exit(self, direction: int, position: dict) -> bool:
        """
        Exit conditions:
        - Opposite signal
        - Trailing stop (simple version)
        """

        if direction == -position["direction"]:
            return True

        # Trailing stop (based on price move)
        entry = position["entry_price"]
        d = position["direction"]
        size = position["size"]

        # Unrealized PnL %
        pnl_pct = (position["current_price"] - entry) / entry * d

        if pnl_pct < -0.02:  # -2% loss threshold
            return True

        return False

    # ============================================================
    # POSITION SIZING
    # ============================================================

    def _construct_position(self, direction: int, price: float, equity: float, atr: float):
        """
        ATR_pct_norm → used to scale position size.

        Lower ATR → bigger size allowed
        Higher ATR → smaller size
        """

        # Volatility-based size reduction
        vol_scale = float(np.exp(-abs(atr)))     # Smooth scaling

        # Money at risk
        capital_at_risk = equity * self.config.max_risk_per_trade * vol_scale

        # Position size (contracts)
        size = capital_at_risk * self.config.max_leverage / max(price, 1e-8)

        return {
            "direction": direction,
            "entry_price": price,
            "size": size,
            "current_price": price,
        }