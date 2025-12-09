import numpy as np
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """
    Risk configuration:

    - max_risk_per_trade: fraction of equity to risk per trade (e.g. 1%)
    - max_leverage: leverage cap
    - max_drawdown_limit: kill-switch vs initial_equity (e.g. 30%)
    - atr_sl_mult: Stop-loss distance in ATR units
    - atr_tp_mult: Take-profit distance in ATR units
    """
    max_risk_per_trade: float = 0.01
    max_leverage: float = 5.0
    max_drawdown_limit: float = 0.30   # 30% kill-switch
    atr_sl_mult: float = 2.0          # SL = 2 × ATR
    atr_tp_mult: float = 4.0          # TP = 4 × ATR


class RiskEngine:
    """
    Risk Engine with:

    - Kill-switch based on drawdown vs initial_equity
    - ATR(14)-based stop-loss and take-profit in PRICE space:
        * Long:  SL = entry - atr_sl_mult * ATR
                 TP = entry + atr_tp_mult * ATR
        * Short: SL = entry + atr_sl_mult * ATR
                 TP = entry - atr_tp_mult * ATR

      ATR here is the absolute ATR_14 (not percentage, not normalized).

    - No volatility-based entry gating:
      PPO decides when to trade; risk engine enforces exits and sizing.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.initial_equity: float = 0.0
        self.kill_switch_triggered: bool = False

    # ============================================================
    # RESET
    # ============================================================

    def reset(self, initial_equity: float):
        """Reset engine at the beginning of each episode."""
        self.initial_equity = float(initial_equity)
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
        price: current price
        equity: current equity
        atr: absolute ATR_14 at current bar (not normalized, not %)
        position: current open position dict OR None
        """

        price = float(price)
        equity = float(equity)

        # 1) Kill-switch: deep drawdown vs starting equity
        if equity < self.initial_equity * (1.0 - self.config.max_drawdown_limit):
            self.kill_switch_triggered = True
            return {
                "kill_switch": True,
                "should_enter": False,
                "should_exit": position is not None,
                "new_position": None,
            }

        # 2) If already in a position -> manage exits
        if position is not None:
            should_exit = self._should_exit(direction, position, price)
            return {
                "kill_switch": False,
                "should_enter": False,
                "should_exit": should_exit,
                "new_position": None,
            }

        # 3) Flat -> entry decisions (no ATR gating; ATR only for SL/TP)
        if position is None and direction != 0:
            new_pos = self._construct_position(direction, price, equity, atr)
            return {
                "kill_switch": False,
                "should_enter": True,
                "should_exit": False,
                "new_position": new_pos,
            }

        # 4) Default -> do nothing
        return {
            "kill_switch": False,
            "should_enter": False,
            "should_exit": False,
            "new_position": None,
        }

    # ============================================================
    # EXIT LOGIC (ATR-based SL/TP)
    # ============================================================

    def _should_exit(
        self,
        direction: int,
        position: dict,
        price: float,
    ) -> bool:
        """
        Exit conditions:

        - Opposite PPO signal
        - Hit stored stop-loss or take-profit level
        """

        d = int(position["direction"])
        stop_loss = float(position["stop_loss"])
        take_profit = float(position["take_profit"])
        price = float(price)

        # 1) Opposite PPO signal → exit immediately
        if direction != 0 and direction == -d:
            return True

        # 2) SL/TP checks
        if d == 1:
            # Long: exit if price <= SL or price >= TP
            if price <= stop_loss or price >= take_profit:
                return True
        else:
            # Short: exit if price >= SL or price <= TP
            if price >= stop_loss or price <= take_profit:
                return True

        return False

    # ============================================================
    # POSITION SIZING (ATR used only for SL/TP distances)
    # ============================================================

    def _construct_position(
        self,
        direction: int,
        price: float,
        equity: float,
        atr: float
    ) -> dict:
        """
        Position sizing:

            capital_at_risk = equity * max_risk_per_trade
            notional        = capital_at_risk * max_leverage
            size            = notional / price

        SL/TP are set in PRICE units using ATR_14:

            Long:
                SL = entry - atr_sl_mult * ATR
                TP = entry + atr_tp_mult * ATR

            Short:
                SL = entry + atr_sl_mult * ATR
                TP = entry - atr_tp_mult * ATR
        """

        price = float(price)
        equity = float(equity)
        atr = float(atr)

        # Safety: ensure ATR positive and not insane
        if not np.isfinite(atr) or atr <= 0:
            # fallback to a small fraction of price, e.g. 0.2%
            atr = 0.002 * price

        # ===== Position size =====
        capital_at_risk = equity * self.config.max_risk_per_trade
        notional = capital_at_risk * self.config.max_leverage
        size = notional / max(price, 1e-8)

        # ===== SL/TP levels =====
        k_sl = self.config.atr_sl_mult
        k_tp = self.config.atr_tp_mult

        if direction == 1:
            # Long
            stop_loss = price - k_sl * atr
            take_profit = price + k_tp * atr
        else:
            # Short
            stop_loss = price + k_sl * atr
            take_profit = price - k_tp * atr

        return {
            "direction": int(direction),
            "entry_price": price,
            "size": float(size),
            "current_price": price,
            "stop_loss": float(stop_loss),
            "take_profit": float(take_profit),
        }