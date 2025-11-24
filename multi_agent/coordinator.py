from collections import Counter
from typing import List, Dict, Literal, Optional

Action = int  # 0 = HOLD, 1 = BUY, 2 = SELL

class MultiAgentCoordinator:
    """
    Multi-agent action fusion logic.

    It takes individual agent actions (one per timeframe) and produces
    a single final action to execute in the shared trading environment.

    Parameters
    ----------
    strategy : str
        Coordination strategy name:
        - "majority_vote": most common action wins
        - "priority_order": use a fixed priority over timeframes
    priority : list of str, optional
        Ordered list of timeframes from highest to lowest priority.
        Used only when strategy == "priority_order".
        Example: ["5m", "15m", "1h", "4h"]
    """

    def __init__(self, strategy: Literal["majority_vote", "priority_order"] = "majority_vote", priority: Optional[List[str]] = None,) -> None:
        self.strategy = strategy
        self.priority = priority or ["5m", "15m", "1h", "4h"]

    def decide(self, actions: Dict[str, Action]) -> Action:
        """
        Decide final action given individual agent actions.

        Parameters
        ----------
        actions : dict
            Mapping from timeframe -> action.
            Example: {"5m": 1, "15m": 0, "1h": 2}
        Returns
        -------
        int
            Final fused action: 0 = HOLD, 1 = BUY, 2 = SELL
        """
        if not actions:
            # Failsafe: if no actions, do nothing
            return 0

        if self.strategy == "majority_vote":
            return self._majority_vote(actions)
        elif self.strategy == "priority_order":
            return self._priority_order(actions)
        else:
            raise ValueError(f"Unknown coordination strategy: {self.strategy}")
        
    
    # ------------------------------------------------------------------
    # Strategies
    # ------------------------------------------------------------------

    def _majority_vote(self, actions: Dict[str, Action]) -> Action:
        """
        Majority vote strategy.

        - Count how many agents vote BUY/SELL/HOLD.
        - Use the most common action.
        - In case of tie, apply a deterministic tie-break rule.

        Tie-break rule:
            BUY (1) > SELL (2) > HOLD (0)
        """
        counts = Counter(actions.values())
        # Find action(s) with max count
        max_count = max(counts.values())
        candidates = [a for a, c in counts.items() if c == max_count]

        if len(candidates) == 1:
            return candidates[0]

        # Tie-breaking:
        # prefer BUY (1) over SELL (2), then HOLD (0)
        for preferred in [1, 2, 0]:
            if preferred in candidates:
                return preferred

        # Very unlikely to reach here
        return 0

    def _priority_order(self, actions: Dict[str, Action]) -> Action:
        """
        Priority-based strategy.

        - Go through timeframes in self.priority order.
        - Return the first action that is not HOLD (0).
        - If all HOLD, final action = HOLD.

        Example:
            priority = ["5m", "15m", "1h"]
            actions = {"5m": 0, "15m": 2, "1h": 1}
            -> final action = 2 (SELL) from 15m
        """
        for tf in self.priority:
            if tf in actions:
                a = actions[tf]
                if a != 0:  # non-HOLD action
                    return a

        # All agents HOLD or not present in mapping
        return 0