# tests/test_coordinator_strategies.py
"""
Quick sanity checks for MultiAgentCoordinator strategies.

Run:
    python -m tests.test_coordinator_strategies
"""

from src.multi_agent.coordinator import MultiAgentCoordinator


def pretty(actions_dict):
    return ", ".join(f"{tf}:{a}" for tf, a in actions_dict.items())


def main():
    # Majority vote coordinator
    coord_majority = MultiAgentCoordinator(strategy="majority_vote")

    # Weighted vote coordinator (example: give more weight to lower timeframes)
    coord_weighted = MultiAgentCoordinator(
        strategy="weighted_vote",
        weights={"5m": 2.0, "15m": 1.5, "1h": 1.0, "4h": 1.0},
    )

    test_cases = [
        {
            "name": "All BUY",
            "actions": {"5m": 1, "15m": 1, "1h": 1, "4h": 1},
        },
        {
            "name": "2 BUY vs 2 SELL (tie case)",
            "actions": {"5m": 1, "15m": 1, "1h": 2, "4h": 2},
        },
        {
            "name": "BUY vs SELL vs HOLD mix",
            "actions": {"5m": 1, "15m": 0, "1h": 2, "4h": 0},
        },
        {
            "name": "Only one active agent",
            "actions": {"5m": 1},
        },
        {
            "name": "Weighted advantage to 5m BUY",
            "actions": {"5m": 1, "15m": 2, "1h": 2, "4h": 0},
        },
    ]

    print("\n=== Coordinator Strategy Sanity Checks ===\n")

    for case in test_cases:
        actions = case["actions"]
        maj = coord_majority.decide(actions)
        wgt = coord_weighted.decide(actions)

        print(f"[{case['name']}]")
        print(f"  actions:         {pretty(actions)}")
        print(f"  majority_vote →  {maj}")
        print(f"  weighted_vote →  {wgt}")
        print("-" * 60)

    print("\nDone.\n")


if __name__ == "__main__":
    main()