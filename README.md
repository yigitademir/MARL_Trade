# Multi-Agent PPO Trading Framework
**Author:** Yiğit Ali Demir  
**Program:** MSc Applied Information & Data Science (HSLU)  
**Thesis Topic:** Multi-Agent PPO Trading Across Multiple Timeframes

---

## 📌 Overview

This repository contains a complete **multi-agent reinforcement learning (MARL)** trading system built around PPO agents trained on multiple timeframes (5m, 15m, 1h, 4h).  
The goal of the thesis project is:

- Train independent agents for each timeframe  
- Coordinate their predictions via a multi-agent decision module  
- Backtest the combined policy  
- Provide a foundation for live algorithmic trading with risk management  

The system is modular, maintainable, and structured for academic reproducibility.

---

## 📂 Repository Structure

```
MARL_Trade/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── BTCUSDT_*.parquet
│   └── processed/
│       └── BTCUSDT_*_features.parquet
│
├── docs/
│   ├── thesis_notes/
│   └── diagrams/
│
├── logs/
│   ├── single_agent/
│   │   ├── results.csv
│   │   └── tensorboard/
│   └── multi_agent/
│       ├── backtests/
│       ├── equity/
│       └── trades/
│
├── models/
│   ├── single_agents/
│   └── multi_agent/
│
├── src/
│   ├── agents/
│   │   ├── train_single_agent.py
│   │   └── train_all_timeframes.py
│   ├── env/
│   │   └── trading_env.py
│   ├── multi_agent/
│   │   ├── coordinator.py
│   │   └── backtester.py
│   └── utils/
│       ├── data_fetcher.py
│       ├── features.py
│       └── data_checker.py
│
├── tests/
│   ├── test_envshapes.py
│   └── multiagent_test.py
│
├── main.py
├── README.md
└── requirements.txt
```

---

## 🧠 System Architecture (High-Level)

```
     DATA PIPELINE
 (fetch → clean → features)
             │
             ▼
   SINGLE-AGENT TRAINING
  PPO_5m, PPO_15m, PPO_1h, PPO_4h
             │
             ▼
 MULTI-AGENT COORDINATOR
 (majority vote → final action)
             │
             ▼
     BACKTEST ENGINE
```

---

## 🧪 Testing

Before any training or backtesting:

```
python -m tests.test_envshapes
```

This verifies:

- Observation shapes  
- Feature integrity  
- Environment consistency across timeframes  

---

## 🚀 Training PPO Agents (All Timeframes)

```
python src/agents/train_all_timeframes.py \
    --symbol BTCUSDT \
    --timeframes 5m,15m,1h,4h \
    --total_timesteps 100000
```

Results saved under:

```
logs/single_agent/results.csv
```

Trained models saved under:

```
models/single_agents/
```

---

## 📊 Multi-Agent Backtest

```
python -m src.multi_agent.multiagent_test
```

Outputs:

- `multi_agent_equity_curve.csv`
- `multi_agent_trades.csv`
- Console summary statistics

---

## 📄 Planned Extensions

- Full MARL (parameter sharing or central critic)
- Market regime classifier agent
- ATR-based trailing stops
- Position sizing agent
- Hyperparameter sweeps  
- Live trading connector  
- Risk management engine (SL/TP, volatility filters)

---

## 📚 License

This project is created as part of a Master's thesis at **HSLU**.  
Use permitted for academic and research purposes.