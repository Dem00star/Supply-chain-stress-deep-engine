# 🏗️ Global Supply Chain Stress Engine 
**Sector 2: The Infrastructure Fleet (Copper & Iron Ore)**

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch_Lightning-High_Performance-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-FF4B4B?style=for-the-badge&logo=streamlit)
![Architecture](https://img.shields.io/badge/Architecture-Domain_Driven_Design-00E676?style=for-the-badge)

## 📖 Executive Summary
The **Infrastructure Fleet Engine** is a quantitative macro-forecasting platform designed to predict the price trajectories of global industrial metals (Copper and Iron Ore). 

Instead of relying on isolated time-series data or lagging news sentiment, this engine uses a **Domain-Driven Microservice Architecture** to track the physical realities of the global supply chain. It ingests real-time proxies for Dry Bulk shipping capacity, Australian currency demand, and US Dollar liquidity, fusing them into a shared macroeconomic foundation. 

This data is then processed by a **Temporal Fusion Transformer (TFT)**—a state-of-the-art deep learning architecture utilizing attention mechanisms to output probabilistic, 7-day forward forecasts.

---

## 🏛️ Architectural Philosophy: Domain-Driven Design (DDD)
Monolithic AI models suffer from "Feature Bleed"—where data relevant to one asset (e.g., European weather) adds mathematical noise to an unrelated asset (e.g., Copper). 

This repository is strictly organized using **Sector-Specific Domain-Driven Design**. Copper and Iron Ore belong to the same macro "Infrastructure" sector. They share global macroeconomic drivers (like Chinese construction demand and US interest rates), but rely on completely different physical supply chains.

**Enterprise Advantages of this Architecture:**
1. **The DRY Principle (Don't Repeat Yourself):** A `shared_macro_collector` pulls US Dollar and Global Freight data once, feeding it downstream to all metal engines. This drastically reduces API overhead and compute time.
2. **Feature Isolation:** The Iron Ore model utilizes the AUD/USD exchange rate (as Australia is the primary exporter), while the Copper model isolates COMEX futures.
3. **Fault Tolerance:** If the Copper data pipeline breaks, the Iron Ore microservice remains 100% operational.

---

## ⚙️ The Data Pipeline & Engineering

### 1. The Shared Macro Foundation (Sector 2 Lifeblood)
The system leverages the `fredapi` and `yfinance` to build a unified economic state vector, tracking:
* **Dry Bulk Shipping (`BDRY`):** A proxy for physical Capesize/Panamax maritime congestion.
* **Global Mining Health (`PICK`):** The aggregate health of the world's infrastructure producers.
* **Dollar Liquidity (`DTWEXBGS` & `DGS10`):** Federal Reserve interest rates and dollar strength, the ultimate ceiling on commodity prices.

### 2. The Isolated Asset Microservices
* **⚡ Copper Engine:** Merges the Shared Macro data with `HG=F` (COMEX Futures) and `COPX` (Global Copper Miners ETF).
* **🏗️ Iron Ore Engine:** Merges the Shared Macro data with physical supply proxies (`BHP`, `RIO`) and currency demand proxies (`AUDUSD=X`).

---

## 🧠 Deep Learning Architecture (The Meta-Learner)
Both engines utilize a **Temporal Fusion Transformer (TFT)** powered by `lightning.pytorch` and `pytorch-forecasting`.

* **Why TFT?** Unlike standard LSTMs or GRUs, the TFT uses a multi-head attention mechanism to dynamically weight variables. It can mathematically decide if today's Copper price is being driven more by a spike in Dry Bulk shipping rates or a surge in the US Dollar.
* **Probabilistic Forecasting:** The network uses `QuantileLoss([0.1, 0.5, 0.9])` to generate an 80% confidence interval, providing a Risk Floor (P10), an Expected Median, and a Risk Ceiling (P90) rather than a fragile single-point prediction.
* **Cloud-Native Optimization:** Training environments are explicitly mapped to CPU accelerators to ensure seamless, free deployment on Streamlit Community Cloud.

---

## 📂 Repository Structure (Monorepo)

```text
sector_2_infrastructure/
│
├── data/
│   ├── raw/                 # Ephemeral API pulls
│   │   └── shared_macro/    # Unified Sector 2 economic data
│   └── processed/           # Fused, model-ready PyTorch matrices
│
├── models/                  # Compiled .ckpt PyTorch Lightning weights
│
└── src/
    ├── shared_ingestion/
    │   └── macro_collector.py   # The Central Bank: Pulls shared metrics
    │
    └── engines/
        ├── copper/
        │   ├── asset_collector.py # Fuses COMEX data with Shared Macro
        │   ├── dataset.py         # PyTorch TimeSeriesDataSet builder
        │   ├── trainer.py         # TFT Neural Network Optimizer
        │   └── app.py             # Streamlit Visualization UI
        │
        └── iron_ore/
            ├── asset_collector.py # Fuses BHP/AUD data with Shared Macro
            ├── dataset.py         
            ├── trainer.py         
            └── app.py


## 🚀 Local Development Setup

### 1. Environment Setup
Create a .env file in the root directory and add your Federal Reserve API key:

Code snippet
FRED_API_KEY="your_api_key_here"
Install dependencies:

Bash
pip install torch lightning pytorch-forecasting pandas yfinance fredapi streamlit plotly
2. Run the Data Pipeline (In Order)
First, generate the shared macroeconomic foundation:

Bash
python sector_2_infrastructure/src/shared_ingestion/macro_collector.py
Next, synthesize the asset-specific training matrices:

Bash
python sector_2_infrastructure/src/engines/copper/asset_collector.py
python sector_2_infrastructure/src/engines/iron_ore/asset_collector.py
3. Train the Meta-Learners
Optimize the PyTorch network weights (Restricted to 15 epochs, CPU-safe):

Bash
python sector_2_infrastructure/src/engines/copper/trainer.py
python sector_2_infrastructure/src/engines/iron_ore/trainer.py
4. Launch the Dashboards
Visualize the live global state and inference trajectories:

Bash
# Launch Copper Dashboard
streamlit run sector_2_infrastructure/src/engines/copper/app.py

# Launch Iron Ore Dashboard
streamlit run sector_2_infrastructure/src/engines/iron_ore/app.py