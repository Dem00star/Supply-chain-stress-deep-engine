# Methodology — Sector 2 Infrastructure Macro-Forecasting Pipeline

This document details the mathematical and architectural pipeline used in the **Sector 2 Infrastructure Fleet Engine** to forecast the pricing trajectories of global industrial metals (Copper and Iron Ore).

Rather than relying on simple moving averages or isolated time-series models (like ARIMA), this engine utilizes a **Domain-Driven Design (DDD)** to synthesize a multi-variable macroeconomic state vector. This vector is processed by a **Temporal Fusion Transformer (TFT)** to output probabilistic risk bounds.

The pipeline consists of **six architectural stages**.

---

## Architectural Pipeline Overview


Global Macro APIs (FRED, yfinance)
      │
      ▼
[Stage 1] Shared Macro State Vector Synthesis (Sector 2 Foundation)
      │
      ▼
[Stage 2] Asset-Specific Proxy Injection (Copper or Iron Ore)
      │
      ▼
[Stage 3] Temporal Alignment & Feature Engineering
      │
      ▼
[Stage 4] Deep Learning Optimization (Temporal Fusion Transformer)
      │
      ▼
[Stage 5] Quantile Loss Calculation (P10, P50, P90)
      │
      ▼
[Stage 6] 7-Day Forward Inference & Risk Bounding



## Stage 1 — Shared Macro State Vector Synthesis

Assets within the same sector are driven by identical macroeconomic forces. Instead of redundantly calling APIs, the system builds a single "Shared Macro Foundation" representing global liquidity and shipping capacity.

**Variables extracted:**

* **Dry_Bulk_Freight_Index (BDRY):** Proxy for physical shipping bottlenecks.
* **Global_Mining_Health (PICK):** Proxy for corporate infrastructure supply constraints.
* **US_Dollar_Index (DTWEXBGS):** Proxy for global currency liquidity.
* **US_10Yr_Treasury (DGS10):** Proxy for the risk-free capital rate.

*Mathematical Intuition:* The US Dollar operates inversely to commodity prices. By feeding this vector into the model, the neural network learns to penalize commodity price projections when Dollar Liquidity shrinks.

---

## Stage 2 — Asset-Specific Proxy Injection

The system forks into specific microservices (Engines) for Copper and Iron Ore. Each engine pulls its unique physical and currency proxies and merges them with the Shared Macro Vector.

**Iron Ore Example:**
Because Australia is the world's premier exporter of Iron Ore, the model does not look at the raw commodity price. It looks at the **Australian Dollar (AUD/USD)** and **BHP Group Equities** as leading physical proxies.

**Copper Example:**
Copper relies on highly liquid futures markets, so the engine directly ingests `HG=F` (COMEX Futures) alongside `COPX` (Copper Miners ETF).

---

## Stage 3 — Temporal Alignment & Feature Engineering

Financial and macroeconomic data operate on different time frequencies (e.g., equities trade daily, Fed data updates weekly/monthly).

1. **Temporal Alignment:** All dataframes are resampled to Business Days (`'B'`). Missing values resulting from frequency mismatches are resolved using Forward Filling (`ffill`), assuming the last known macroeconomic state remains true until updated.
2. **Categorical Extraction:** The date index is transformed into cyclical categorical variables (`day_of_week`, `month`) to allow the neural network to learn seasonal mining and shipping cycles.
3. **Time Indexing:** A continuous integer `time_idx` is generated, which is mathematically required for the Transformer to establish sequential causality.

---

## Stage 4 — Deep Learning Optimization (Temporal Fusion Transformer)

The fused dataset is passed into a **Temporal Fusion Transformer (TFT)**. Unlike basic recurrent neural networks (LSTMs) that struggle with multiple varying inputs, the TFT is purpose-built for complex quantitative finance.

**Key Mathematical Mechanisms inside the TFT:**

1. **Variable Selection Networks:** The model mathematically calculates which inputs actually matter today. If the US Dollar is stable but Dry Bulk shipping rates are spiking, the network dynamically assigns higher "Attention Weights" to the shipping data.
2. **Sequence-to-Sequence Processing:** It uses local LSTMs to identify short-term momentum (e.g., a 3-day bull run) while using Multi-Head Attention to remember long-term macro dependencies (e.g., interest rate cycles).

---

## Stage 5 — The Quantile Loss Formulation

In quantitative trading, predicting a single absolute price (a "point forecast") is useless because it carries a 100% probability of being slightly wrong. Instead, this engine uses a **Quantile Loss Function** to predict a probability distribution.

The model is optimized simultaneously across three quantiles (q = 0.1, 0.5, 0.9) using the formula:

`Loss(y, ŷ, q) = q * max(y - ŷ, 0) + (1 - q) * max(ŷ - y, 0)`

*Where `y` is the actual price, `ŷ` is the predicted price, and `q` is the target quantile.*

* **q = 0.1 (P10 Risk Floor):** The model heavily penalizes over-predictions. It draws a line where it is 90% confident the price will stay *above*.
* **q = 0.5 (P50 Median):** The standard Mean Absolute Error equivalent. The baseline expectation.
* **q = 0.9 (P90 Risk Ceiling):** The model heavily penalizes under-predictions. It draws a line where it is 90% confident the price will stay *below*.

---

## Stage 6 — 7-Day Forward Inference & Risk Bounding

During live inference (executed dynamically via Streamlit), the engine isolates the last known state vector, increments the `time_idx`, and projects the Temporal Fusion Transformer 7 days into the future.

The output is not a single price, but an **80% Confidence Interval** (the spread between P10 and P90).

**Interpretation for Decision Makers:**

* If the spread between P10 and P90 is **narrow**, the macroeconomic data is aligned and the model is highly confident.
* If the spread is **wide**, the system is detecting conflicting signals (e.g., the US Dollar is strong, but Dry Bulk shipping is choked), indicating high market volatility.

