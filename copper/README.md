# 🚢 Global Supply Chain Stress Engine

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit)

An autonomous, zero-intervention machine learning pipeline that predicts global macroeconomic supply chain stress by forecasting the price trajectory of Dr. Copper (the primary industrial proxy asset).

The system utilizes a **Temporal Fusion Transformer (TFT)** to analyze a multi-modal dataset consisting of institutional financial indicators, physical maritime port congestion, and NLP-derived market sentiment.

## 🧠 Architectural Overview

This engine is designed as a decoupled microservice architecture, completely automated via GitHub Actions for continuous data ingestion and model inference.

### 1. Ingestion Layer (Multi-Modal Data)
* **Quantitative:** Ingests daily macroeconomic leading indicators (S&P 500, Crude Oil WTI) via the **Federal Reserve Economic Data (FRED) API**.
* **Physical:** Sources daily maritime port call volumes (Shanghai, Singapore, Rotterdam, etc.) simulating the **IMF PortWatch / UN Global Platform**.
* **Qualitative:** Scrapes live logistical and maritime industry news via RSS feeds using `feedparser`.

### 2. Signal Processing Layer
* **Variational Mode Decomposition (VMD):** Filters chaotic, high-frequency daily price volatility out of the Copper proxy, isolating the true macroeconomic structural trend into discrete Intrinsic Mode Functions (IMFs).
* **FinBERT Sentiment Analysis:** Processes scraped news headlines through a specialized HuggingFace transformer model to generate a daily numerical stress/panic score.
* **Feature Factory:** Synchronizes asynchronous data streams into a master temporal matrix, engineering 7-day rolling averages and n-day lag features to prevent data leakage.

### 3. Deep Sequence Modeling (PyTorch)
* Bypasses traditional ML in favor of a **Temporal Fusion Transformer (TFT)** built with `lightning.pytorch` and `pytorch-forecasting`.
* Utilizes self-attention mechanisms over a 30-day historical look-back window.
* Trained using `QuantileLoss` to output a robust probabilistic confidence cone (P10 to P90 bounds) rather than a fragile single-point estimate.

### 4. Deployment & UI
* **Inference API:** The heavy PyTorch `.ckpt` model is loaded into RAM via a **FastAPI** server, exposing a strict Pydantic-validated `/predict` endpoint.
* **Executive Dashboard:** A highly interactive, zero-latency **Streamlit** GUI visualizes the 7-day forecast and risk bounds using `plotly`. 

---

## 🚀 Local Development Setup

**1. Clone the repository and install dependencies:**
```bash
git clone [https://github.com/Dem00star/supply-chain-stress-deep-engine.git](https://github.com/Dem00star/supply-chain-stress-deep-engine.git)
cd supply-chain-stress-deep-engine
conda create -n bdi_engine python=3.12
conda activate bdi_engine
pip install -r requirements.txt