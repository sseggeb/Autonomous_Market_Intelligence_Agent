# Autonomous Market Intelligence Agent (AMIA)

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-0.1-green.svg)

## Overview
**AMIA** is a hybrid AI system that combines **Deep Learning (Time-Series Forecasting)** with **Generative AI (Reasoning Agents)** to provide comprehensive financial analysis.

Unlike standard chatbots that hallucinate numbers, AMIA uses a **PyTorch LSTM** model for quantitative prediction and grounds its qualitative analysis in real-world data using a **RAG (Retrieval-Augmented Generation)** pipeline.

The agent is orchestrated using **LangGraph**, allowing for cyclical decision-making where the AI can "change its mind" or seek more data based on confidence intervals.

---

## Architecture

The system operates on a loop of **Perception → Prediction → Reasoning → Action**.

```mermaid
graph TD
    Start[User Input: Ticker] --> Fetch(Fetch Market Data)
    Fetch --> Model(PyTorch LSTM Prediction)
    Model --> Check{Confidence > 90%?}
    Check -- Yes --> Report(Generate Report)
    Check -- No --> RAG(Retrieve Financial News)
    RAG --> Reval(Re-evaluate Context)
    Reval --> Report
    Report --> End
