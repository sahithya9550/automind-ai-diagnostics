# AutoMind — AI-Powered Predictive Vehicle Diagnostics

AutoMind is an end-to-end GenAI + ML project that simulates real-time OBD-II vehicle telemetry, detects sensor anomalies, generates plain-English diagnostic reports, recommends maintenance actions, and estimates repair costs.

The project is designed for vehicle service centers, fleet managers, and OEMs that handle large volumes of IoT sensor data but lack an intelligent system to predict failures before they happen. AutoMind helps reduce unplanned downtime by combining machine learning, LLM-based diagnostics, and an interactive Streamlit dashboard.

## Why this project matters

Vehicle service centers, fleet operators, and automotive manufacturers generate terabytes of IoT telemetry data every day. However, most systems still rely on manual inspection or static threshold rules instead of intelligent failure prediction and AI-assisted diagnostics.

AutoMind addresses this gap by:
- simulating realistic OBD-II sensor streams
- identifying abnormal behavior in real time using Isolation Forest
- converting sensor anomalies into easy-to-understand diagnostic reports with GPT-4o
- recommending maintenance plans
- estimating repair costs based on issue type
- visualizing telemetry and AI insights through a Streamlit dashboard

## Target users

This solution is relevant for:
- Tesla
- General Motors
- Ford
- Rivian
- Bosch Automotive
- Cox Automotive
- CDK Global
- Dealertrack
- Fleet management platforms like Samsara, Verizon Connect, and Geotab


## Architecture Overview

### Components
- **Telemetry Simulator**  
  Generates realistic OBD-II sensor data such as engine RPM, engine temperature, battery voltage, oil pressure, fuel efficiency, brake pad thickness, and tire pressure.

- **Anomaly Detector (ML)**  
  Uses Scikit-learn Isolation Forest to detect abnormal sensor behavior in recent telemetry windows.

- **Diagnostic Agent (LLM)**  
  Uses GPT-4o with LangChain tools to generate plain-English diagnostic reports, severity insights, repair recommendations, and failure explanations.

- **Maintenance Plan Agent**  
  Recommends maintenance schedules based on mileage and likely wear trends.

- **Cost Estimator Agent**  
  Estimates repair quotes using a parts and labor cost database.

- **Streamlit Dashboard**  
  Provides real-time sensor visualizations and an AI-powered diagnostic chat interface.

## Tech Stack

- Python
- OpenAI GPT-4o
- LangChain
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Plotly

## Key Features

- Realistic vehicle telemetry simulation
- Fault injection for testing overheating and battery degradation scenarios
- ML-based anomaly detection
- LLM-generated root cause analysis in plain English
- Repair cost estimation
- Maintenance schedule recommendations
- Interactive dashboard for monitoring and diagnostics

## Example questions to ask

- Why is my engine overheating?
- What repairs do I need and what will they cost?
- Is my battery failing?
- What maintenance is due soon?
- How serious is this anomaly pattern?

## How to Run

```bash
mkdir automind && cd automind
python3 -m venv venv
source venv/bin/activate
pip install openai langchain langchain-openai scikit-learn pandas numpy streamlit plotly python-dotenv
streamlit run app.py