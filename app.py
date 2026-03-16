import streamlit as st
import pandas as pd
import plotly.express as px

from telemetry_simulator import generate_vehicle_telemetry
from anomaly_detector import detect_anomalies
from diagnostic_agent import build_diagnostic_agent


st.set_page_config(page_title="AutoMind", layout="wide")
st.title("AutoMind — AI-Powered Predictive Vehicle Diagnostics")

st.sidebar.header("Simulation Settings")
vehicle_id = st.sidebar.selectbox(
    "Select Vehicle",
    ["TESLA-101", "GM-204", "FORD-309", "RIVIAN-404"]
)
days = st.sidebar.slider("Telemetry Days", 7, 60, 30)
inject_fault = st.sidebar.checkbox("Enable Fault Injection", value=True)

if "df" not in st.session_state:
    st.session_state.df = None

if "agent" not in st.session_state:
    st.session_state.agent = build_diagnostic_agent()

if st.sidebar.button("Load Telemetry"):
    st.session_state.df = generate_vehicle_telemetry(
        vehicle_id=vehicle_id,
        days=days,
        inject_fault=inject_fault
    )

if st.session_state.df is not None:
    df = st.session_state.df
    result = detect_anomalies(df)

    st.subheader("Latest Vehicle Telemetry")
    st.dataframe(df.tail(20), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_temp = px.line(
            df.tail(300),
            x="timestamp",
            y="engine_temp_c",
            title="Engine Temperature Over Time"
        )
        st.plotly_chart(fig_temp, use_container_width=True)

    with col2:
        fig_battery = px.line(
            df.tail(300),
            x="timestamp",
            y="battery_voltage",
            title="Battery Voltage Over Time"
        )
        st.plotly_chart(fig_battery, use_container_width=True)

    st.subheader("Anomaly Summary")
    st.write(f"**Anomaly Count:** {result['anomaly_count']}")
    st.write(f"**Anomaly Rate:** {result['anomaly_rate_pct']}%")
    st.write("**Alerts:**")
    st.json(result["alerts"])

    st.subheader("AI Diagnostic Chat")
    user_question = st.text_input(
        "Ask AutoMind",
        placeholder="Why is my engine overheating?"
    )

    if user_question:
        diagnostic_input = f"""
Vehicle ID: {vehicle_id}
Latest Readings: {result['latest_readings']}
Alerts: {result['alerts']}
Anomaly Count: {result['anomaly_count']}
Anomaly Rate: {result['anomaly_rate_pct']}%

User Question: {user_question}
"""
        response = st.session_state.agent.invoke(
            {
                "input": diagnostic_input,
                "chat_history": [],
            }
        )
        st.write("### Diagnostic Report")
        st.write(response["output"])
else:
    st.info("Click 'Load Telemetry' from the left panel to begin.")