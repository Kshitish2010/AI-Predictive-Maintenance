import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from maintenance_agent import analyze_logs_for_equipment, generate_maintenance_schedule, build_topic_model, query_history

st.set_page_config(page_title="AI Maintenance Scheduler", layout="wide")

st.title("🔧 Generative AI Predictive Maintenance System")
st.subheader("AI-based log analysis and predictive maintenance scheduling")

# ------------------------------------------------
# Sample logs
# ------------------------------------------------

@st.cache_data
def get_sample_logs():
    return pd.DataFrame([
        {'timestamp':'2025-09-01 10:23','equipment_id':'EQ-01','text':'Unusual vibration detected in motor assembly','type':'operational_note','severity':'high'},
        {'timestamp':'2025-09-03 14:05','equipment_id':'EQ-03','text':'Cooling fluid level low temperature rising','type':'maintenance_log','severity':'critical'},
        {'timestamp':'2025-09-05 09:11','equipment_id':'EQ-02','text':'Hydraulic leak detected near hose connection','type':'incident_report','severity':'medium'},
        {'timestamp':'2025-09-08 07:42','equipment_id':'EQ-01','text':'Routine inspection passed','type':'maintenance_log','severity':'low'},
        {'timestamp':'2025-09-10 16:20','equipment_id':'EQ-03','text':'Power surge detected sensor error','type':'incident_report','severity':'high'},
        {'timestamp':'2025-09-12 11:30','equipment_id':'EQ-02','text':'Bearing wear and grinding noise detected','type':'operational_note','severity':'high'},
    ])


# ------------------------------------------------
# Upload data
# ------------------------------------------------

st.header("1️⃣ Upload Equipment Logs")

uploaded = st.file_uploader("Upload CSV file", type=["csv"])

if "logs_df" not in st.session_state:
    st.session_state.logs_df = get_sample_logs()
    st.session_state.analysis_ready = False

if uploaded:
    logs_df = pd.read_csv(uploaded)
    st.session_state.logs_df = logs_df
    st.session_state.analysis_ready = False
    st.success("Logs uploaded successfully")

st.dataframe(st.session_state.logs_df)

# ------------------------------------------------
# Run AI analysis
# ------------------------------------------------

st.header("2️⃣ AI Log Analysis")

run_analysis = st.button("Run AI Analysis")

if run_analysis:

    logs_df = st.session_state.logs_df.copy()
    logs_df["timestamp"] = pd.to_datetime(logs_df["timestamp"])

    analyzed, equipment_scores = analyze_logs_for_equipment(logs_df)

    schedule = generate_maintenance_schedule(equipment_scores)

    topics = build_topic_model(analyzed)

    st.session_state.analyzed = analyzed
    st.session_state.equipment_scores = equipment_scores
    st.session_state.schedule = schedule
    st.session_state.topics = topics
    st.session_state.analysis_ready = True

# ------------------------------------------------
# Results
# ------------------------------------------------

if st.session_state.get("analysis_ready"):

    st.subheader("Equipment Risk Summary")
    st.dataframe(st.session_state.equipment_scores)

    st.subheader("Maintenance Priority Schedule")
    st.dataframe(st.session_state.schedule)

    st.subheader("AI Log Topics")

    for t in st.session_state.topics:
        st.write(f"Cluster {t['cluster']} : {t['example']}")

# ------------------------------------------------
# Equipment Health Dashboard
# ------------------------------------------------

st.header("3️⃣ Equipment Health Dashboard")

if st.session_state.get("analysis_ready"):

    scores = st.session_state.equipment_scores

    scores["health_score"] = 100 - scores["priority_value"]

    st.bar_chart(scores.set_index("equipment_id")["health_score"])

# ------------------------------------------------
# Failure Trend
# ------------------------------------------------

st.header("4️⃣ Failure Trend")

if st.session_state.get("analysis_ready"):

    df = st.session_state.analyzed

    trend = df.groupby(pd.Grouper(key="timestamp", freq="W")).size()

    st.line_chart(trend)

# ------------------------------------------------
# Query logs
# ------------------------------------------------

st.header("5️⃣ Query Maintenance History")

equipment_query = st.text_input("Equipment ID")
text_query = st.text_input("Search text")

if st.button("Search Logs"):

    results = query_history(
        st.session_state.analyzed,
        equipment_id=equipment_query,
        query_text=text_query
    )

    st.dataframe(results)

# ------------------------------------------------
# Risk Distribution
# ------------------------------------------------

st.header("6️⃣ Risk Insights")

if st.session_state.get("analysis_ready"):

    scores = st.session_state.equipment_scores

    critical = len(scores[scores["priority_value"] > 80])
    high = len(scores[(scores["priority_value"] > 60) & (scores["priority_value"] <= 80)])
    medium = len(scores[(scores["priority_value"] > 40) & (scores["priority_value"] <= 60)])
    low = len(scores[scores["priority_value"] <= 40])

    risk_df = pd.DataFrame({
        "category":["Critical","High","Medium","Low"],
        "count":[critical,high,medium,low]
    })

    fig, ax = plt.subplots()

    ax.pie(
        risk_df["count"],
        labels=risk_df["category"],
        autopct="%1.1f%%"
    )

    st.pyplot(fig)

# ------------------------------------------------
# AI Chatbot
# ------------------------------------------------

st.header("7️⃣ AI Maintenance Assistant")

question = st.text_input("Ask about equipment maintenance")

if st.button("Ask AI"):

    logs = " ".join(st.session_state.logs_df["text"])

    answer = f"""
Based on maintenance logs:

{logs}

Possible answer:
Check equipment with vibration, cooling issues, and hydraulic leaks first.
"""

    st.write(answer)