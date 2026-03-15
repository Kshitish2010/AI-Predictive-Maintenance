import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# -----------------------------------
# Root cause detection
# -----------------------------------

def detect_root_cause(text):

    text = text.lower()

    if "vibration" in text:
        return "Possible bearing issue"

    if "temperature" in text or "cooling":
        return "Cooling system problem"

    if "leak" in text:
        return "Hydraulic leak"

    if "power" in text:
        return "Electrical issue"

    return "General inspection required"


# -----------------------------------
# Log analysis
# -----------------------------------

def analyze_logs_for_equipment(df):

    severity_score = {
        "critical":100,
        "high":80,
        "medium":50,
        "low":20
    }

    df["severity_score"] = df["severity"].map(severity_score)

    df["root_cause"] = df["text"].apply(detect_root_cause)

    equipment_scores = df.groupby("equipment_id").agg(
        priority_value=("severity_score","mean"),
        incidents=("text","count")
    ).reset_index()

    return df, equipment_scores


# -----------------------------------
# Maintenance schedule
# -----------------------------------

def generate_maintenance_schedule(scores):

    schedule = scores.sort_values(
        "priority_value",
        ascending=False
    )

    return schedule


# -----------------------------------
# Topic modeling
# -----------------------------------

def build_topic_model(df, n_clusters=3):

    vectorizer = TfidfVectorizer(stop_words="english")

    X = vectorizer.fit_transform(df["text"])

    kmeans = KMeans(n_clusters=min(n_clusters,len(df)))

    df["cluster"] = kmeans.fit_predict(X)

    topics = []

    for c in df["cluster"].unique():

        subset = df[df["cluster"]==c]

        topics.append({
            "cluster":int(c),
            "count":len(subset),
            "example":subset.iloc[0]["text"]
        })

    return topics


# -----------------------------------
# Query logs
# -----------------------------------

def query_history(df, equipment_id="", query_text=""):

    result = df.copy()

    if equipment_id:
        result = result[result["equipment_id"].str.contains(equipment_id)]

    if query_text:
        result = result[result["text"].str.contains(query_text, case=False)]

    return result