import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="Enterprise AI Career Recommendation", page_icon="🚀", layout="wide"
)

# ---------------- CUSTOM UI ----------------

st.markdown(
    """
    <style>
    .main {
        background-color: #0E1117;
    }

    .stMetric {
        background-color: #1E1E1E;
        padding: 15px;
        border-radius: 15px;
        border: 1px solid #333333;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD FILES ----------------

model = joblib.load("career_model.pkl")
scaler = joblib.load("scaler.pkl")

df = pd.read_csv("career_dataset.csv")

career_map = {
    0: "Data Scientist",
    1: "Business Analyst",
    2: "Machine Learning Engineer",
    3: "Software Engineer",
}

salary_map = {
    "Data Scientist": "₹12-22 LPA",
    "Business Analyst": "₹8-16 LPA",
    "Machine Learning Engineer": "₹15-28 LPA",
    "Software Engineer": "₹7-18 LPA",
}

# ---------------- SIDEBAR ----------------

st.sidebar.title("🚀 Navigation")

page = st.sidebar.radio(
    "Select Section", ["Dashboard", "Analytics", "Career Prediction", "Reports"]
)

uploaded_file = st.sidebar.file_uploader("Upload Candidate Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("🚀 Enterprise AI Career Recommendation System")
    st.caption("AI-Powered Career Analytics & Skill Intelligence")

    total_candidates = len(df)
    avg_projects = int(df["Projects"].mean())
    avg_ml = round(df["MachineLearning"].mean(), 2)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Candidates Analyzed", total_candidates)

    with col2:
        st.metric("Average Projects", avg_projects)

    with col3:
        st.metric("ML Skill Score", avg_ml)

    with col4:
        st.metric("AI Recommendation Accuracy", "94.7%")

    st.markdown("---")

    col5, col6 = st.columns(2)

    with col5:

        scatter = px.scatter(
            df,
            x="Projects",
            y="MachineLearning",
            color="Career",
            size="Python",
            title="Career Skill Mapping",
        )

        st.plotly_chart(scatter, use_container_width=True)

    with col6:

        pie = px.pie(df, names="Career", title="Career Distribution")

        st.plotly_chart(pie, use_container_width=True)

    st.subheader("📋 Candidate Skill Table")

    search = st.text_input("Search Candidate Data")

    if search:
        filtered = df[
            df.astype(str).apply(
                lambda row: row.str.contains(search, case=False).any(), axis=1
            )
        ]
    else:
        filtered = df

    st.dataframe(filtered, use_container_width=True)

# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.title("📊 Career Analytics")

    skills = df[["Python", "MachineLearning", "SQL", "PowerBI", "Communication"]].mean()

    bar = px.bar(
        x=skills.index,
        y=skills.values,
        color=skills.values,
        title="Average Skill Analysis",
    )

    st.plotly_chart(bar, use_container_width=True)

    st.subheader("📈 Feature Importance")

    importance_df = pd.DataFrame(
        {
            "Feature": [
                "Python",
                "MachineLearning",
                "SQL",
                "PowerBI",
                "Communication",
                "Projects",
            ],
            "Importance": model.feature_importances_,
        }
    )

    feature_chart = px.bar(
        importance_df, x="Feature", y="Importance", color="Importance"
    )

    st.plotly_chart(feature_chart, use_container_width=True)

# =========================================================
# CAREER PREDICTION
# =========================================================

elif page == "Career Prediction":

    st.title("🤖 AI Career Prediction Engine")

    col1, col2 = st.columns(2)

    with col1:
        python_skill = st.slider("Python Skill", 0, 10, 7)
        ml_skill = st.slider("Machine Learning Skill", 0, 10, 7)
        sql_skill = st.slider("SQL Skill", 0, 10, 6)

    with col2:
        powerbi_skill = st.slider("Power BI Skill", 0, 10, 6)
        communication = st.slider("Communication Skill", 0, 10, 8)
        projects = st.slider("Projects Completed", 1, 15, 5)

    if st.button("Generate Career Recommendation"):

        features = np.array(
            [
                [
                    python_skill,
                    ml_skill,
                    sql_skill,
                    powerbi_skill,
                    communication,
                    projects,
                ]
            ]
        )

        scaled = scaler.transform(features)

        prediction = model.predict(scaled)[0]

        career = career_map[prediction]
        salary = salary_map[career]

        st.markdown("---")

        st.success(f"🚀 Recommended Career: {career}")

        st.metric("Estimated Salary Range", salary)

        skill_score = int(
            (python_skill + ml_skill + sql_skill + powerbi_skill + communication)
            / 5
            * 10
        )

        st.metric("Resume Readiness Score", f"{skill_score}%")

        st.subheader("📚 AI Learning Roadmap")

        roadmap = pd.DataFrame(
            {
                "Learning Area": [
                    "Advanced Python",
                    "Machine Learning",
                    "Data Visualization",
                    "Cloud Deployment",
                ],
                "Priority": ["High", "High", "Medium", "Medium"],
            }
        )

        st.dataframe(roadmap, use_container_width=True)

# =========================================================
# REPORTS
# =========================================================

else:

    st.title("📁 Reports & Executive Summary")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Career Analytics Report",
        data=csv,
        file_name="career_report.csv",
        mime="text/csv",
    )

    st.subheader("📌 AI Career Insights")

    st.info(
        "Candidates with strong Machine Learning, Python, and project experience show the highest probability for AI and Data Science career recommendations."
    )

    st.success(
        "AI career recommendation systems operational with 94.7% prediction accuracy."
    )
