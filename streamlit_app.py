import streamlit as st
import pandas as pd
import os
import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

st.set_page_config(
    page_title="Salary Prediction Dashboard",
    page_icon="💼",
    layout="wide"
)

SUPABASE_URL = st.secrets["SUPABASE_URL"] if "SUPABASE_URL" in st.secrets else os.getenv("SUPABASE_URL")
SUPABASE_KEY = st.secrets["SUPABASE_KEY"] if "SUPABASE_KEY" in st.secrets else os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Missing SUPABASE_URL or SUPABASE_KEY")
    st.stop()

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
API_URL = "https://salarypredictionapp.onrender.com/predict"

@st.cache_data(ttl=30)
def load_data():
    response = supabase.table("salary_predictions").select("*").order("created_at", desc=True).execute()
    return pd.DataFrame(response.data or [])


def save_single_record_to_supabase(record: dict):
    return supabase.table("salary_predictions").insert(record).execute()


st.title("💼 Salary Prediction Dashboard")
st.caption("Explore salary estimates for data-related roles using a trained Decision Tree model.")

df = load_data()

tab1, tab2, tab3 = st.tabs(["📊 Overview", "⚡ Live Prediction", "🗂️ Stored Results"])

with tab1:
    st.subheader("Project Overview")
    st.write(
        "This dashboard summarizes salary predictions generated from job-related inputs "
        "such as experience level, employment type, company size, residence, remote ratio, and company location."
    )

    if df.empty:
        st.warning("No prediction data was found in Supabase.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(df))
        col2.metric("Average Predicted Salary", f"${df['predicted_salary'].mean():,.0f}")
        col3.metric("Unique Job Titles", df["job_title"].nunique())

        st.markdown("---")
        st.subheader("Salary Visualizations")
        st.write("These charts highlight how predicted salaries vary across experience levels and job titles.")

        st.markdown("#### 1) Average Predicted Salary by Experience Level")
        if os.path.exists("chart_by_experience.png"):
            st.image("chart_by_experience.png", use_container_width=True)
        else:
            st.info("chart_by_experience.png not found")

        st.markdown("#### 2) Average Predicted Salary by Job Title")
        if os.path.exists("chart_by_job_title.png"):
            
            st.image("chart_by_job_title.png", use_container_width=True)
        else:
            st.info("chart_by_job_title.png not found")
        


with tab2:
    st.subheader("Live Salary Prediction")
    st.write(
        "Enter job details below to generate a fresh salary estimate. "
        "The result will be displayed instantly and saved to Supabase."
    )

    with st.form("prediction_form"):
        left, right = st.columns(2)

        with left:
            experience_level = st.selectbox(
                "Experience Level",
                ["EN", "MI", "SE", "EX"],
                help="EN = Entry, MI = Mid, SE = Senior, EX = Executive"
            )
            employment_type = st.selectbox(
                "Employment Type",
                ["FT", "PT", "CT", "FL"],
                help="FT = Full-time, PT = Part-time, CT = Contract, FL = Freelance"
            )
            job_title = st.selectbox("Job Title", ["Data Scientist", "Data Engineer", "Data Analyst","ML Engineer","Research Scientist"], value="Data Scientist")
            company_size = st.selectbox("Company Size", ["S", "M", "L"])

        with right:
            employee_residence = st.selectbox("Employee Residence", ["US", "DE"], value="US")
            remote_ratio = st.selectbox("Remote Ratio", [0, 50, 100],value=0)
            company_location = st.selectbox("Company Location", ["US", "DE"], value="US")

        submitted = st.form_submit_button("Generate Prediction")

    if submitted:
        params = {
            "experience_level": experience_level,
            "employment_type": employment_type,
            "job_title": job_title.strip(),
            "company_size": company_size,
            "employee_residence": employee_residence.strip(),
            "remote_ratio": remote_ratio,
            "company_location": company_location.strip()
        }

        with st.spinner("Calculating salary prediction..."):
            try:
                response = requests.get(API_URL, params=params, timeout=20)

                if response.status_code == 200:
                    result = response.json()
                    predicted_salary = float(result["predicted_salary"])

                    llm_analysis = f"""
The model predicts that a **{job_title}** with **{experience_level}** experience and **{employment_type}** employment
at a **{company_size}** company may earn around **${predicted_salary:,.2f}**.

Key observations:
- Experience level and job title are major salary drivers.
- Geographic fields such as employee residence and company location can strongly affect compensation.
- Remote ratio may influence salary depending on the role and market.

This value is a machine learning estimate and should be treated as guidance, not a guaranteed real-world salary.
""".strip()

                    st.success("Prediction generated successfully.")
                    st.balloons()

                    c1, c2 = st.columns([1, 1])

                    with c1:
                        st.markdown("### 💰 Expected Salary")
                        st.metric("💵 Estimated Salary", f"${predicted_salary:,.2f}")

                        st.markdown("### Input Summary")
                        st.write({
                            "Experience Level": experience_level,
                            "Employment Type": employment_type,
                            "Job Title": job_title,
                            "Company Size": company_size,
                            "Employee Residence": employee_residence,
                            "Remote Ratio": remote_ratio,
                            "Company Location": company_location
                        })

                    with c2:
                        st.markdown("### 🧠 Analysis Panel")
                        st.write(llm_analysis)

                    record = {
                        "experience_level": experience_level,
                        "employment_type": employment_type,
                        "job_title": job_title,
                        "company_size": company_size,
                        "employee_residence": employee_residence,
                        "remote_ratio": int(remote_ratio),
                        "company_location": company_location,
                        "predicted_salary": predicted_salary,
                        "llm_analysis": llm_analysis,
                        "chart_experience_path": "salary_chart.png",
                        "chart_job_title_path": "salary_by_job_title.png"
                    }

                    save_single_record_to_supabase(record)
                    st.info("The new prediction has been saved to Supabase.")

                else:
                    st.error(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.RequestException as e:
                st.error(f"Request failed: {e}")


with tab3:
    st.subheader("Stored Prediction Records")
    st.write("Browse previously generated salary predictions stored in Supabase.")

    if df.empty:
        st.warning("No stored records are available.")
    else:
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            selected_experience = st.selectbox(
                "Filter by Experience Level",
                ["All"] + sorted(df["experience_level"].dropna().unique().tolist())
            )

        with filter_col2:
            selected_job = st.selectbox(
                "Filter by Job Title",
                ["All"] + sorted(df["job_title"].dropna().unique().tolist())
            )

        filtered_df = df.copy()

        if selected_experience != "All":
            filtered_df = filtered_df[filtered_df["experience_level"] == selected_experience]

        if selected_job != "All":
            filtered_df = filtered_df[filtered_df["job_title"] == selected_job]

        st.dataframe(filtered_df, use_container_width=True)

        if not filtered_df.empty:
            latest = filtered_df.iloc[0]

            with st.expander("View Latest Selected Record", expanded=True):
                st.write({
                    "Experience Level": latest["experience_level"],
                    "Employment Type": latest["employment_type"],
                    "Job Title": latest["job_title"],
                    "Company Size": latest["company_size"],
                    "Employee Residence": latest["employee_residence"],
                    "Remote Ratio": latest["remote_ratio"],
                    "Company Location": latest["company_location"],
                    "Predicted Salary": latest["predicted_salary"]
                })

                st.markdown("### Stored Analysis")
                st.write(latest.get("llm_analysis", "No analysis available."))