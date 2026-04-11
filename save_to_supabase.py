import os
import math
import time
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

predictions_df = pd.read_csv("predictions.csv")

if os.path.exists("llm_analysis.txt"):
    with open("llm_analysis.txt", "r", encoding="utf-8") as f:
        llm_analysis = f.read().strip()
else:
    llm_analysis = """
Salary Prediction Overview

The generated salary predictions show variation across experience level, job title, company size,
employee residence, remote ratio, and company location.

Key Insights:
- Senior roles tend to receive higher salary predictions.
- Job title strongly affects predicted salary.
- Residence, location, and remote ratio may influence salary levels.

Warning:
These values are machine learning predictions and should be treated as estimates, not guaranteed real salaries.
""".strip()

chart_experience_path = "salary_chart.png"
chart_job_title_path = "salary_by_job_title.png"

records = []

for _, row in predictions_df.iterrows():
    records.append({
        "experience_level": row["experience_level"],
        "employment_type": row["employment_type"],
        "job_title": row["job_title"],
        "company_size": row["company_size"],
        "employee_residence": row["employee_residence"],
        "remote_ratio": int(row["remote_ratio"]),
        "company_location": row["company_location"],
        "predicted_salary": float(row["predicted_salary"]),
        "llm_analysis": llm_analysis,
        "chart_experience_path": chart_experience_path,
        "chart_job_title_path": chart_job_title_path
    })

if not records:
    raise ValueError("No records found in predictions.csv")

table_name = "salary_predictions"

# Delete all old rows first
try:
    supabase.table(table_name).delete().neq("experience_level", "").execute()
    print("Old records deleted successfully.")
except Exception as e:
    print(f"Failed to delete old records: {e}")
    raise

# Insert fresh rows in batches
batch_size = 500
total_batches = math.ceil(len(records) / batch_size)

for i in range(0, len(records), batch_size):
    batch = records[i:i + batch_size]
    batch_number = (i // batch_size) + 1

    success = False
    for attempt in range(3):
        try:
            supabase.table(table_name).insert(batch).execute()
            print(f"Inserted batch {batch_number}/{total_batches} ({len(batch)} records)")
            success = True
            break
        except Exception as e:
            print(f"Batch {batch_number} failed on attempt {attempt + 1}: {e}")
            time.sleep(2)

    if not success:
        raise Exception(f"Failed to insert batch {batch_number} after 3 attempts")

print("Finished replacing Supabase table data.")