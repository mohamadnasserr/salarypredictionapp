from google import genai
import pandas as pd
import time

# Load prediction results
df = pd.read_csv("predictions.csv")

client = genai.Client()

average_salary = df["predicted_salary"].mean()
max_salary = df["predicted_salary"].max()
min_salary = df["predicted_salary"].min()

highest_row = df.loc[df["predicted_salary"].idxmax()].to_dict()
lowest_row = df.loc[df["predicted_salary"].idxmin()].to_dict()

prompt = f"""
You are a data analyst.

I have salary prediction results from a machine learning model.

Summary:
- Total predictions: {len(df)}
- Average salary: {average_salary:.2f}
- Highest salary: {max_salary:.2f}
- Lowest salary: {min_salary:.2f}

Highest case:
{highest_row}

Lowest case:
{lowest_row}

Write:
1. A short title
2. A paragraph explanation
3. Three insights (bullet points)
4. A short warning about model predictions
"""

analysis_text = None

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        analysis_text = response.text
        break
    except Exception as e:
        print(f"Attempt {attempt + 1} failed: {e}")
        time.sleep(5)

# Fallback text if Gemini fails
if not analysis_text:
    analysis_text = f"""
Salary Prediction Overview

The generated salary predictions show an average predicted salary of {average_salary:.2f} USD, with the highest prediction reaching {max_salary:.2f} USD and the lowest prediction at {min_salary:.2f} USD. This suggests that salary levels vary meaningfully across combinations of experience level, job title, company size, remote ratio, employee residence, and company location.

Key Insights:
- Higher experience levels generally tend to correspond to higher predicted salaries.
- Job title and geographic factors appear to play a major role in salary variation.
- Company size and remote ratio may also influence salary outcomes depending on the role and region.

Warning:
These values are predictions from a machine learning model and should be treated as estimates, not guaranteed real-world salaries.
""".strip()

print("\nLLM ANALYSIS:\n")
print(analysis_text)

with open("llm_analysis.txt", "w", encoding="utf-8") as f:
    f.write(analysis_text)

print("\nSaved to llm_analysis.txt")