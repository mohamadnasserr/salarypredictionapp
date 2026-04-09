import requests
import itertools
import pandas as pd

try:
    test = requests.get("http://127.0.0.1:8000/", timeout=5)
    print("API is running:", test.status_code)
except requests.exceptions.RequestException:
    print("API is not running. Start it first with: uvicorn api:app --reload")
    raise SystemExit

url = "http://127.0.0.1:8000/predict"

# Load raw salary file to collect job titles
raw_df = pd.read_csv("ds_salaries.csv")

experience_levels = ["SE", "MI"]
employment_types = ["FT"]
job_titles = raw_df["job_title"].value_counts().head(8).index.tolist()
company_sizes = ["M", "L"]
employee_residences = ["US", "DE"]
remote_ratios = [0, 100]
company_locations = ["US", "DE"]

results = []

for combo in itertools.product(
    experience_levels,
    employment_types,
    job_titles,
    company_sizes,
    employee_residences,
    remote_ratios,
    company_locations
):
    params = {
        "experience_level": combo[0],
        "employment_type": combo[1],
        "job_title": combo[2],
        "company_size": combo[3],
        "employee_residence": combo[4],
        "remote_ratio": combo[5],
        "company_location": combo[6]
    }

    try:
        response = requests.get(url, params=params, timeout=15)

        if response.status_code == 200:
            data = response.json()
            results.append(data)
            print("Success:", data["predicted_salary"], "|", data["job_title"])
        else:
            print("Error:", response.status_code, response.text)

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

df = pd.DataFrame(results)
df.to_csv("predictions.csv", index=False)

print(f"\nSaved {len(df)} predictions to predictions.csv")