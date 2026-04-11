import pandas as pd

df = pd.read_csv("predictions.csv")

print("Rows:", len(df))
print("Unique job titles:", df["job_title"].nunique())
print("Job titles:", df["job_title"].unique())
print("Employment types:", df["employment_type"].unique())
print("Company sizes:", df["company_size"].unique())
print("Remote ratios:", df["remote_ratio"].unique())