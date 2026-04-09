import pandas as pd

# Load data
df = pd.read_csv("ds_salaries.csv")

# 1. Drop duplicates
df = df.drop_duplicates()

# 2. Keep only relevant columns
df = df[[
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "employee_residence",
    "remote_ratio",
    "company_location",
    "salary_in_usd"
]]

# 3. Drop rows where target is missing
df = df.dropna(subset=["salary_in_usd"])

# 4. Fill missing values in features
categorical_columns = [
    "experience_level",
    "employment_type",
    "job_title",
    "company_size",
    "employee_residence",
    "company_location"
]

for col in categorical_columns:
    df[col] = df[col].fillna("Unknown").astype(str).str.strip()

df["remote_ratio"] = df["remote_ratio"].fillna(df["remote_ratio"].median())

# 5. One-hot encode categorical variables
df = pd.get_dummies(df, columns=categorical_columns)

# 6. Convert boolean columns to integers
bool_columns = df.select_dtypes(include="bool").columns
df[bool_columns] = df[bool_columns].astype(int)

# 7. Final check
print(df.head())
print(df.columns.tolist())

# 8. Save cleaned file
df.to_csv("cleaned_data.csv", index=False)
print("cleaned_data.csv saved successfully")