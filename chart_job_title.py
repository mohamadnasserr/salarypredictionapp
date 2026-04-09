import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

summary = (
    df.groupby("job_title", as_index=False)["predicted_salary"]
    .mean()
    .sort_values("predicted_salary", ascending=False)
)

plt.figure(figsize=(10, 6))
plt.bar(summary["job_title"], summary["predicted_salary"])
plt.title("Average Predicted Salary by Job Title")
plt.xlabel("Job Title")
plt.ylabel("Predicted Salary (USD)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("salary_by_job_title.png")
plt.show()

print("Chart saved as salary_by_job_title.png")