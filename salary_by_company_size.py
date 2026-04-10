import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

summary = df.groupby("company_size", as_index=False)["predicted_salary"].mean()

plt.figure(figsize=(8, 5))
plt.bar(summary["company_size"], summary["predicted_salary"])
plt.title("Average Predicted Salary by Company Size")
plt.xlabel("Company Size")
plt.ylabel("Predicted Salary (USD)")
plt.tight_layout()

plt.savefig("company_size_chart.png")
plt.show()

print("Chart saved as company_size_chart.png")