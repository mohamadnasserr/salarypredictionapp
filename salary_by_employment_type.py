import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

summary = df.groupby("employment_type", as_index=False)["predicted_salary"].mean()

plt.figure(figsize=(8, 5))
plt.bar(summary["employment_type"], summary["predicted_salary"])
plt.title("Average Predicted Salary by Employment Type")
plt.xlabel("Employment Type")
plt.ylabel("Predicted Salary (USD)")
plt.tight_layout()

plt.savefig("employment_type_chart.png")
plt.show()

print("Chart saved as employment_type_chart.png")