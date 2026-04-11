import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

summary = df.groupby("experience_level", as_index=False)["predicted_salary"].mean()

plt.figure(figsize=(8, 5))
plt.bar(summary["experience_level"], summary["predicted_salary"])
plt.title("Average Predicted Salary by Experience Level")
plt.xlabel("Experience Level")
plt.ylabel("Predicted Salary (USD)")
plt.tight_layout()

plt.savefig("chart_by_experience.png")
plt.show()

print("Chart saved as chart_by_experience.png")