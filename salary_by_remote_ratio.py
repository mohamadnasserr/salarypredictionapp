import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("predictions.csv")

summary = df.groupby("remote_ratio", as_index=False)["predicted_salary"].mean()

plt.figure(figsize=(8, 5))
plt.bar(summary["remote_ratio"], summary["predicted_salary"])
plt.title("Average Predicted Salary by Remote Ratio")
plt.xlabel("Remote Ratio")
plt.ylabel("Predicted Salary (USD)")
plt.tight_layout()

plt.savefig("remote_ratio_chart.png")
plt.show()

print("Chart saved as remote_ratio_chart.png")