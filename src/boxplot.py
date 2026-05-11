from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load California Housing dataset
housing = fetch_california_housing(as_frame=True)

# Features + target as a single DataFrame
df = housing.frame

# Quick check
print(df.head())
print(df.shape)

# Plot boxplot
fig, ax = plt.subplots(figsize=(12, 6))
df.boxplot(ax=ax)
ax.set_title("California Housing Dataset — Feature Distributions")
ax.set_xlabel("Features")
ax.set_ylabel("Values")
plt.xticks(rotation=45)
plt.tight_layout()

# Save the figure
plt.savefig("figs/boxplot.png", dpi=150)
print("Boxplot saved to figs/boxplot.png")
plt.show()
