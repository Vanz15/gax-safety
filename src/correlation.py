import pandas as pd

# Create DataFrame
df = pd.DataFrame({
    "Model": ["v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9"],
    "Accuracy": [93.14, 91.73, 92.60, 92.47, 91.86, 92.33, 92.13, 90.99],
    "Recall": [88.70, 89.37, 87.87, 85.22, 87.04, 88.54, 86.71, 91.36],
    "Cheating_Rate": [93.81, 93.88, 91.5, 92.8, 89.9, 89.44, 92.6, 89.44],
    "Avg_Cheating": [0.7426, 0.7229, 0.7230, 0.7257, 0.7088, 0.72, 0.7316, 0.7065]
})

# Set Model as index (optional but cleaner)
df.set_index("Model", inplace=True)

# Compute Pearson correlation matrix
corr_matrix = df.corr(method="pearson")

print("=== Pearson Correlation Matrix ===")
print(corr_matrix)

# Select only relevant correlations
selected_corr = {
    "Accuracy vs Cheating Rate": df["Accuracy"].corr(df["Cheating_Rate"]),
    "Accuracy vs Avg Cheating": df["Accuracy"].corr(df["Avg_Cheating"]),
    "Recall vs Cheating Rate": df["Recall"].corr(df["Cheating_Rate"]),
    "Recall vs Avg Cheating": df["Recall"].corr(df["Avg_Cheating"])
}

print("\n=== Key Correlations ===")
for k, v in selected_corr.items():
    print(f"{k}: {v:.4f}")

import matplotlib.pyplot as plt

def plot_scatter(x, y, x_label, y_label):
    plt.figure()
    plt.scatter(df[x], df[y])
    
    # Add labels per point
    for i, model in enumerate(df.index):
        plt.text(df[x][i], df[y][i], model)
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"{x_label} vs {y_label}")
    plt.grid(True)
    plt.show()

# Generate plots
plot_scatter("Accuracy", "Cheating_Rate", "Accuracy", "Cheating Rate")
plot_scatter("Accuracy", "Avg_Cheating", "Accuracy", "Avg Cheating Score")
plot_scatter("Recall", "Cheating_Rate", "Recall", "Cheating Rate")
plot_scatter("Recall", "Avg_Cheating", "Recall", "Avg Cheating Score")

import numpy as np

def plot_with_trendline(x, y):
    plt.figure()
    plt.scatter(df[x], df[y])

    # Trendline
    z = np.polyfit(df[x], df[y], 1)
    p = np.poly1d(z)
    plt.plot(df[x], p(df[x]))

    for i, model in enumerate(df.index):
        plt.text(df[x][i], df[y][i], model)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} (with Trendline)")
    plt.grid(True)
    plt.show()

plot_with_trendline("Accuracy", "Cheating_Rate")
plot_with_trendline("Accuracy", "Avg_Cheating")
plot_with_trendline("Recall", "Cheating_Rate")
plot_with_trendline("Recall", "Avg_Cheating")