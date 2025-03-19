import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load datasets
features_df = pd.read_csv('stock_features.csv', header=None).iloc[1:, 1:].astype(float)
targets_df = pd.read_csv('stock_targets.csv', header=None).iloc[1:, 1:].astype(float)

test_df = pd.read_csv('test_features.csv', header=None).iloc[1:, 1:].astype(float)
ans_df = pd.read_csv('test_targets.csv', header=None).iloc[1:, 1:].astype(float)
name = pd.read_csv('test_features.csv', header=None).iloc[1:, 0:1].to_numpy()

# Extract feature matrix (X) and target values (y)
X = features_df.to_numpy()
y = targets_df.to_numpy()

# Apply K-Means clustering
n_clusters = 5  # You can tune this hyperparameter
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(X)

# Predict clusters for training and test sets
train_clusters = kmeans.predict(X)
test_clusters = kmeans.predict(test_df)

# Predict target using mean value of each cluster
cluster_means = [y[train_clusters == i].mean() for i in range(n_clusters)]
predictions = np.array([cluster_means[c] for c in test_clusters])

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(ans_df, predictions))
mae = mean_absolute_error(ans_df, predictions)
r2 = r2_score(ans_df, predictions)

print("K-Means model trained and evaluated!")
print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

for i, stock in enumerate(ans_df.values):
    print(f"{name[i][0]}: Predicted = {predictions[i]:.2f}, Actual = {ans_df.iloc[i].item():.2f}")

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

dates = pd.date_range(start='2024-01-01', periods=92, freq='D')
# First graph: Plot continuous stock price lines
for i in range(len(test_df)):
    ax1.plot(dates, test_df.iloc[i, :92], label=name[i][0])

ax1.set_title('Stock Price Change Over 2024 (Test Data)')
ax1.set_xlabel('Date')
ax1.set_ylabel('Stock Price')
ax1.grid(True)

# Second graph: Plot target and prediction for each stock
for i in range(len(test_df)):
    # Plot target value as a horizontal line
    ax2.hlines(ans_df.iloc[i, 0], i - 0.4, i + 0.4, colors='tab:blue')  # Horizontal line for target
    # Plot prediction as an 'x' marker
    ax2.scatter(i, predictions[i], color='red', marker='x')  # 'x' for prediction


# Set the x-axis to show the stock names
ax2.set_xticks(np.arange(len(test_df)))
ax2.set_xticklabels([name[i][0] for i in range(len(test_df))], rotation=90, fontsize=8)
ax2.set_title('Target vs. Prediction for Each Stock')
ax2.set_xlabel('Stock Index')
ax2.set_ylabel('Stock Price')
ax2.grid(True)

# Add legends in both subplots
ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
ax2.legend(['Target Value', 'Prediction'], loc='upper left', fontsize=8)

plt.tight_layout()
plt.show()