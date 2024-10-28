import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
cpu_data = pd.read_csv('data/fabricated_data.csv', parse_dates=['DateTime'])

# Prepare the data for both CPU and Disk usage (use past 24 hours to predict the next hour)
cpu_values = cpu_data['CPU_Usage'].values.reshape(-1, 1)
disk_values = cpu_data['Disk_Usage'].values.reshape(-1, 1)

# Normalize the CPU and Disk usage values
scaler_cpu = MinMaxScaler()
scaler_disk = MinMaxScaler()

cpu_values_scaled = scaler_cpu.fit_transform(cpu_values)
disk_values_scaled = scaler_disk.fit_transform(disk_values)

# Combine CPU and Disk values
combined_values_scaled = np.hstack((cpu_values_scaled, disk_values_scaled))

X = []
y = []  # Combined target array for CPU and Disk

window_size = 24
for i in range(len(combined_values_scaled) - window_size):
    X.append(combined_values_scaled[i:i+window_size])
    y.append(combined_values_scaled[i+window_size])  # Next hour's CPU and Disk usage

X = np.array(X)
y = np.array(y)  # Convert y into a NumPy array with shape (batch_size, 2)

# Split the data into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build an LSTM model to predict both CPU and Disk usage
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(window_size, 2), return_sequences=True),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(2)  # Predict two values: CPU and Disk usage
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Function to predict future values for both CPU and Disk usage
def predict_future(combined_values, model, window_size, steps):
    future_predictions = []
    last_values = combined_values[-window_size:].copy()

    for _ in range(steps):
        prediction = model.predict(last_values.reshape(1, window_size, 2))
        future_predictions.append(prediction[0])  # Append both CPU and Disk predictions

        # Shift the window and append the new predictions
        last_values = np.vstack((last_values[1:], prediction[0]))

    return np.array(future_predictions)

# Predict the next 24 hours for both CPU and Disk usage
steps = 24
next_predictions = predict_future(combined_values_scaled, model, window_size, steps)

# Rescale the predictions back to the original values
next_cpu_predictions_rescaled = scaler_cpu.inverse_transform(next_predictions[:, 0].reshape(-1, 1))
next_disk_predictions_rescaled = scaler_disk.inverse_transform(next_predictions[:, 1].reshape(-1, 1))

# Create a DataFrame for the predictions
future_dates = pd.date_range(start=cpu_data['DateTime'].max() + pd.Timedelta(hours=1), periods=steps, freq='h')
predicted_data = pd.DataFrame({
    'DateTime': future_dates,
    'Predicted_CPU_Usage': next_cpu_predictions_rescaled.flatten(),
    'Predicted_Disk_Usage': next_disk_predictions_rescaled.flatten()
})

# Save the predictions to a CSV file
predicted_data.to_csv('data/predicted_data.csv', index=False)

print(f"Done predicting CPU and Disk usage! \nPredicted {len(predicted_data)} rows.")
