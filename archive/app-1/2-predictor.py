import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Function to write to a TXT file
def debugger(name, data):
    with open('data/debug.txt', 'a') as f:
        f.write(str(name) + '\n' + str(data) + '\n'*3)


# Load the dataset
# Parsing the 'DateTime' column as dates
cpu_data = pd.read_csv('data/fabricated_data.csv', parse_dates=['DateTime'])
debugger('cpu_data', cpu_data)

# Prepare the data (use past 24 hours of CPU usage to predict the next hour)
cpu_values = cpu_data['CPU_Usage'].values.reshape(-1, 1)  # Reshaping the data to a 2D array
scaler = MinMaxScaler()  # Using MinMaxScaler to normalize the CPU usage values between 0 and 1
cpu_values_scaled = scaler.fit_transform(cpu_values)  # Scale the values

debugger('cpu_values', cpu_values)
debugger('scaler', scaler)
debugger('cpu_values_scaled', cpu_values_scaled)

X = []  # This will hold the past 24 hours of data
y = []  # This will hold the next hour's data (target variable)

window_size = 24  # We're using the past 24 hours to predict the next hour
for i in range(len(cpu_values_scaled) - window_size):
    X.append(cpu_values_scaled[i:i+window_size])  # Append the past 24 hours to X
    y.append(cpu_values_scaled[i+window_size])  # Append the next hour's value to y

debugger('X', X)
debugger('y', y)

X = np.array(X)  # Convert the list to a NumPy array for input to the model
y = np.array(y)  # Convert the list to a NumPy array for the output

debugger('X', X)
debugger('y', y)

# Split the data into training and test sets (80% train, 20% test)
# shuffle=False ensures the data remains sequential
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

debugger('X_train', X_train)
debugger('X_test', X_test)
debugger('y_train', y_train)
debugger('y_test', y_test)

# Build a simple LSTM (Long Short-Term Memory) model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(window_size, 1), return_sequences=True),  # First LSTM layer with 64 units
    tf.keras.layers.LSTM(32),  # Second LSTM layer with 32 units, no return_sequences as it's the last LSTM
    tf.keras.layers.Dense(1)  # Dense layer to predict a single value (next hour's CPU usage)
])

# Compile the model using the Adam optimizer and Mean Squared Error loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
# - epochs=10: The model will go through the data 10 times
# - batch_size=16: Model will process 16 sequences at a time
# - validation_data=(X_test, y_test): Use the test set for validation
model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

# Function to predict future CPU usage values
def predict_future(cpu_values, model, window_size, steps):
    """
    Predicts future CPU usage values based on past observations using a trained model.

    Args:
    cpu_values (numpy.ndarray): Scaled array containing past CPU usage data.
    model (tf.keras.Model): Trained LSTM model for predicting future values.
    window_size (int): Number of past observations to use for each prediction.
    steps (int): Number of future time steps to predict.

    Returns:
    numpy.ndarray: Array of predicted future CPU usage values.
    """
    future_predictions = []  # List to hold the predicted values
    last_values = cpu_values[-window_size:].copy()  # Start with the last observed values (last 24 hours)

    debugger('last_values', last_values)

    for _ in range(steps):  # Predict multiple future steps (e.g., 24 hours)
        prediction = model.predict(last_values.reshape(1, window_size, 1))  # Predict the next hour's CPU usage
        future_predictions.append(prediction[0, 0])  # Append the prediction

        # Shift the window: remove the oldest value and append the predicted value
        last_values = np.append(last_values[1:], prediction[0, 0])

    return np.array(future_predictions)  # Return the predictions as a NumPy array

# Predict the next 24 hours of CPU usage
next_week_predictions = predict_future(cpu_values_scaled, model, window_size, steps=24)

debugger('next_week_predictions', next_week_predictions)

# Rescale the predicted values back to the original CPU usage scale
next_week_predictions_rescaled = scaler.inverse_transform(next_week_predictions.reshape(-1, 1))

debugger('next_week_predictions_rescaled', next_week_predictions_rescaled)

# Create a DataFrame for the predictions with DateTime for the next 24 hours
future_dates = pd.date_range(start=cpu_data['DateTime'].max() + pd.Timedelta(hours=1), periods=24, freq='h')
predicted_cpu_data = pd.DataFrame({'DateTime': future_dates, 'Predicted_CPU_Usage': next_week_predictions_rescaled.flatten()})

debugger('future_dates', future_dates)
debugger('predicted_cpu_data', predicted_cpu_data)

# Save the predictions to a CSV file
predicted_cpu_data.to_csv('data/predicted_data.csv', index=False)

# Print confirmation of prediction completion
print(f"Done predicting CPU usage! \nPredicted {len(predicted_cpu_data)} rows.")
