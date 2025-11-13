import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# --- AQI Calculation (CPCB Standards) ---
# [This section is identical to the previous scripts]
# CPCB breakpoints for sub-index calculation
AQI_BREAKPOINTS = {
    'pm2_5': [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 10000)],
    'pm10': [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 10000)],
    'no2': [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 10000)],
    'so2': [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 10000)],
    'co': [(0, 1.0), (1.1, 2.0), (2.1, 10.0), (10.1, 17.0), (17.1, 34.0), (34.1, 10000)],
    'o3': [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 10000)],
}
# AQI "Good" to "Severe" categories
AQI_CATEGORIES = [(0, 50), (51, 100), (101, 200), (201, 300), (301, 400), (401, 500)]

def get_sub_index(value, pollutant):
    """Calculates the sub-index for a single pollutant."""
    if pd.isna(value):
        return np.nan
    try:
        breakpoints = AQI_BREAKPOINTS[pollutant]
    except KeyError:
        print(f"Warning: No AQI breakpoints defined for {pollutant}")
        return np.nan

    for i, (low, high) in enumerate(breakpoints):
        if low <= value <= high:
            aqi_low, aqi_high = AQI_CATEGORIES[i]
            # Linear interpolation formula
            sub_index = ((aqi_high - aqi_low) / (high - low)) * (value - low) + aqi_low
            return sub_index
    
    if value > breakpoints[-1][1]:
        return 500
    return 0

def calculate_aqi_from_pollutants(row):
    """Calculates the final AQI from pre-calculated rolling averages."""
    sub_indices = [
        get_sub_index(row.get('pm2_5_24h_avg'), 'pm2_5'),
        get_sub_index(row.get('pm10_24h_avg'), 'pm10'),
        get_sub_index(row.get('no2_24h_avg'), 'no2'),
        get_sub_index(row.get('so2_24h_avg'), 'so2'),
        get_sub_index(row.get('co_8h_avg_mg'), 'co'),
        get_sub_index(row.get('o3_8h_avg'), 'o3'),
    ]
    valid_indices = [idx for idx in sub_indices if pd.notna(idx)]
    if not valid_indices:
        return np.nan
    return max(valid_indices)

# --- LSTM Model Definition ---

def build_lstm_model(seq_len, num_features, forecast_horizon, lstm_units=64, dropout_rate=0.1):
    """Builds and compiles a stacked LSTM model."""
    
    inputs = layers.Input(shape=(seq_len, num_features))
    
    # First LSTM layer - return_sequences=True to feed the next LSTM layer
    x = layers.LSTM(lstm_units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    
    # Second LSTM layer - return_sequences=False to get the last output
    x = layers.LSTM(lstm_units, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)
    
    # Output Head
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer with N units for N forecast steps
    outputs = layers.Dense(forecast_horizon)(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="mse", # Mean Squared Error
        metrics=["mae"] # Mean Absolute Error
    )
    return model

# --- Helper function for creating sequences ---
def create_sequences(data_features, data_target, seq_length, forecast_horizon):
    """
    Creates sequences of (X, y) for time-series forecasting.
    X shape: (n_samples, seq_length, n_features)
    y shape: (n_samples, forecast_horizon)
    """
    X, y = [], []
    for i in range(len(data_features) - seq_length - forecast_horizon + 1):
        # Input sequence (past data)
        X.append(data_features[i:(i + seq_length)])
        # Output sequence (future data to predict)
        y.append(data_target[(i + seq_length):(i + seq_length + forecast_horizon)])
    return np.array(X), np.array(y)

# --- Main Model Training and Prediction ---

def main():
    """Main function to run the full pipeline."""
    
    # --- 1. Load and Preprocess Data (Same as before) ---
    print("Loading data...")
    try:
        df = pd.read_csv('jharia.csv', skiprows=3)
    except FileNotFoundError:
        print("Error: 'jharia.csv' not found. Please upload it to your Colab session.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    df.columns = ['time', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co2', 'co', 'dust', 'aod']
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')
    df = df.ffill().bfill()
    if df.isnull().values.any():
        df = df.dropna()
    print("Data loaded and preprocessed.")

    # --- 2. Calculate AQI (Target Variable) (Same as before) ---
    print("Calculating historical AQI...")
    df['pm2_5_24h_avg'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    df['pm10_24h_avg'] = df['pm10'].rolling(window=24, min_periods=1).mean()
    df['no2_24h_avg'] = df['no2'].rolling(window=24, min_periods=1).mean()
    df['so2_24h_avg'] = df['so2'].rolling(window=24, min_periods=1).mean()
    df['o3_8h_avg'] = df['o3'].rolling(window=8, min_periods=1).mean()
    df['co_8h_avg_mg'] = (df['co'].rolling(window=8, min_periods=1).mean()) / 1000.0
    df['AQI'] = df.apply(calculate_aqi_from_pollutants, axis=1)
    df = df.dropna()
    if df.empty:
        print("Error: No data remaining after AQI calculation.")
        return
    print("AQI calculation complete.")

    # --- 3. Feature Engineering (Same as Transformer) ---
    print("Engineering features for LSTM...")
    
    # Add time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # Define all features to be used
    all_features = ['AQI', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co2', 'co', 'dust', 'aod', 'hour_of_day', 'day_of_week', 'month']
    target_col = 'AQI'
    n_features = len(all_features)
    
    # Define sequence lengths
    N_LAGS = 24      # Use last 24 hours of data
    N_FORECAST = 72  # Predict next 72 hours
    
    # Split data chronologically (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # **Scale the data** (CRITICAL for neural networks)
    # Feature scaler (for all input columns)
    feature_scaler = StandardScaler()
    train_features_scaled = feature_scaler.fit_transform(train_df[all_features])
    test_features_scaled = feature_scaler.transform(test_df[all_features])
    
    # Target scaler (for the AQI column only)
    target_scaler = StandardScaler()
    train_target_scaled = target_scaler.fit_transform(train_df[[target_col]])
    test_target_scaled = target_scaler.transform(test_df[[target_col]])

    print("Creating sequences...")
    # Create training sequences
    X_train, y_train = create_sequences(
        train_features_scaled, 
        train_target_scaled.ravel(), # flatten target
        N_LAGS, 
        N_FORECAST
    )
    
    # Create test sequences
    X_test, y_test_scaled = create_sequences(
        test_features_scaled, 
        test_target_scaled.ravel(), 
        N_LAGS, 
        N_FORECAST
    )

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Error: Not enough data to create training/test sequences. Try a smaller N_LAGS or N_FORECAST.")
        return
        
    print(f"Feature engineering complete. Training on {len(X_train)} sequences.")

    # --- 4. Build and Train LSTM Model ---
    print("Building LSTM model...")
    model = build_lstm_model(
        seq_len=N_LAGS,
        num_features=n_features,
        forecast_horizon=N_FORECAST
    )
    model.summary()

    # Callback for early stopping
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=5, # Stop if val_loss doesn't improve for 5 epochs
        restore_best_weights=True
    )
    
    print("Training LSTM model... (This may take a few minutes)")
    history = model.fit(
        X_train,
        y_train,
        epochs=30, # Max epochs
        batch_size=32,
        validation_split=0.2, # Use 20% of training data for validation
        callbacks=[early_stopping],
        verbose=1
    )
    print("Model training complete.")

    # --- 5. Evaluate Model ---
    print("\nEvaluating model on test data...")
    # Predict on test data (scaled)
    y_pred_scaled = model.predict(X_test)
    
    # **Inverse transform predictions** to get back to original AQI values
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test = target_scaler.inverse_transform(y_test_scaled)
    
    # Evaluate performance for t+1 and t+72
    y_test_t1 = y_test[:, 0]
    y_pred_t1 = y_pred[:, 0]
    
    y_test_t72 = y_test[:, -1]
    y_pred_t72 = y_pred[:, -1]

    me_t1 = np.mean(y_test_t1 - y_pred_t1)
    rmse_t1 = np.sqrt(mean_squared_error(y_test_t1, y_pred_t1))
    
    me_t72 = np.mean(y_test_t72 - y_pred_t72)
    rmse_t72 = np.sqrt(mean_squared_error(y_test_t72, y_pred_t72))

    print("\n--- Model Evaluation (Inverse-Transformed) ---")
    print(f"Forecast (t+1 hour)   - Mean Error: {me_t1:.2f}, RMSE: {rmse_t1:.2f}")
    print(f"Forecast (t+72 hours) - Mean Error: {me_t72:.2f}, RMSE: {rmse_t72:.2f}")

    # --- 6. Data Fitting Map (Plot) ---
    print("\nGenerating data fitting map...")
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test_t1, y_pred_t1, alpha=0.5, label='t+1 Forecast')
    
    min_val = min(y_test_t1.min(), y_pred_t1.min())
    max_val = max(y_test_t1.max(), y_pred_t1.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    plt.title('Data Fitting Map: Actual vs. Predicted AQI (t+1 hour)')
    plt.xlabel('Actual AQI (Inverse-Transformed)')
    plt.ylabel('Predicted AQI (Inverse-Transformed)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aqi_lstm_actual_vs_predicted.png')
    print("Plot saved to 'aqi_lstm_actual_vs_predicted.png'")

    # --- 7. Forecast Next 72 Hours ---
    print("\n--- AQI Forecast for Next 72 Hours ---")
    
    # Get the last N_LAGS hours of data from the *full* dataset
    last_sequence = df[all_features].iloc[-N_LAGS:]
    
    # Scale this data using the *fitted* feature_scaler
    last_sequence_scaled = feature_scaler.transform(last_sequence)
    
    # Reshape to (1, seq_len, n_features) for the model
    last_sequence_scaled = last_sequence_scaled.reshape(1, N_LAGS, n_features)

    # Predict the next 72 values (scaled)
    forecast_scaled = model.predict(last_sequence_scaled)
    
    # Inverse transform the forecast
    forecast = target_scaler.inverse_transform(forecast_scaled)
    
    # Flatten the result
    future_72_hours_aqi = forecast.flatten()

    # Get the last timestamp from the original data
    last_timestamp = df.index[-1]
    
    # Create the date range for the forecast
    forecast_dates = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=N_FORECAST,
        freq='h'
    )

    # Create a DataFrame to display the forecast
    forecast_df = pd.DataFrame({
        'Timestamp': forecast_dates,
        'Predicted_AQI': future_72_hours_aqi
    })
    
    forecast_df['Predicted_AQI'] = forecast_df['Predicted_AQI'].round(2)
    
    # Print the forecast
    with pd.option_context('display.max_rows', None):
        print(forecast_df)

if __name__ == "__main__":
    main()