import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
# [This section is identical to the previous script]
AQI_BREAKPOINTS = {
    'pm2_5': [(0, 30), (31, 60), (61, 90), (91, 120), (121, 250), (251, 10000)],
    'pm10': [(0, 50), (51, 100), (101, 250), (251, 350), (351, 430), (431, 10000)],
    'no2': [(0, 40), (41, 80), (81, 180), (181, 280), (281, 400), (401, 10000)],
    'so2': [(0, 40), (41, 80), (81, 380), (381, 800), (801, 1600), (1601, 10000)],
    'co': [(0, 1.0), (1.1, 2.0), (2.1, 10.0), (10.1, 17.0), (17.1, 34.0), (34.1, 10000)],
    'o3': [(0, 50), (51, 100), (101, 168), (169, 208), (209, 748), (749, 10000)],
}
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

# --- Transformer Model Definition ---
# [This section is identical to the previous script]

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = layers.Dense(embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)
        self.maxlen = maxlen
        self.embed_dim = embed_dim

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "maxlen": self.maxlen,
            "vocab_size": 0,
            "embed_dim": self.embed_dim,
        })
        return config

def transformer_encoder(inputs, embed_dim, num_heads, ff_dim, rate=0.1):
    attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attention = layers.Dropout(rate)(attention)
    attention = layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
    ffn = layers.Dense(ff_dim, activation="relu")(attention)
    ffn = layers.Dropout(rate)(ffn)
    ffn = layers.Dense(embed_dim)(ffn)
    outputs = layers.LayerNormalization(epsilon=1e-6)(attention + ffn)
    return outputs

def build_transformer_model(seq_len, num_features, forecast_horizon, embed_dim=64, num_heads=4, ff_dim=64, num_transformer_blocks=2, dropout_rate=0.1):
    inputs = layers.Input(shape=(seq_len, num_features))
    x = TokenAndPositionEmbedding(maxlen=seq_len, vocab_size=0, embed_dim=embed_dim)(inputs)
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, embed_dim, num_heads, ff_dim, dropout_rate)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(forecast_horizon)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
    return model

# --- Helper function for creating sequences ---
def create_sequences(data_features, data_target, seq_length, forecast_horizon):
    """Creates sequences of (X, y) for time-series forecasting."""
    X, y = [], []
    for i in range(len(data_features) - seq_length - forecast_horizon + 1):
        X.append(data_features[i:(i + seq_length)])
        y.append(data_target[(i + seq_length):(i + seq_length + forecast_horizon)])
    return np.array(X), np.array(y)

# --- NEW: Function to load and tag data ---
def load_all_city_data(city_files):
    """Loads data from a list of CSV files and tags them with a 'city' column."""
    all_dfs = []
    print("Loading data for 4 cities...")
    for city_file in city_files:
        city_name = city_file.split('.')[0] # Use filename as city name
        try:
            df = pd.read_csv(city_file, skiprows=3)
        except FileNotFoundError:
            print(f"Error: '{city_file}' not found. Please upload it and update the filename.")
            return None
        except Exception as e:
            print(f"Error loading {city_file}: {e}")
            return None
            
        df.columns = ['time', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co2', 'co', 'dust', 'aod']
        df['time'] = pd.to_datetime(df['time'])
        df = df.set_index('time')
        df['city'] = city_name # Add the city tag
        all_dfs.append(df)
        print(f"Loaded {city_file} (as '{city_name}')...")

    if not all_dfs:
        print("No data was loaded.")
        return None
        
    # Combine all city data into one DataFrame
    full_df = pd.concat(all_dfs)
    full_df = full_df.ffill().bfill()
    if full_df.isnull().values.any():
        full_df = full_df.dropna()
    print("All city data loaded and preprocessed.")
    return full_df

# --- Main Model Training and Prediction ---

def main():
    """Main function to run the full pipeline."""
    
    # --- 1. Load and Preprocess Data ---
    
    # !!! IMPORTANT: Update this list with your 4 city filenames !!!
    # Make sure one of them is 'jharia.csv'
    city_files = ['jharia.csv', 'dhanbad.csv', 'raipur.csv', 'nashik.csv']
    print("="*50)
    print("Please make sure you have uploaded all 4 city CSV files and")
    print(f"updated the 'city_files' list in this script.")
    print("One of these files MUST be 'jharia.csv'.")
    print("="*50)

    df = load_all_city_data(city_files)
    if df is None:
        return

    # --- 2. Calculate AQI (Target Variable) ---
    print("Calculating historical AQI for all cities...")
    
    # NEW: Sort the DataFrame to ensure rolling operations work correctly
    # We must reset the index to sort by 'time', then set it back.
    df.reset_index(inplace=True)
    df.sort_values(by=['city', 'time'], inplace=True)
    df.set_index('time', inplace=True)
    print("DataFrame sorted by city and time.")

    # MODIFIED: Assign .values from the groupby operation
    # This works because the df is now sorted in the same order as the groupby
    df['pm2_5_24h_avg'] = df.groupby('city')['pm2_5'].rolling(window=24, min_periods=1).mean().values
    df['pm10_24h_avg'] = df.groupby('city')['pm10'].rolling(window=24, min_periods=1).mean().values
    df['no2_24h_avg'] = df.groupby('city')['no2'].rolling(window=24, min_periods=1).mean().values
    df['so2_24h_avg'] = df.groupby('city')['so2'].rolling(window=24, min_periods=1).mean().values
    df['o3_8h_avg'] = df.groupby('city')['o3'].rolling(window=8, min_periods=1).mean().values
    df['co_8h_avg_mg'] = (df.groupby('city')['co'].rolling(window=8, min_periods=1).mean().values) / 1000.0
    
    df['AQI'] = df.apply(calculate_aqi_from_pollutants, axis=1)
    df = df.dropna()
    if df.empty:
        print("Error: No data remaining after AQI calculation.")
        return
    print("AQI calculation complete.")

    # --- 3. Feature Engineering (Multi-City) ---
    print("Engineering features for Transformer...")
    
    # Add time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month

    # ** NEW: One-Hot Encode the 'city' column **
    city_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    city_one_hot = city_encoder.fit_transform(df[['city']])
    city_one_hot_df = pd.DataFrame(city_one_hot, columns=city_encoder.get_feature_names_out(['city']), index=df.index)
    df = pd.concat([df, city_one_hot_df], axis=1)

    # Define all features to be used
    base_features = ['AQI', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co2', 'co', 'dust', 'aod', 'hour_of_day', 'day_of_week', 'month']
    city_features = list(city_encoder.get_feature_names_out(['city']))
    all_features = base_features + city_features
    
    target_col = 'AQI'
    n_features = len(all_features)
    
    # Define sequence lengths
    N_LAGS = 24      # Use last 24 hours of data
    N_FORECAST = 72  # Predict next 72 hours
    
    # ** NEW: Split data by city first, then 80/20 **
    print("Splitting data into train/test sets (80/20 per city)...")
    train_dfs = []
    test_dfs = []
    for city in df['city'].unique():
        city_df = df[df['city'] == city]
        split_idx = int(len(city_df) * 0.8)
        train_dfs.append(city_df.iloc[:split_idx])
        test_dfs.append(city_df.iloc[split_idx:])
    
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    # **Scale the data**
    print("Scaling data...")
    feature_scaler = StandardScaler()
    train_df_scaled_features = feature_scaler.fit_transform(train_df[all_features])
    test_df_scaled_features = feature_scaler.transform(test_df[all_features])
    
    target_scaler = StandardScaler()
    train_df_scaled_target = target_scaler.fit_transform(train_df[[target_col]])
    test_df_scaled_target = target_scaler.transform(test_df[[target_col]])
    
    # Store scaled data back in DataFrames for easy grouping
    train_scaled_df = pd.DataFrame(train_df_scaled_features, columns=all_features, index=train_df.index)
    train_scaled_df[target_col] = train_df_scaled_target
    train_scaled_df['city'] = train_df['city']

    test_scaled_df = pd.DataFrame(test_df_scaled_features, columns=all_features, index=test_df.index)
    test_scaled_df[target_col] = test_df_scaled_target
    test_scaled_df['city'] = test_df['city']

    # ** NEW: Create sequences per-city **
    print("Creating sequences (grouped by city)...")
    X_train, y_train = [], []
    for city in train_scaled_df['city'].unique():
        city_data = train_scaled_df[train_scaled_df['city'] == city]
        X_c, y_c = create_sequences(
            city_data[all_features].values, 
            city_data[target_col].values.ravel(),
            N_LAGS, 
            N_FORECAST
        )
        X_train.append(X_c)
        y_train.append(y_c)
    
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    X_test, y_test_scaled = [], []
    for city in test_scaled_df['city'].unique():
        city_data = test_scaled_df[test_scaled_df['city'] == city]
        X_c, y_c = create_sequences(
            city_data[all_features].values, 
            city_data[target_col].values.ravel(),
            N_LAGS, 
            N_FORECAST
        )
        X_test.append(X_c)
        y_test_scaled.append(y_c)

    X_test = np.concatenate(X_test)
    y_test_scaled = np.concatenate(y_test_scaled)

    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("Error: Not enough data to create training/test sequences. Try a smaller N_LAGS or N_FORECAST.")
        return
        
    print(f"Feature engineering complete. Training on {len(X_train)} sequences from 4 cities.")

    # --- 4. Build and Train Transformer Model ---
    print("Building Transformer model...")
    model = build_transformer_model(
        seq_len=N_LAGS,
        num_features=n_features,
        forecast_horizon=N_FORECAST
    )
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    print("Training Transformer model... (This may take a few minutes)")
    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_split=0.2, # Validation split from the combined training data
        callbacks=[early_stopping],
        verbose=1
    )
    print("Model training complete.")

    # --- 5. Evaluate Model ---
    print("\nEvaluating model on test data...")
    y_pred_scaled = model.predict(X_test)
    
    y_pred = target_scaler.inverse_transform(y_pred_scaled)
    y_test = target_scaler.inverse_transform(y_test_scaled)
    
    y_test_t1 = y_test[:, 0]
    y_pred_t1 = y_pred[:, 0]
    y_test_t72 = y_test[:, -1]
    y_pred_t72 = y_pred[:, -1]

    me_t1 = np.mean(y_test_t1 - y_pred_t1)
    rmse_t1 = np.sqrt(mean_squared_error(y_test_t1, y_pred_t1))
    r2_t1 = r2_score(y_test_t1, y_pred_t1)
    
    me_t72 = np.mean(y_test_t72 - y_pred_t72)
    rmse_t72 = np.sqrt(mean_squared_error(y_test_t72, y_pred_t72))
    r2_t72 = r2_score(y_test_t72, y_pred_t72)

    print("\n--- Model Evaluation (Global, all cities) ---")
    print(f"Forecast (t+1 hour)   - Mean Error: {me_t1:.2f}, RMSE: {rmse_t1:.2f}, R-squared: {r2_t1:.2f}")
    print(f"Forecast (t+72 hours) - Mean Error: {me_t72:.2f}, RMSE: {rmse_t72:.2f}, R-squared: {r2_t72:.2f}")

    # --- 6. Data Fitting Map (Plot) ---
    print("\nGenerating data fitting map (global)...")
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test_t1, y_pred_t1, alpha=0.3, label='t+1 Forecast (All Cities)')
    min_val = min(y_test_t1.min(), y_pred_t1.min())
    max_val = max(y_test_t1.max(), y_pred_t1.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    plt.title('Data Fitting Map: Actual vs. Predicted AQI (t+1 hour, All Cities)')
    plt.xlabel('Actual AQI (Inverse-Transformed)')
    plt.ylabel('Predicted AQI (Inverse-Transformed)')
    plt.legend()
    plt.grid(True)
    plt.savefig('aqi_transformer_actual_vs_predicted_4_cities.png')
    print("Plot saved to 'aqi_transformer_actual_vs_predicted_4_cities.png'")

    # --- 7. Forecast Next 72 Hours (for Jharia only) ---
    print("\n--- AQI Forecast for Next 72 Hours ---")
    
    city_to_predict = 'jharia' # Target city for prediction
    all_trained_cities = df['city'].unique()
    
    if city_to_predict not in all_trained_cities:
        print(f"Error: Target city '{city_to_predict}' was not found in the training data.")
        print(f"Available cities are: {all_trained_cities}")
        print("Please make sure one of your city CSV files is named 'jharia.csv'.")
        return

    print(f"\n--- Generating Forecast for: {city_to_predict} ---")
    
    # Get the last N_LAGS hours of data for this city
    last_sequence_df = df[df['city'] == city_to_predict][all_features].iloc[-N_LAGS:]
    
    if len(last_sequence_df) < N_LAGS:
        print(f"Error: Not enough data for {city_to_predict} to create a full sequence ({len(last_sequence_df)}/{N_LAGS}).")
        return
        
    # Scale this data
    last_sequence_scaled = feature_scaler.transform(last_sequence_df)
    
    # Reshape to (1, seq_len, n_features)
    last_sequence_scaled = last_sequence_scaled.reshape(1, N_LAGS, n_features)

    # Predict (scaled)
    forecast_scaled = model.predict(last_sequence_scaled)
    
    # Inverse transform
    forecast = target_scaler.inverse_transform(forecast_scaled)
    future_72_hours_aqi = forecast.flatten()

    # Get the last timestamp for this city
    last_timestamp = last_sequence_df.index[-1]
    
    forecast_dates = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=N_FORECAST,
        freq='h'
    )

    forecast_df = pd.DataFrame({
        'Timestamp': forecast_dates,
        'Predicted_AQI': future_72_hours_aqi
    })
    forecast_df['Predicted_AQI'] = forecast_df['Predicted_AQI'].round(2)
    
    with pd.option_context('display.max_rows', None):
        print(forecast_df)


if __name__ == "__main__":
    main()