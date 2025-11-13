import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- AQI Calculation (CPCB Standards) ---
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
    
    # Handle values above the highest breakpoint
    if value > breakpoints[-1][1]:
        return 500 # Cap at 500
    
    return 0 # Default for values below lowest (e.g., 0)

def calculate_aqi_from_pollutants(row):
    """
    Calculates the final AQI for a row of data, based on 
    pre-calculated rolling averages.
    """
    sub_indices = [
        get_sub_index(row.get('pm2_5_24h_avg'), 'pm2_5'),
        get_sub_index(row.get('pm10_24h_avg'), 'pm10'),
        get_sub_index(row.get('no2_24h_avg'), 'no2'),
        get_sub_index(row.get('so2_24h_avg'), 'so2'),
        get_sub_index(row.get('co_8h_avg_mg'), 'co'), # Use converted CO
        get_sub_index(row.get('o3_8h_avg'), 'o3'),
    ]
    
    # Filter out NaN values (e.g., from pollutants not present)
    valid_indices = [idx for idx in sub_indices if pd.notna(idx)]
    
    if not valid_indices:
        return np.nan
    
    # The final AQI is the maximum of the sub-indices
    return max(valid_indices)

# --- Main Model Training and Prediction ---

def main():
    """Main function to run the full pipeline."""
    
    # --- 1. Load and Preprocess Data ---
    print("Loading data...")
    try:
        # Load the CSV, skipping the metadata rows at the top
        df = pd.read_csv('jharia.csv', skiprows=3)
    except FileNotFoundError:
        print("Error: 'jharia.csv' not found. Please place it in the same directory.")
        return
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # Rename columns for easier programmatic access
    df.columns = [
        'time', 'pm10', 'pm2_5', 'no2', 'so2', 'o3',
        'co2', 'co', 'dust', 'aod'
    ]

    # Convert time to datetime and set as index
    df['time'] = pd.to_datetime(df['time'])
    df = df.set_index('time')

    # Handle missing values (e.g., fill with last known value, then backfill)
    df = df.ffill().bfill()
    if df.isnull().values.any():
        print("Warning: Data still contains NaNs after filling. Dropping rows.")
        df = df.dropna()

    print("Data loaded and preprocessed.")

    # --- 2. Calculate AQI (Target Variable) ---
    print("Calculating historical AQI...")
    
    # Calculate rolling averages as per CPCB (24h or 8h)
    df['pm2_5_24h_avg'] = df['pm2_5'].rolling(window=24, min_periods=1).mean()
    df['pm10_24h_avg'] = df['pm10'].rolling(window=24, min_periods=1).mean()
    df['no2_24h_avg'] = df['no2'].rolling(window=24, min_periods=1).mean()
    df['so2_24h_avg'] = df['so2'].rolling(window=24, min_periods=1).mean()
    
    # 8-hour for CO and O3
    df['o3_8h_avg'] = df['o3'].rolling(window=8, min_periods=1).mean()
    # Convert CO from μg/m³ to mg/m³ for CPCB calculation
    df['co_8h_avg_mg'] = (df['co'].rolling(window=8, min_periods=1).mean()) / 1000.0

    # Apply the AQI calculation to each row
    df['AQI'] = df.apply(calculate_aqi_from_pollutants, axis=1)

    # We need to drop NaNs created by rolling windows or AQI calc
    df = df.dropna()
    if df.empty:
        print("Error: No data remaining after AQI calculation. Check data file.")
        return
        
    print("AQI calculation complete.")

    # --- 3. Feature Engineering ---
    print("Engineering features...")
    
    N_LAGS = 24      # Use last 24 hours of data to predict
    N_FORECAST = 72  # Predict next 72 hours
    
    # List of all parameters to use as features
    # This includes pollutants and non-pollutants (co2, dust, aod)
    all_features = ['AQI', 'pm10', 'pm2_5', 'no2', 'so2', 'o3', 'co2', 'co', 'dust', 'aod']
    
    # Create the feature DataFrame (X)
    X_list = []
    
    # Add lagged features for all parameters
    for col in all_features:
        for i in range(1, N_LAGS + 1):
            X_list.append(df[col].shift(i).rename(f'{col}_lag{i}'))
            
    # Add time-based features
    X_list.append(df.index.hour.to_series(name='hour_of_day', index=df.index))
    X_list.append(df.index.dayofweek.to_series(name='day_of_week', index=df.index))
    X_list.append(df.index.month.to_series(name='month', index=df.index))

    X = pd.concat(X_list, axis=1)

    # Create the target DataFrame (y)
    # We want to predict AQI at t+1, t+2, ..., t+72
    y_list = []
    for i in range(1, N_FORECAST + 1):
        y_list.append(df['AQI'].shift(-i).rename(f'AQI_forecast{i}'))
        
    y = pd.concat(y_list, axis=1)

    # Combine and drop NaNs (created by all the shifting)
    full_df = pd.concat([X, y], axis=1)
    full_df = full_df.dropna()

    # Separate X and y again from the cleaned data
    X = full_df[X.columns]
    y = full_df[y.columns]

    if X.empty or y.empty:
        print("Error: Not enough data to create training set after lagging.")
        print(f"Original rows: {len(df)}, Rows after lagging: {len(full_df)}")
        return
        
    print(f"Feature engineering complete. Training on {len(X)} samples.")

    # --- 4. Train Model ---
    
    # Split data. IMPORTANT: For time-series, do NOT shuffle.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    print("Training Random Forest Regressor...")
    # Random Forest can handle multi-output regression directly
    model = RandomForestRegressor(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,  # Use all available CPU cores
        min_samples_leaf=5 # Prevent overfitting
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Evaluate Model ---
    y_pred = model.predict(X_test)

    # Evaluate performance for the 1-hour-ahead forecast
    y_test_t1 = y_test.iloc[:, 0]
    y_pred_t1 = y_pred[:, 0]
    
    # Evaluate performance for the 72-hour-ahead forecast
    y_test_t72 = y_test.iloc[:, -1]
    y_pred_t72 = y_pred[:, -1]

    me_t1 = np.mean(y_test_t1 - y_pred_t1)
    rmse_t1 = np.sqrt(mean_squared_error(y_test_t1, y_pred_t1))
    
    me_t72 = np.mean(y_test_t72 - y_pred_t72)
    rmse_t72 = np.sqrt(mean_squared_error(y_test_t72, y_pred_t72))

    print("\n--- Model Evaluation ---")
    print(f"Forecast (t+1 hour)   - Mean Error: {me_t1:.2f}, RMSE: {rmse_t1:.2f}")
    print(f"Forecast (t+72 hours) - Mean Error: {me_t72:.2f}, RMSE: {rmse_t72:.2f}")

    # --- 6. Data Fitting Map (Plot) ---
    print("\nGenerating data fitting map...")
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test_t1, y_pred_t1, alpha=0.5, label='t+1 Forecast')
    
    # Add a 45-degree line for reference
    min_val = min(y_test_t1.min(), y_pred_t1.min())
    max_val = max(y_test_t1.max(), y_pred_t1.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
    
    plt.title('Data Fitting Map: Actual vs. Predicted AQI (t+1 hour)')
    plt.xlabel('Actual AQI')
    plt.ylabel('Predicted AQI')
    plt.legend()
    plt.grid(True)
    plt.savefig('aqi_actual_vs_predicted.png')
    print("Plot saved to 'aqi_actual_vs_predicted.png'")

    # --- 7. Forecast Next 72 Hours ---
    print("\n--- AQI Forecast for Next 72 Hours ---")
    
    # Get the last row of features from the *pre-split* data
    # This row has all the lagged data needed for prediction
    last_feature_row = X.iloc[[-1]]

    # Predict the next 72 values
    future_72_hours_aqi = model.predict(last_feature_row)[0]

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
    with pd.option_context('display.max_rows', None): # Print all 72 rows
        print(forecast_df)

if __name__ == "__main__":
    main()