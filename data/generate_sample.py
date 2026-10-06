import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pathlib import Path
import sys

# Add parent directory to path so config can be imported
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_SAMPLE_PATH, DATA_DIR, REQUIRED_COLUMNS

def generate_weather_dataset(num_rows=100000, output_path=DEFAULT_SAMPLE_PATH):
    """
    Generates a realistic synthetic weather dataset for HPC testing.
    
    Args:
        num_rows (int): Number of weather observation rows (default 100,000)
        output_path (Path/str): Target CSV output path
    """
    print(f"Generating synthetic weather dataset with {num_rows:,} rows...")
    
    locations = [
        "New York", "London", "Tokyo", "Sydney", "Mumbai", 
        "Cairo", "Paris", "Chicago", "Singapore", "Toronto"
    ]
    
    # Base climate parameters per location (Mean Temp °C, Temp Variance, Base Humidity %, Base Pressure hPa)
    climate_profiles = {
        "New York": (13.0, 15.0, 65.0, 1014.0),
        "London": (11.0, 8.0, 75.0, 1015.0),
        "Tokyo": (15.5, 12.0, 70.0, 1013.0),
        "Sydney": (18.5, 7.0, 68.0, 1017.0),
        "Mumbai": (27.5, 4.0, 80.0, 1009.0),
        "Cairo": (22.5, 10.0, 45.0, 1016.0),
        "Paris": (12.5, 10.0, 72.0, 1015.0),
        "Chicago": (10.0, 16.0, 68.0, 1014.0),
        "Singapore": (28.0, 2.0, 84.0, 1008.0),
        "Toronto": (9.0, 16.0, 70.0, 1015.0)
    }

    start_date = datetime(2018, 1, 1)
    
    # Pre-generate arrays for speed
    loc_array = np.random.choice(locations, size=num_rows)
    day_offsets = np.random.randint(0, 365 * 5, size=num_rows)
    dates = [start_date + timedelta(days=int(d)) for d in day_offsets]
    
    temps = np.zeros(num_rows)
    humidities = np.zeros(num_rows)
    rainfalls = np.zeros(num_rows)
    wind_speeds = np.zeros(num_rows)
    pressures = np.zeros(num_rows)

    for i in range(num_rows):
        loc = loc_array[i]
        base_t, var_t, base_h, base_p = climate_profiles[loc]
        day_of_year = dates[i].timetuple().tm_yday
        
        # Seasonal sine wave variation
        seasonal_t = var_t * np.cos(2 * np.pi * (day_of_year - 200) / 365)
        daily_noise = np.random.normal(0, 3.0)
        t = round(base_t + seasonal_t + daily_noise, 2)
        
        # Humidity inverse to temperature + noise
        h = round(np.clip(base_h - (t - base_t) * 1.2 + np.random.normal(0, 5), 10, 100), 2)
        
        # Rainfall probability based on humidity
        rain_prob = 0.35 if h > 75 else 0.1
        rain = round(np.random.exponential(scale=12.0) if np.random.rand() < rain_prob else 0.0, 2)
        
        # Wind speed & pressure
        w = round(np.clip(np.random.gamma(shape=2.0, scale=4.0), 0.5, 85.0), 2)
        p = round(base_p + np.random.normal(0, 8.0) - (w * 0.1), 2)
        
        temps[i] = t
        humidities[i] = h
        rainfalls[i] = rain
        wind_speeds[i] = w
        pressures[i] = p

    df = pd.DataFrame({
        "Date": [d.strftime("%Y-%m-%d") for d in dates],
        "Location": loc_array,
        "Temperature": temps,
        "Humidity": humidities,
        "Rainfall": rainfalls,
        "Wind_Speed": wind_speeds,
        "Pressure": pressures
    })

    # Introduce ~0.1% random missing values to test robustness / clean-up code
    nan_mask = np.random.rand(num_rows) < 0.001
    df.loc[nan_mask, "Temperature"] = np.nan

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created at {output_path} ({df.memory_usage(deep=True).sum() / (1024*1024):.2f} MB)")
    return output_path

if __name__ == "__main__":
    generate_weather_dataset(100000)
