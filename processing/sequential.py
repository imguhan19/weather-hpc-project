import time
import pandas as pd
import numpy as np

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans dataset by stripping whitespace, parsing numeric fields,
    handling missing values, and validating required columns.
    """
    df = df.copy()
    
    # Standardize column names (strip whitespace)
    df.columns = [col.strip() for col in df.columns]

    numeric_cols = ["Temperature", "Humidity", "Rainfall", "Wind_Speed", "Pressure"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill or drop missing numeric values
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.strftime("%Y-%m")

    return df

def run_sequential_processing(df: pd.DataFrame) -> dict:
    """
    Executes sequential baseline processing on the weather dataset.
    
    Args:
        df (pd.DataFrame): Raw or uploaded weather dataset DataFrame
        
    Returns:
        dict: Complete statistical analysis and execution timing
    """
    start_time = time.perf_counter()
    
    # Step 1: Clean & prepare data
    clean_df = clean_data(df)
    total_rows = len(clean_df)

    if total_rows == 0:
        raise ValueError("The provided CSV file contains no valid data rows.")

    # Step 2: Global aggregations
    avg_temp = float(clean_df["Temperature"].mean())
    max_temp = float(clean_df["Temperature"].max())
    min_temp = float(clean_df["Temperature"].min())
    avg_humidity = float(clean_df["Humidity"].mean())
    total_rainfall = float(clean_df["Rainfall"].sum())
    avg_wind_speed = float(clean_df["Wind_Speed"].mean())
    avg_pressure = float(clean_df["Pressure"].mean())

    # Step 3: Grouped aggregations (Location-wise)
    loc_grouped = clean_df.groupby("Location").agg(
        Avg_Temp=("Temperature", "mean"),
        Max_Temp=("Temperature", "max"),
        Min_Temp=("Temperature", "min"),
        Avg_Humidity=("Humidity", "mean"),
        Total_Rainfall=("Rainfall", "sum"),
        Avg_Wind_Speed=("Wind_Speed", "mean"),
        Record_Count=("Temperature", "count")
    ).reset_index()

    # Step 4: Grouped aggregations (Monthly)
    monthly_grouped = clean_df.groupby("Month").agg(
        Avg_Temp=("Temperature", "mean"),
        Total_Rainfall=("Rainfall", "sum"),
        Avg_Humidity=("Humidity", "mean"),
        Record_Count=("Temperature", "count")
    ).reset_index().sort_values("Month")

    # Step 5: Grouped aggregations (Yearly)
    yearly_grouped = clean_df.groupby("Year").agg(
        Avg_Temp=("Temperature", "mean"),
        Total_Rainfall=("Rainfall", "sum"),
        Avg_Wind_Speed=("Wind_Speed", "mean"),
        Record_Count=("Temperature", "count")
    ).reset_index().sort_values("Year")

    end_time = time.perf_counter()
    execution_time = end_time - start_time

    return {
        "metrics": {
            "avg_temp": round(avg_temp, 2),
            "max_temp": round(max_temp, 2),
            "min_temp": round(min_temp, 2),
            "avg_humidity": round(avg_humidity, 2),
            "total_rainfall": round(total_rainfall, 2),
            "avg_wind_speed": round(avg_wind_speed, 2),
            "avg_pressure": round(avg_pressure, 2),
            "total_rows": total_rows
        },
        "location_stats": loc_grouped,
        "monthly_stats": monthly_grouped,
        "yearly_stats": yearly_grouped,
        "execution_time": execution_time,
        "workers": 1,
        "mode": "Sequential"
    }
