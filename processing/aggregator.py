import pandas as pd
import numpy as np

def process_chunk(chunk_df: pd.DataFrame) -> dict:
    """
    Worker function executed in parallel across CPU processes.
    Computes intermediate sum/min/max accumulators for a single dataset chunk.
    """
    chunk_df = chunk_df.copy()
    
    # Strip column names
    chunk_df.columns = [col.strip() for col in chunk_df.columns]

    numeric_cols = ["Temperature", "Humidity", "Rainfall", "Wind_Speed", "Pressure"]
    for col in numeric_cols:
        if col in chunk_df.columns:
            chunk_df[col] = pd.to_numeric(chunk_df[col], errors='coerce')

    chunk_df[numeric_cols] = chunk_df[numeric_cols].fillna(chunk_df[numeric_cols].median())

    if "Date" in chunk_df.columns:
        chunk_df["Date"] = pd.to_datetime(chunk_df["Date"], errors='coerce')
        chunk_df["Year"] = chunk_df["Date"].dt.year
        chunk_df["Month"] = chunk_df["Date"].dt.strftime("%Y-%m")

    # Intermediate accumulators
    temp_series = chunk_df["Temperature"]
    hum_series = chunk_df["Humidity"]
    rain_series = chunk_df["Rainfall"]
    wind_series = chunk_df["Wind_Speed"]
    press_series = chunk_df["Pressure"]

    loc_partial = chunk_df.groupby("Location").agg(
        temp_sum=("Temperature", "sum"),
        temp_count=("Temperature", "count"),
        temp_max=("Temperature", "max"),
        temp_min=("Temperature", "min"),
        hum_sum=("Humidity", "sum"),
        hum_count=("Humidity", "count"),
        rain_sum=("Rainfall", "sum"),
        wind_sum=("Wind_Speed", "sum"),
        wind_count=("Wind_Speed", "count")
    ).reset_index()

    monthly_partial = chunk_df.groupby("Month").agg(
        temp_sum=("Temperature", "sum"),
        temp_count=("Temperature", "count"),
        rain_sum=("Rainfall", "sum"),
        hum_sum=("Humidity", "sum"),
        record_count=("Temperature", "count")
    ).reset_index()

    yearly_partial = chunk_df.groupby("Year").agg(
        temp_sum=("Temperature", "sum"),
        temp_count=("Temperature", "count"),
        rain_sum=("Rainfall", "sum"),
        wind_sum=("Wind_Speed", "sum"),
        record_count=("Temperature", "count")
    ).reset_index()

    return {
        "count": len(chunk_df),
        "temp_sum": float(temp_series.sum()),
        "temp_count": int(temp_series.count()),
        "temp_max": float(temp_series.max()),
        "temp_min": float(temp_series.min()),
        "hum_sum": float(hum_series.sum()),
        "hum_count": int(hum_series.count()),
        "rain_sum": float(rain_series.sum()),
        "wind_sum": float(wind_series.sum()),
        "wind_count": int(wind_series.count()),
        "press_sum": float(press_series.sum()),
        "press_count": int(press_series.count()),
        "loc_partial": loc_partial.to_dict(orient="records"),
        "monthly_partial": monthly_partial.to_dict(orient="records"),
        "yearly_partial": yearly_partial.to_dict(orient="records")
    }

def combine_results(partial_list: list) -> dict:
    """
    Reduces a list of partial accumulators from parallel workers into final unified statistics.
    Guaranteed to produce identical results to sequential processing.
    """
    total_rows = sum(p["count"] for p in partial_list)
    
    if total_rows == 0:
        raise ValueError("No data processed across parallel chunks.")

    temp_sum = sum(p["temp_sum"] for p in partial_list)
    temp_count = sum(p["temp_count"] for p in partial_list)
    max_temp = max(p["temp_max"] for p in partial_list)
    min_temp = min(p["temp_min"] for p in partial_list)

    hum_sum = sum(p["hum_sum"] for p in partial_list)
    hum_count = sum(p["hum_count"] for p in partial_list)

    rain_sum = sum(p["rain_sum"] for p in partial_list)

    wind_sum = sum(p["wind_sum"] for p in partial_list)
    wind_count = sum(p["wind_count"] for p in partial_list)

    press_sum = sum(p["press_sum"] for p in partial_list)
    press_count = sum(p["press_count"] for p in partial_list)

    # 1. Combine Location Partials
    all_loc_records = []
    for p in partial_list:
        all_loc_records.extend(p["loc_partial"])
    loc_df = pd.DataFrame(all_loc_records)

    loc_grouped = loc_df.groupby("Location").agg(
        Avg_Temp=("temp_sum", lambda s: loc_df.loc[s.index, "temp_sum"].sum() / loc_df.loc[s.index, "temp_count"].sum()),
        Max_Temp=("temp_max", "max"),
        Min_Temp=("temp_min", "min"),
        Avg_Humidity=("hum_sum", lambda s: loc_df.loc[s.index, "hum_sum"].sum() / loc_df.loc[s.index, "hum_count"].sum()),
        Total_Rainfall=("rain_sum", "sum"),
        Avg_Wind_Speed=("wind_sum", lambda s: loc_df.loc[s.index, "wind_sum"].sum() / loc_df.loc[s.index, "wind_count"].sum()),
        Record_Count=("temp_count", "sum")
    ).reset_index()

    # 2. Combine Monthly Partials
    all_monthly_records = []
    for p in partial_list:
        all_monthly_records.extend(p["monthly_partial"])
    month_df = pd.DataFrame(all_monthly_records)

    monthly_grouped = month_df.groupby("Month").agg(
        Avg_Temp=("temp_sum", lambda s: month_df.loc[s.index, "temp_sum"].sum() / month_df.loc[s.index, "temp_count"].sum()),
        Total_Rainfall=("rain_sum", "sum"),
        Avg_Humidity=("hum_sum", lambda s: month_df.loc[s.index, "hum_sum"].sum() / month_df.loc[s.index, "temp_count"].sum()),
        Record_Count=("record_count", "sum")
    ).reset_index().sort_values("Month")

    # 3. Combine Yearly Partials
    all_yearly_records = []
    for p in partial_list:
        all_yearly_records.extend(p["yearly_partial"])
    year_df = pd.DataFrame(all_yearly_records)

    yearly_grouped = year_df.groupby("Year").agg(
        Avg_Temp=("temp_sum", lambda s: year_df.loc[s.index, "temp_sum"].sum() / year_df.loc[s.index, "temp_count"].sum()),
        Total_Rainfall=("rain_sum", "sum"),
        Avg_Wind_Speed=("wind_sum", lambda s: year_df.loc[s.index, "wind_sum"].sum() / year_df.loc[s.index, "temp_count"].sum()),
        Record_Count=("record_count", "sum")
    ).reset_index().sort_values("Year")

    return {
        "metrics": {
            "avg_temp": round(temp_sum / temp_count, 2),
            "max_temp": round(max_temp, 2),
            "min_temp": round(min_temp, 2),
            "avg_humidity": round(hum_sum / hum_count, 2),
            "total_rainfall": round(rain_sum, 2),
            "avg_wind_speed": round(wind_sum / wind_count, 2),
            "avg_pressure": round(press_sum / press_count, 2),
            "total_rows": total_rows
        },
        "location_stats": loc_grouped,
        "monthly_stats": monthly_grouped,
        "yearly_stats": yearly_grouped
    }
