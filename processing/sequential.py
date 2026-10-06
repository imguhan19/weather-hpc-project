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

from processing.aggregator import process_chunk, combine_results

def run_sequential_processing(df: pd.DataFrame) -> dict:
    """
    Executes sequential baseline processing on the weather dataset.
    """
    start_time = time.perf_counter()
    
    # Clean data and execute single-chunk sequential accumulator
    clean_df = clean_data(df)
    if len(clean_df) == 0:
        raise ValueError("The provided CSV file contains no valid data rows.")

    partial = process_chunk(clean_df)
    res = combine_results([partial])

    end_time = time.perf_counter()
    execution_time = max(0.0001, end_time - start_time)

    res["execution_time"] = execution_time
    res["workers"] = 1
    res["mode"] = "Sequential"

    return res
