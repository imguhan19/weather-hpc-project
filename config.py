import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
TESTS_DIR = BASE_DIR / "tests"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Default sample dataset path
DEFAULT_SAMPLE_PATH = DATA_DIR / "sample_weather.csv"

# Required Dataset Columns
REQUIRED_COLUMNS = [
    "Date",
    "Location",
    "Temperature",
    "Humidity",
    "Rainfall",
    "Wind_Speed",
    "Pressure"
]

# AWS S3 Settings (Can be overridden via UI or Environment variables)
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "weather-hpc-data-bucket")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID", "")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "")

# Processing Defaults
DEFAULT_WORKER_COUNTS = [1, 2, 4, 8]
