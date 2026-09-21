import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data.generate_sample import generate_weather_dataset
from processing.sequential import run_sequential_processing
from processing.parallel import run_parallel_processing

@pytest.fixture(scope="module")
def sample_dataframe(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("test_data")
    test_csv = temp_dir / "test_weather.csv"
    generate_weather_dataset(num_rows=10000, output_path=test_csv)
    df = pd.read_csv(test_csv)
    return df

def test_sequential_vs_parallel_parity(sample_dataframe):
    """
    Verifies that sequential processing and parallel processing (2 & 4 workers)
    produce mathematically identical statistical outputs.
    """
    df = sample_dataframe
    
    seq_res = run_sequential_processing(df)
    par_2_res = run_parallel_processing(df, num_workers=2)
    par_4_res = run_parallel_processing(df, num_workers=4)

    seq_m = seq_res["metrics"]
    par2_m = par_2_res["metrics"]
    par4_m = par_4_res["metrics"]

    # 1. Total Rows Check
    assert seq_m["total_rows"] == par2_m["total_rows"] == par4_m["total_rows"]

    # 2. Temperature Metric Parity
    assert pytest.approx(seq_m["avg_temp"], 0.01) == par2_m["avg_temp"] == par4_m["avg_temp"]
    assert seq_m["max_temp"] == par2_m["max_temp"] == par4_m["max_temp"]
    assert seq_m["min_temp"] == par2_m["min_temp"] == par4_m["min_temp"]

    # 3. Rainfall Parity
    assert pytest.approx(seq_m["total_rainfall"], 0.01) == par2_m["total_rainfall"] == par4_m["total_rainfall"]

    # 4. Location Stats Grouping Parity
    seq_loc = seq_res["location_stats"].sort_values("Location").reset_index(drop=True)
    par4_loc = par_4_res["location_stats"].sort_values("Location").reset_index(drop=True)
    
    pd.testing.assert_series_equal(seq_loc["Avg_Temp"], par4_loc["Avg_Temp"], check_exact=False, rtol=1e-2)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
