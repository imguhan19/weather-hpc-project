import time
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from processing.aggregator import process_chunk, combine_results

def split_dataframe(df: pd.DataFrame, num_chunks: int) -> list:
    """
    Splits a pandas DataFrame into N roughly equal chunks for worker processes.
    """
    if num_chunks <= 1:
        return [df]
    chunk_size = int(np.ceil(len(df) / num_chunks))
    return [df.iloc[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks) if i * chunk_size < len(df)]

def run_parallel_processing(df: pd.DataFrame, num_workers: int = 4) -> dict:
    """
    Executes parallel processing on the weather dataset using multiple CPU worker processes.
    
    Args:
        df (pd.DataFrame): Input weather dataset DataFrame
        num_workers (int): Number of parallel CPU worker processes (e.g., 2, 4, 8)
        
    Returns:
        dict: Statistical metrics, aggregated tables, worker counts, and execution time
    """
    start_time = time.perf_counter()
    
    # 1. Divide dataset into chunks
    chunks = split_dataframe(df, num_workers)
    
    # 2. Process chunks in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        partial_results = list(executor.map(process_chunk, chunks))

    # 3. Combine partial accumulators into unified results
    combined = combine_results(partial_results)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    combined["execution_time"] = execution_time
    combined["workers"] = num_workers
    combined["mode"] = f"Parallel ({num_workers} Workers)"
    
    return combined

def run_full_hpc_benchmark(df: pd.DataFrame, worker_list: list = None, sequential_res: dict = None) -> list:
    """
    Runs comprehensive HPC benchmark comparing Sequential vs 2, 4, 8 workers.
    Computes Execution Time, Speedup, Efficiency %, and Throughput.
    
    Args:
        df (pd.DataFrame): Weather dataset DataFrame
        worker_list (list): Worker counts to test (e.g. [2, 4, 8])
        sequential_res (dict): Pre-computed sequential result (optional)
        
    Returns:
        list of dict: Full benchmark statistics across all worker counts
    """
    from processing.sequential import run_sequential_processing

    if worker_list is None:
        worker_list = [2, 4, 8]

    # Run or use sequential baseline
    if sequential_res is None:
        seq_res = run_sequential_processing(df)
    else:
        seq_res = sequential_res

    t1 = seq_res["execution_time"]
    total_rows = seq_res["metrics"]["total_rows"]

    benchmark_results = []
    
    # Baseline Sequential Entry
    benchmark_results.append({
        "Workers": 1,
        "Mode": "Sequential (1 Core)",
        "Execution_Time_Sec": round(t1, 4),
        "Speedup": 1.0,
        "Efficiency_Pct": 100.0,
        "Throughput_Rows_Sec": round(total_rows / t1, 2) if t1 > 0 else 0,
        "Full_Result": seq_res
    })

    # Benchmark parallel workers
    for p in worker_list:
        par_res = run_parallel_processing(df, num_workers=p)
        tp = par_res["execution_time"]
        
        speedup = t1 / tp if tp > 0 else 1.0
        efficiency = (speedup / p) * 100.0
        throughput = total_rows / tp if tp > 0 else 0.0

        benchmark_results.append({
            "Workers": p,
            "Mode": f"Parallel ({p} Workers)",
            "Execution_Time_Sec": round(tp, 4),
            "Speedup": round(speedup, 2),
            "Efficiency_Pct": round(efficiency, 2),
            "Throughput_Rows_Sec": round(throughput, 2),
            "Full_Result": par_res
        })

    return benchmark_results
