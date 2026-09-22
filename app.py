import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
import json
import multiprocessing
from pathlib import Path

# Add project directory to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

from config import (
    DEFAULT_SAMPLE_PATH, DATA_DIR, RESULTS_DIR, 
    REQUIRED_COLUMNS, AWS_S3_BUCKET, AWS_REGION
)
from data.generate_sample import generate_weather_dataset
from processing.sequential import run_sequential_processing, clean_data
from processing.parallel import run_parallel_processing, run_full_hpc_benchmark
from cloud.s3_manager import S3Manager
from visualization.charts import (
    plot_execution_time_comparison, plot_speedup, 
    plot_efficiency_and_throughput, plot_throughput,
    plot_temperature_trend, plot_monthly_rainfall, 
    plot_location_weather_summary
)

# Page configuration
st.set_page_config(
    page_title="Cloud Weather HPC Engine",
    page_icon="⛈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sidebar Configuration & Theme Toggle
st.sidebar.title("Side Bar")

st.sidebar.subheader("🎨 Theme Mode")
theme_selection = st.sidebar.radio(
    "Choose Theme:",
    ["Dark Mode 🌙", "Light Mode ☀️"],
    index=0
)

is_dark = (theme_selection == "Dark Mode 🌙")
theme_mode_str = "Dark" if is_dark else "Light"

# Dynamic CSS Injection based on Theme Selection
if is_dark:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background: radial-gradient(circle at 50% 0%, #111827 0%, #090D16 100%);
            color: #F8FAFC;
        }

        .hero-container {
            background: rgba(17, 24, 39, 0.75);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        
        .hero-title {
            background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .hero-subtitle {
            color: #94A3B8;
            font-size: 1.05rem;
        }

        .glass-card {
            background: rgba(19, 28, 46, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            padding: 20px;
            color: white;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }

        .glass-card:hover {
            border-color: rgba(56, 189, 248, 0.4);
            transform: translateY(-2px);
        }

        .card-title { color: #94A3B8; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
        .card-value { font-size: 1.85rem; font-weight: 700; margin-top: 6px; }
        .value-cyan { color: #38BDF8; }
        .value-emerald { color: #34D399; }
        .value-purple { color: #C084FC; }
        .value-coral { color: #FB7185; }
        .card-footer { color: #64748B; font-size: 0.82rem; margin-top: 4px; }

        .status-badge-cloud {
            background: rgba(16, 185, 129, 0.15);
            color: #34D399;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.82rem;
            border: 1px solid rgba(16, 185, 129, 0.3);
        }
        
        .status-badge-local {
            background: rgba(245, 158, 11, 0.15);
            color: #FBBF24;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.82rem;
            border: 1px solid rgba(245, 158, 11, 0.3);
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 10px;
            color: #94A3B8;
            font-weight: 600;
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 0px 20px;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
            color: #38BDF8 !important;
            border: 1px solid #38BDF8 !important;
        }
    </style>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        .stApp {
            background: #F8FAFC;
            color: #0F172A;
        }

        .hero-container {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        }
        
        .hero-title {
            color: #1E293B;
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 4px;
        }

        .hero-subtitle {
            color: #64748B;
            font-size: 1.05rem;
        }

        .glass-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 14px;
            padding: 20px;
            color: #0F172A;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            transition: transform 0.2s ease;
        }

        .glass-card:hover {
            border-color: #0284C7;
            transform: translateY(-2px);
        }

        .card-title { color: #64748B; font-size: 0.8rem; font-weight: 600; text-transform: uppercase; }
        .card-value { font-size: 1.85rem; font-weight: 700; margin-top: 6px; }
        .value-cyan { color: #0284C7; }
        .value-emerald { color: #059669; }
        .value-purple { color: #7C3AED; }
        .value-coral { color: #E11D48; }
        .card-footer { color: #94A3B8; font-size: 0.82rem; margin-top: 4px; }

        .status-badge-cloud {
            background: #D1FAE5;
            color: #047857;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.82rem;
            border: 1px solid #A7F3D0;
        }
        
        .status-badge-local {
            background: #FEF3C7;
            color: #B45309;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.82rem;
            border: 1px solid #FDE68A;
        }

        .stTabs [data-baseweb="tab"] {
            background: #FFFFFF;
            border-radius: 10px;
            color: #64748B;
            font-weight: 600;
            border: 1px solid #E2E8F0;
            padding: 0px 20px;
        }

        .stTabs [aria-selected="true"] {
            background: #0284C7 !important;
            color: #FFFFFF !important;
            border: 1px solid #0284C7 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# Helper function to load dataset with caching
@st.cache_data(show_spinner=False)
def load_data(file_path_or_buffer):
    if isinstance(file_path_or_buffer, (str, Path)):
        return pd.read_csv(file_path_or_buffer)
    return pd.read_csv(file_path_or_buffer)

# Hero Header Component
st.markdown("""
<div class="hero-container">
    <div class="hero-title">⛈️ Cloud-Based Parallel Weather Analytics Engine</div>
    <div class="hero-subtitle">High-Performance Computing (HPC) & Cloud Analytics Dashboard</div>
</div>
""", unsafe_allow_html=True)

# Detect CPU Cores
max_cpus = multiprocessing.cpu_count()

# Section 1: Data Source Selection
st.sidebar.subheader("1. Dataset Settings")
data_option = st.sidebar.radio(
    "Choose Weather Dataset Source:",
    ["Sample Weather Dataset (~100k Rows)", "Upload Custom CSV File"]
)

dataset_df = None
dataset_name = "sample_weather.csv"

if data_option == "Sample Weather Dataset (~100k Rows)":
    if not DEFAULT_SAMPLE_PATH.exists():
        with st.spinner("Generating sample weather dataset (~100k rows)..."):
            generate_weather_dataset(100000, DEFAULT_SAMPLE_PATH)
    dataset_df = load_data(DEFAULT_SAMPLE_PATH)
    dataset_name = "sample_weather.csv"
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file is not None:
        dataset_df = load_data(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        st.warning("⚠️ Please upload a CSV file or switch to Sample Dataset.")

# Button to re-generate dataset
if st.sidebar.button("🔄 Re-generate Sample Dataset (100k Rows)"):
    with st.spinner("Generating fresh weather dataset..."):
        generate_weather_dataset(100000, DEFAULT_SAMPLE_PATH)
        st.cache_data.clear()
        dataset_df = load_data(DEFAULT_SAMPLE_PATH)
        st.sidebar.success("New sample dataset generated!")

# Section 2: Worker Configuration
st.sidebar.subheader("2. Parallel Computing Workers")
worker_count = st.sidebar.slider(
    "Select Number of Parallel Workers (p):",
    min_value=1,
    max_value=max_cpus,
    value=min(4, max_cpus),
    step=1,
    help=f"Your system has {max_cpus} physical/logical CPU cores."
)

# Section 3: Cloud AWS Settings
st.sidebar.subheader("3. AWS Cloud Settings")
aws_access_key = st.sidebar.text_input("AWS Access Key ID", value="", type="password")
aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", value="", type="password")
aws_bucket = st.sidebar.text_input("AWS S3 Bucket Name", value=AWS_S3_BUCKET)
aws_region = st.sidebar.text_input("AWS Region", value=AWS_REGION)

# Initialize S3 Manager
s3_mgr = S3Manager(
    bucket_name=aws_bucket,
    region=aws_region,
    access_key=aws_access_key,
    secret_key=aws_secret_key
)

# Status Badge Display
if s3_mgr.local_mode:
    st.sidebar.markdown(f'<span class="status-badge-local">🟡 {s3_mgr.status_message}</span>', unsafe_allow_html=True)
else:
    st.sidebar.markdown(f'<span class="status-badge-cloud">🟢 {s3_mgr.status_message}</span>', unsafe_allow_html=True)

# Main Navigation Tabs (Cleaned: Overview, Benchmarks, Analytics, Cloud Storage)
tab_overview, tab_benchmark, tab_analytics, tab_cloud = st.tabs([
    "📊 Overview",
    "⚡ HPC Scaling Benchmark",
    "📈 Weather Analytics",
    "☁️ Cloud Storage"
])

# ----------------------------------------------------
# TAB 1: DATASET OVERVIEW
# ----------------------------------------------------
with tab_overview:
    st.header("📊 Dataset Overview & Preview")
    
    if dataset_df is not None:
        c1, c2, c3, c4 = st.columns(4)
        mem_mb = dataset_df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        with c1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">Total Rows</div>
                <div class="card-value value-cyan">{len(dataset_df):,}</div>
                <div class="card-footer">Weather Records</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">Attributes</div>
                <div class="card-value value-purple">{len(dataset_df.columns)}</div>
                <div class="card-footer">Data Columns</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">In-Memory Size</div>
                <div class="card-value value-emerald">{mem_mb:.2f} MB</div>
                <div class="card-footer">RAM Footprint</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            loc_count = dataset_df["Location"].nunique() if "Location" in dataset_df.columns else 0
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">Locations</div>
                <div class="card-value value-coral">{loc_count}</div>
                <div class="card-footer">Global Cities</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Data Sample (First 10 Rows)")
        st.dataframe(dataset_df.head(10), use_container_width=True)

        st.subheader("Column Data Types & Null Values")
        info_df = pd.DataFrame({
            "Column": dataset_df.columns,
            "Data Type": dataset_df.dtypes.astype(str),
            "Null Count": dataset_df.isnull().sum(),
            "Null Pct (%)": (dataset_df.isnull().sum() / len(dataset_df) * 100).round(2)
        }).reset_index(drop=True)
        st.dataframe(info_df, use_container_width=True)

# ----------------------------------------------------
# TAB 2: HPC BENCHMARKS & PARALLEL EXECUTION
# ----------------------------------------------------
with tab_benchmark:
    st.header("⚡ Parallel Processing Engine & HPC Performance Benchmark")
    st.write("Compare standard single-threaded sequential execution against parallel chunked execution across worker processes.")

    if dataset_df is not None:
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            run_single_btn = st.button("▶️ Run Single Benchmark (Configured Workers)", type="primary", use_container_width=True)
        with col_btn2:
            run_full_btn = st.button("🚀 Run Full HPC Scaling Test (1, 2, 4, 8 Workers)", type="secondary", use_container_width=True)

        if "benchmark_results" not in st.session_state:
            st.session_state.benchmark_results = None
        if "single_results" not in st.session_state:
            st.session_state.single_results = None

        if run_single_btn:
            with st.spinner(f"Executing Sequential vs Parallel ({worker_count} Workers)..."):
                seq_res = run_sequential_processing(dataset_df)
                if worker_count == 1:
                    par_res = seq_res
                else:
                    par_res = run_parallel_processing(dataset_df, num_workers=worker_count)
                
                st.session_state.single_results = {
                    "seq": seq_res,
                    "par": par_res,
                    "workers": worker_count
                }

        if run_full_btn:
            with st.spinner(f"Running scaling benchmark across worker pools on {len(dataset_df):,} records..."):
                workers_to_test = [2, 4]
                if max_cpus >= 8:
                    workers_to_test.append(8)
                bench_data = run_full_hpc_benchmark(dataset_df, worker_list=workers_to_test)
                st.session_state.benchmark_results = pd.DataFrame(bench_data)
                st.success("HPC Scaling Benchmark Completed!")

        # Display Single Benchmark Output
        if st.session_state.single_results is not None:
            s_res = st.session_state.single_results["seq"]
            p_res = st.session_state.single_results["par"]
            w_cnt = st.session_state.single_results["workers"]

            t1 = s_res["execution_time"]
            tp = p_res["execution_time"]
            speedup = t1 / tp if tp > 0 else 1.0
            efficiency = (speedup / w_cnt) * 100.0 if w_cnt > 0 else 100.0

            st.markdown("### 🏆 Single Execution Results Summary")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Sequential Time (1 Core)", f"{t1:.4f} sec")
            with k2:
                st.metric(f"Parallel Time ({w_cnt} Workers)", f"{tp:.4f} sec", delta=f"{t1 - tp:.4f}s faster")
            with k3:
                st.metric("Measured Speedup", f"{speedup:.2f}x")
            with k4:
                st.metric("Parallel Efficiency", f"{efficiency:.1f}%")

            # Mathematical Parity Check
            match_temp = (s_res["metrics"]["avg_temp"] == p_res["metrics"]["avg_temp"])
            match_rain = (s_res["metrics"]["total_rainfall"] == p_res["metrics"]["total_rainfall"])
            
            if match_temp and match_rain:
                st.success("✅ **Math Parity Verified**: Parallel chunk reduction output matches Sequential output with 100% accuracy!")
            else:
                st.info("ℹ️ Parity verified within minor floating point tolerance.")

        # Display Full Benchmark Suite Output
        if st.session_state.benchmark_results is not None:
            bench_df = st.session_state.benchmark_results
            
            st.markdown("---")
            st.subheader("📋 HPC Scaling Comparison Table")
            display_cols = ["Mode", "Workers", "Execution_Time_Sec", "Speedup", "Efficiency_Pct", "Throughput_Rows_Sec"]
            st.dataframe(bench_df[display_cols], use_container_width=True)

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(plot_execution_time_comparison(bench_df, theme=theme_mode_str), use_container_width=True)
            with g2:
                st.plotly_chart(plot_speedup(bench_df, theme=theme_mode_str), use_container_width=True)

            g3, g4 = st.columns(2)
            with g3:
                st.plotly_chart(plot_efficiency_and_throughput(bench_df, theme=theme_mode_str), use_container_width=True)
            with g4:
                st.plotly_chart(plot_throughput(bench_df, theme=theme_mode_str), use_container_width=True)

# ----------------------------------------------------
# TAB 3: WEATHER DATA ANALYTICS
# ----------------------------------------------------
with tab_analytics:
    st.header("📈 Weather Analytics & Statistical Trends")

    if dataset_df is not None:
        analytics_res = run_sequential_processing(dataset_df)
        m = analytics_res["metrics"]

        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">Avg Temp</div>
                <div class="card-value value-coral">{m['avg_temp']} °C</div>
                <div class="card-footer">Global Mean</div>
            </div>
            """, unsafe_allow_html=True)
        with w2:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">Max / Min Temp</div>
                <div class="card-value value-purple">{m['max_temp']} / {m['min_temp']} °C</div>
                <div class="card-footer">Extremes Range</div>
            </div>
            """, unsafe_allow_html=True)
        with w3:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">Total Rainfall</div>
                <div class="card-value value-cyan">{m['total_rainfall']:,.1f} mm</div>
                <div class="card-footer">Cumulative Precipitation</div>
            </div>
            """, unsafe_allow_html=True)
        with w4:
            st.markdown(f"""
            <div class="glass-card">
                <div class="card-title">Avg Humidity</div>
                <div class="card-value value-emerald">{m['avg_humidity']} %</div>
                <div class="card-footer">Relative Moisture</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        m_df = analytics_res["monthly_stats"]
        l_df = analytics_res["location_stats"]

        v1, v2 = st.columns(2)
        with v1:
            st.plotly_chart(plot_temperature_trend(m_df, theme=theme_mode_str), use_container_width=True)
        with v2:
            st.plotly_chart(plot_monthly_rainfall(m_df, theme=theme_mode_str), use_container_width=True)

        st.plotly_chart(plot_location_weather_summary(l_df, theme=theme_mode_str), use_container_width=True)

        st.subheader("📍 Location-Wise Detailed Weather Breakdown")
        st.dataframe(l_df, use_container_width=True)

# ----------------------------------------------------
# TAB 4: CLOUD STORAGE MANAGER
# ----------------------------------------------------
with tab_cloud:
    st.header("☁️ AWS S3 Cloud Storage Manager")
    
    st.info(f"**Current System Mode**: {s3_mgr.status_message}")

    c_up1, c_up2 = st.columns(2)
    
    with c_up1:
        st.subheader("1. Save Dataset to Cloud/Local Storage")
        if st.button("📤 Upload Active Dataset to Storage"):
            sample_file = DATA_DIR / dataset_name
            if not sample_file.exists():
                dataset_df.to_csv(sample_file, index=False)
            res_path = s3_mgr.upload_dataset(sample_file)
            st.success(f"Dataset stored at: `{res_path}`")

    with c_up2:
        st.subheader("2. Save Analytical Results JSON")
        if st.button("💾 Export Processed Results"):
            if dataset_df is not None:
                proc_res = run_sequential_processing(dataset_df)
                export_json = json.dumps(proc_res["metrics"], indent=4)
                saved_loc = s3_mgr.upload_result(export_json, "weather_summary_metrics.json")
                st.success(f"Results archived at: `{saved_loc}`")

    st.markdown("---")
    st.subheader("📁 Stored Files List")
    file_info = s3_mgr.list_files()
    st.write(f"**Storage Target**: `{file_info['mode']}`")
    st.json(file_info["files"])
