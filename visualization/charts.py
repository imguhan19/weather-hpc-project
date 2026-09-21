import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Palette dark theme styling
DARK_TEMPLATE = "plotly_dark"
COLOR_PRIMARY = "#00D2FF"
COLOR_SECONDARY = "#00E676"
COLOR_ACCENT = "#FF9100"
COLOR_WARNING = "#FF1744"

def plot_execution_time_comparison(benchmark_df: pd.DataFrame):
    """
    Bar chart showing Sequential vs Parallel (2, 4, 8 workers) execution times.
    """
    fig = px.bar(
        benchmark_df,
        x="Mode",
        y="Execution_Time_Sec",
        text="Execution_Time_Sec",
        color="Execution_Time_Sec",
        color_continuous_scale="Blues_r",
        title="⚡ Execution Time Comparison (Sequential vs Parallel Workers)",
        labels={"Execution_Time_Sec": "Execution Time (Seconds)", "Mode": "Processing Mode"}
    )
    fig.update_traces(texttemplate='%{text:.4f}s', textposition='outside')
    fig.update_layout(
        template=DARK_TEMPLATE,
        coloraxis_showscale=False,
        yaxis=dict(title="Execution Time (s)"),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_speedup(benchmark_df: pd.DataFrame):
    """
    Line chart showing Measured Speedup vs Ideal Linear Speedup.
    """
    workers = benchmark_df["Workers"].tolist()
    speedups = benchmark_df["Speedup"].tolist()
    ideal_speedups = workers  # Ideal linear speedup S = p

    fig = go.Figure()
    
    # Measured Speedup
    fig.add_trace(go.Scatter(
        x=workers,
        y=speedups,
        mode='lines+markers+text',
        name='Measured Speedup',
        text=[f"{s:.2f}x" for s in speedups],
        textposition="top left",
        line=dict(color=COLOR_PRIMARY, width=3),
        marker=dict(size=10, symbol='circle')
    ))

    # Ideal Speedup
    fig.add_trace(go.Scatter(
        x=workers,
        y=ideal_speedups,
        mode='lines',
        name='Ideal Linear Speedup (S = p)',
        line=dict(color=COLOR_SECONDARY, width=2, dash='dash')
    ))

    fig.update_layout(
        title="🚀 Speedup vs Number of Worker Processes (Amdahl's Law Comparison)",
        xaxis=dict(title="Number of Parallel Workers (p)", dtick=1),
        yaxis=dict(title="Speedup Factor (S)"),
        template=DARK_TEMPLATE,
        legend=dict(x=0.05, y=0.95),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_efficiency_and_throughput(benchmark_df: pd.DataFrame):
    """
    Subplots or combined bar chart showing Parallel Efficiency % and Throughput (Rows/sec).
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=benchmark_df["Mode"],
        y=benchmark_df["Efficiency_Pct"],
        name="Efficiency (%)",
        marker_color=COLOR_SECONDARY,
        text=[f"{e:.1f}%" for e in benchmark_df["Efficiency_Pct"]],
        textposition="auto"
    ))

    fig.update_layout(
        title="📊 Parallel Efficiency Across Worker Configurations",
        xaxis=dict(title="Worker Mode"),
        yaxis=dict(title="Efficiency Percentage (%)", range=[0, 120]),
        template=DARK_TEMPLATE,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_throughput(benchmark_df: pd.DataFrame):
    """
    Bar chart showing Processing Throughput (Rows/Second).
    """
    fig = px.bar(
        benchmark_df,
        x="Mode",
        y="Throughput_Rows_Sec",
        text="Throughput_Rows_Sec",
        color="Throughput_Rows_Sec",
        color_continuous_scale="Viridis",
        title="📈 Data Processing Throughput (Records / Second)",
        labels={"Throughput_Rows_Sec": "Throughput (Rows/sec)", "Mode": "Mode"}
    )
    fig.update_traces(texttemplate='%{text:,.0f} r/s', textposition='outside')
    fig.update_layout(
        template=DARK_TEMPLATE,
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_temperature_trend(monthly_df: pd.DataFrame):
    """
    Line chart showing Average Monthly Temperature trend over time.
    """
    fig = px.line(
        monthly_df,
        x="Month",
        y="Avg_Temp",
        markers=True,
        title="🌡️ Monthly Average Temperature Trend",
        labels={"Avg_Temp": "Average Temperature (°C)", "Month": "Year-Month"}
    )
    fig.update_traces(line_color=COLOR_ACCENT, line_width=3, marker_size=8)
    fig.update_layout(
        template=DARK_TEMPLATE,
        xaxis_tickangle=-45,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_monthly_rainfall(monthly_df: pd.DataFrame):
    """
    Bar chart showing Total Monthly Rainfall distribution.
    """
    fig = px.bar(
        monthly_df,
        x="Month",
        y="Total_Rainfall",
        title="🌧️ Total Monthly Rainfall Distribution",
        labels={"Total_Rainfall": "Total Rainfall (mm)", "Month": "Year-Month"},
        color="Total_Rainfall",
        color_continuous_scale="Teal"
    )
    fig.update_layout(
        template=DARK_TEMPLATE,
        xaxis_tickangle=-45,
        coloraxis_showscale=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_location_weather_summary(location_df: pd.DataFrame):
    """
    Grouped bar chart showing Average Temperature & Average Humidity by Location.
    """
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=location_df["Location"],
        y=location_df["Avg_Temp"],
        name="Avg Temp (°C)",
        marker_color=COLOR_ACCENT
    ))

    fig.add_trace(go.Bar(
        x=location_df["Location"],
        y=location_df["Avg_Humidity"],
        name="Avg Humidity (%)",
        marker_color=COLOR_PRIMARY
    ))

    fig.update_layout(
        barmode='group',
        title="📍 Weather Summary by Location (Temperature & Humidity)",
        xaxis=dict(title="City / Location"),
        yaxis=dict(title="Value"),
        template=DARK_TEMPLATE,
        legend=dict(x=0.8, y=0.95),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
