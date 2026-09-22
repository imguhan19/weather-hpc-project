import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

FONT_FAMILY = "Outfit, Inter, -apple-system, sans-serif"

# Vibrant Color Palette
NEON_CYAN = "#00F2FE"
NEON_BLUE = "#4FACFE"
NEON_GREEN = "#00E676"
NEON_PURPLE = "#7C4DFF"
NEON_CORAL = "#FF4B2B"
NEON_AMBER = "#FFB300"

def apply_theme(fig, title_text, theme="Dark"):
    """Applies high-end dark or light theme to Plotly figures based on user selection."""
    if theme == "Dark":
        bg_color = "rgba(19, 28, 46, 0.6)"
        grid_color = "#1E293B"
        font_title = "#F8FAFC"
        font_axis = "#94A3B8"
        legend_bg = "rgba(15, 23, 42, 0.8)"
        legend_border = "#334155"
    else:
        bg_color = "rgba(248, 250, 252, 0.8)"
        grid_color = "#E2E8F0"
        font_title = "#0F172A"
        font_axis = "#475569"
        legend_bg = "rgba(255, 255, 255, 0.9)"
        legend_border = "#CBD5E1"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, family=FONT_FAMILY, color=font_title)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=bg_color,
        font=dict(family=FONT_FAMILY, color=font_axis),
        xaxis=dict(
            showgrid=True, gridcolor=grid_color, 
            zeroline=False, tickfont=dict(color=font_axis)
        ),
        yaxis=dict(
            showgrid=True, gridcolor=grid_color, 
            zeroline=False, tickfont=dict(color=font_axis)
        ),
        legend=dict(
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
            font=dict(color=font_title)
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_execution_time_comparison(benchmark_df: pd.DataFrame, theme="Dark", **kwargs):
    """Bar chart showing Sequential vs Parallel execution times."""
    fig = px.bar(
        benchmark_df,
        x="Mode",
        y="Execution_Time_Sec",
        text="Execution_Time_Sec",
        color="Execution_Time_Sec",
        color_continuous_scale=["#00F2FE", "#7C4DFF", "#FF4B2B"],
        labels={"Execution_Time_Sec": "Execution Time (s)", "Mode": "Mode"}
    )
    fig.update_traces(
        texttemplate='<b>%{text:.4f}s</b>', 
        textposition='outside',
        marker=dict(line=dict(color='#FFFFFF', width=1))
    )
    apply_theme(fig, "⚡ Execution Time Benchmark (Sequential vs Multi-Worker)", theme=theme)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_speedup(benchmark_df: pd.DataFrame, theme="Dark", **kwargs):
    """Glowing line chart comparing Measured Speedup vs Theoretical Ideal Speedup."""
    workers = benchmark_df["Workers"].tolist()
    speedups = benchmark_df["Speedup"].tolist()
    ideal_speedups = workers

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=workers,
        y=speedups,
        mode='lines+markers+text',
        name='Measured Speedup (S)',
        text=[f"<b>{s:.2f}x</b>" for s in speedups],
        textposition="top left",
        line=dict(color=NEON_CYAN, width=4),
        marker=dict(size=12, color=NEON_CYAN, line=dict(color='#FFFFFF', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0, 242, 254, 0.15)'
    ))

    fig.add_trace(go.Scatter(
        x=workers,
        y=ideal_speedups,
        mode='lines',
        name='Ideal Linear Speedup (S = p)',
        line=dict(color=NEON_GREEN, width=2, dash='dash')
    ))

    apply_theme(fig, "🚀 Speedup Factor vs Number of Worker Processes", theme=theme)
    fig.update_layout(
        xaxis=dict(title="Parallel Worker Processes (p)", dtick=1),
        yaxis=dict(title="Speedup Factor (S)"),
        legend=dict(x=0.05, y=0.95)
    )
    return fig

def plot_efficiency_and_throughput(benchmark_df: pd.DataFrame, theme="Dark", **kwargs):
    """Vibrant bar chart showing Parallel CPU Efficiency %."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=benchmark_df["Mode"],
        y=benchmark_df["Efficiency_Pct"],
        name="Efficiency (%)",
        marker=dict(
            color=benchmark_df["Efficiency_Pct"],
            colorscale=["#FF4B2B", "#FFB300", "#00E676"],
            line=dict(color='#FFFFFF', width=1)
        ),
        text=[f"<b>{e:.1f}%</b>" for e in benchmark_df["Efficiency_Pct"]],
        textposition="auto"
    ))

    apply_theme(fig, "📊 CPU Core Efficiency Percentage (%)", theme=theme)
    fig.update_layout(
        yaxis=dict(title="Efficiency (%)", range=[0, 125])
    )
    return fig

def plot_throughput(benchmark_df: pd.DataFrame, theme="Dark", **kwargs):
    """Glow bar chart showing Processing Throughput (Records/sec)."""
    fig = px.bar(
        benchmark_df,
        x="Mode",
        y="Throughput_Rows_Sec",
        text="Throughput_Rows_Sec",
        color="Throughput_Rows_Sec",
        color_continuous_scale="Plasma",
        labels={"Throughput_Rows_Sec": "Throughput (Rows/s)", "Mode": "Mode"}
    )
    fig.update_traces(
        texttemplate='<b>%{text:,.0f} r/s</b>', 
        textposition='outside'
    )
    apply_theme(fig, "📈 Data Processing Throughput (Records / Second)", theme=theme)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_temperature_trend(monthly_df: pd.DataFrame, theme="Dark", **kwargs):
    """Smooth glowing line chart for Monthly Average Temperatures."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_df["Month"],
        y=monthly_df["Avg_Temp"],
        mode='lines+markers',
        name='Avg Temp (°C)',
        line=dict(color=NEON_CORAL, width=3, shape='spline'),
        marker=dict(size=8, color=NEON_CORAL),
        fill='tozeroy',
        fillcolor='rgba(255, 75, 43, 0.12)'
    ))

    apply_theme(fig, "🌡️ Monthly Average Temperature Trend", theme=theme)
    fig.update_layout(
        xaxis=dict(tickangle=-45, title="Month"),
        yaxis=dict(title="Temperature (°C)")
    )
    return fig

def plot_monthly_rainfall(monthly_df: pd.DataFrame, theme="Dark", **kwargs):
    """Vibrant bar chart for Total Monthly Rainfall."""
    fig = px.bar(
        monthly_df,
        x="Month",
        y="Total_Rainfall",
        color="Total_Rainfall",
        color_continuous_scale="Blues",
        labels={"Total_Rainfall": "Rainfall (mm)", "Month": "Month"}
    )
    fig.update_traces(marker=dict(line=dict(color='#38BDF8', width=1)))
    apply_theme(fig, "🌧️ Total Monthly Rainfall Distribution", theme=theme)
    fig.update_layout(
        coloraxis_showscale=False,
        xaxis=dict(tickangle=-45)
    )
    return fig

def plot_location_weather_summary(location_df: pd.DataFrame, theme="Dark", **kwargs):
    """Grouped bar chart for Temperature & Humidity across global locations."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=location_df["Location"],
        y=location_df["Avg_Temp"],
        name="Avg Temp (°C)",
        marker_color=NEON_AMBER
    ))

    fig.add_trace(go.Bar(
        x=location_df["Location"],
        y=location_df["Avg_Humidity"],
        name="Avg Humidity (%)",
        marker_color=NEON_CYAN
    ))

    apply_theme(fig, "📍 Global Weather Summary by Location", theme=theme)
    fig.update_layout(
        barmode='group',
        xaxis=dict(title="Location"),
        yaxis=dict(title="Value"),
        legend=dict(x=0.8, y=0.95)
    )
    return fig
