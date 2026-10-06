import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

FONT_FAMILY = "Inter, Segoe UI, -apple-system, sans-serif"

def apply_theme(fig, title_text, theme="Dark"):
    """Applies theme to Plotly figures based on user selection (Dark vs Light)."""
    if theme == "Dark":
        paper_bg = "rgba(15, 23, 42, 0)"
        plot_bg = "#1E293B"
        grid_color = "#334155"
        text_color = "#FFFFFF"
        sub_text_color = "#CBD5E1"
        legend_bg = "rgba(15, 23, 42, 0.9)"
        legend_border = "#334155"
    else:
        paper_bg = "rgba(0,0,0,0)"
        plot_bg = "#FFFFFF"
        grid_color = "#E2E8F0"
        text_color = "#0F172A"
        sub_text_color = "#334155"
        legend_bg = "rgba(255, 255, 255, 0.95)"
        legend_border = "#CBD5E1"

    fig.update_layout(
        title=dict(
            text=f"<b>{title_text}</b>",
            font=dict(size=18, family=FONT_FAMILY, color=text_color)
        ),
        paper_bgcolor=paper_bg,
        plot_bgcolor=plot_bg,
        font=dict(family=FONT_FAMILY, color=sub_text_color),
        xaxis=dict(
            showgrid=True, gridcolor=grid_color, 
            zeroline=False, tickfont=dict(color=text_color, size=12)
        ),
        yaxis=dict(
            showgrid=True, gridcolor=grid_color, 
            zeroline=False, tickfont=dict(color=text_color, size=12)
        ),
        legend=dict(
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
            font=dict(color=text_color, size=12)
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
        color_continuous_scale=["#0284C7", "#7C4DFF", "#FF4B2B"],
        labels={"Execution_Time_Sec": "Execution Time (s)", "Mode": "Mode"}
    )
    fig.update_traces(
        texttemplate='<b>%{text:.4f}s</b>', 
        textposition='outside',
        textfont=dict(size=13, color="#FFFFFF" if theme == "Dark" else "#0F172A"),
        marker=dict(line=dict(color='#0284C7', width=1.5))
    )
    apply_theme(fig, "⚡ Execution Time Benchmark (Sequential vs Multi-Worker)", theme=theme)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_speedup(benchmark_df: pd.DataFrame, theme="Dark", **kwargs):
    """Line chart comparing Measured Speedup vs Theoretical Ideal Speedup."""
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
        textfont=dict(color="#38BDF8" if theme == "Dark" else "#0284C7", size=13),
        line=dict(color="#00F2FE" if theme == "Dark" else "#0284C7", width=4),
        marker=dict(size=12, color="#00F2FE" if theme == "Dark" else "#0284C7", line=dict(color='#FFFFFF', width=2)),
        fill='tozeroy',
        fillcolor='rgba(0, 242, 254, 0.15)' if theme == "Dark" else 'rgba(2, 132, 199, 0.08)'
    ))

    fig.add_trace(go.Scatter(
        x=workers,
        y=ideal_speedups,
        mode='lines',
        name='Ideal Linear Speedup (S = p)',
        line=dict(color="#00E676" if theme == "Dark" else "#0D9488", width=2.5, dash='dash')
    ))

    apply_theme(fig, "🚀 Speedup Factor vs Number of Worker Processes", theme=theme)
    fig.update_layout(
        xaxis=dict(title="Parallel Worker Processes (p)", dtick=1),
        yaxis=dict(title="Speedup Factor (S)"),
        legend=dict(x=0.05, y=0.95)
    )
    return fig

def plot_efficiency_and_throughput(benchmark_df: pd.DataFrame, theme="Dark", **kwargs):
    """Bar chart showing Parallel CPU Efficiency %."""
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
        textposition="auto",
        textfont=dict(color="#FFFFFF", size=12)
    ))

    apply_theme(fig, "📊 CPU Core Efficiency Percentage (%)", theme=theme)
    fig.update_layout(
        yaxis=dict(title="Efficiency (%)", range=[0, 125])
    )
    return fig

def plot_throughput(benchmark_df: pd.DataFrame, theme="Dark", **kwargs):
    """Bar chart showing Processing Throughput (Records/sec)."""
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
        textposition='outside',
        textfont=dict(size=12, color="#FFFFFF" if theme == "Dark" else "#0F172A")
    )
    apply_theme(fig, "📈 Data Processing Throughput (Records / Second)", theme=theme)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_live_city_temperature_comparison(city_df: pd.DataFrame, theme="Dark", **kwargs):
    """Bar chart comparing current live temperatures across cities."""
    temp_col = "Temperature (°C)" if "Temperature (°C)" in city_df.columns else "Temperature"
    
    fig = px.bar(
        city_df,
        x="City",
        y=temp_col,
        text=temp_col,
        color=temp_col,
        color_continuous_scale="Viridis",
        labels={temp_col: "Temperature (°C)", "City": "City"}
    )
    fig.update_traces(
        texttemplate='<b>%{text:.1f}°C</b>', 
        textposition='outside',
        textfont=dict(size=13, color="#FFFFFF" if theme == "Dark" else "#0F172A"),
        marker=dict(line=dict(color='#38BDF8', width=1.5))
    )
    apply_theme(fig, "📍 Live Temperature Comparison Across Cities (°C)", theme=theme)
    fig.update_layout(coloraxis_showscale=False)
    return fig

def plot_hourly_temperature_forecast(forecast_df: pd.DataFrame, theme="Dark", **kwargs):
    """Line chart showing hourly temperature forecast."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df["Time"],
        y=forecast_df["Temperature"],
        mode='lines+markers+text',
        name='Temp (°C)',
        text=[f"{t:.1f}" for t in forecast_df["Temperature"]],
        textposition="top center",
        textfont=dict(color="#FFFFFF" if theme == "Dark" else "#0F172A", size=11),
        line=dict(color="#38BDF8" if theme == "Dark" else "#0284C7", width=3.5, shape='spline'),
        marker=dict(size=8, color="#0284C7"),
        fill='tozeroy',
        fillcolor='rgba(56, 189, 248, 0.15)' if theme == "Dark" else 'rgba(2, 132, 199, 0.10)'
    ))

    apply_theme(fig, "⏱️ Hourly Temperature Forecast (°C)", theme=theme)
    fig.update_layout(
        xaxis=dict(tickangle=-45, title="Time"),
        yaxis=dict(title="Temperature (°C)")
    )
    return fig

def plot_humidity_wind_comparison(city_df: pd.DataFrame, theme="Dark", **kwargs):
    """Grouped bar chart showing Humidity % and Wind Speed (km/h)."""
    hum_col = "Humidity (%)" if "Humidity (%)" in city_df.columns else "Humidity"
    wind_col = "Wind Speed (km/h)" if "Wind Speed (km/h)" in city_df.columns else "Wind_Speed"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=city_df["City"],
        y=city_df[hum_col],
        name="Humidity (%)",
        marker_color="#38BDF8" if theme == "Dark" else "#0284C7",
        text=[f"{h}%" for h in city_df[hum_col]],
        textposition="auto",
        textfont=dict(color="#FFFFFF", size=11)
    ))

    fig.add_trace(go.Bar(
        x=city_df["City"],
        y=city_df[wind_col],
        name="Wind Speed (km/h)",
        marker_color="#34D399" if theme == "Dark" else "#0D9488",
        text=[f"{w} km/h" for w in city_df[wind_col]],
        textposition="auto",
        textfont=dict(color="#FFFFFF", size=11)
    ))

    apply_theme(fig, "💨 Humidity & Wind Speed Comparison", theme=theme)
    fig.update_layout(
        barmode='group',
        xaxis=dict(title="City"),
        yaxis=dict(title="Value")
    )
    return fig
