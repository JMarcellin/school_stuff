import pandas as pd
import streamlit as st
import plotly.express as px
from prophet import Prophet
import plotly.graph_objects as go

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("C:/Users/ASUS/Documents/python_stuff_datasci/dashboard_stuff/BMW sales data (2010-2024) (1).csv")

st.set_page_config(page_title="BMW Worldwide Sales Dashboard", layout="wide")

# --------------------------------------------------
# KPI CARD FUNCTION
# --------------------------------------------------
def kpi_card(title, value):
    st.markdown(
        f"""
        <div style="
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            height: 130px;
            padding: 16px;
            border-radius: 10px;
            background-color: #f5f5f5;
            text-align: center;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.1);
            width: 100%;
        ">
            <p style="margin: 0; font-size: 16px; color: #555;">{title}</p>
            <p style="margin: 4px 0 0 0; font-size: 26px; font-weight: bold; color: #000;">
                {value}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# PAGE TITLE
# --------------------------------------------------
st.title("BMW Worldwide Sales Dashboard (2010–2024)")

# --------------------------------------------------
# TABS
# --------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Overall Statistics", "Filtered Statistics", "Sales Forecasting"])

# --------------------------------------------------
# TAB 1 — OVERALL STATISTICS
# --------------------------------------------------
with tab1:

    st.subheader("Overall Statistics (Full Dataset)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("Total Sales Volume", f"{df['Sales_Volume'].sum():,}")
    with col2:
        kpi_card("Average Price (USD)", f"${df['Price_USD'].mean():,.2f}")
    with col3:
        kpi_card("Average Mileage (KM)", f"{df['Mileage_KM'].mean():,.0f}")
    with col4:
        kpi_card("Average Engine Size (L)", f"{df['Engine_Size_L'].mean():.2f}")

    # Bar Chart — Total Sales by Region
    st.subheader("Total Sales by Region")
    region_sales = df.groupby("Region", as_index=False)["Sales_Volume"].sum()
    fig_bar = px.bar(
        region_sales,
        x="Region",
        y="Sales_Volume",
        color="Region",
        text="Sales_Volume",
        title="Total BMW Sales by Region"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Line Chart — Trend
    st.subheader("BMW Sales Trend Over Years by Region")
    trend_df = df.groupby(["Year", "Region"], as_index=False)["Sales_Volume"].sum()
    fig_trend = px.line(
        trend_df,
        x="Year",
        y="Sales_Volume",
        color="Region",
        markers=True,
        title="BMW Sales Trend Over Years by Region"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Preview full dataset
    st.subheader("Full Dataset Preview")
    st.dataframe(df, use_container_width=True)

# --------------------------------------------------
# TAB 2 — FILTERED STATISTICS
# --------------------------------------------------
with tab2:

    st.subheader("Filtered Statistics")

    # FILTERS ONLY SHOW IN THIS TAB
    with st.expander("Filter Options", expanded=True):

        selected_year = st.selectbox("Select Year", sorted(df["Year"].unique()))
        selected_region = st.multiselect("Region(s)", sorted(df["Region"].unique()), default=list(df["Region"].unique()))
        selected_model = st.multiselect("Model(s)", sorted(df["Model"].unique()), default=list(df["Model"].unique()))
        selected_fuel = st.multiselect("Fuel Type(s)", sorted(df["Fuel_Type"].unique()), default=list(df["Fuel_Type"].unique()))

    # APPLY FILTERS
    df_filtered = df[
        (df["Year"] == selected_year) &
        (df["Region"].isin(selected_region)) &
        (df["Model"].isin(selected_model)) &
        (df["Fuel_Type"].isin(selected_fuel))
    ]

    st.subheader(f"{len(df_filtered)} Records Found for Year {selected_year}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("Total Sales Volume", f"{df_filtered['Sales_Volume'].sum():,}")
    with col2:
        kpi_card("Average Price (USD)", f"${df_filtered['Price_USD'].mean():,.2f}")
    with col3:
        kpi_card("Average Mileage (KM)", f"{df_filtered['Mileage_KM'].mean():,.0f}")
    with col4:
        kpi_card("Average Engine Size (L)", f"{df_filtered['Engine_Size_L'].mean():.2f}")

    # Bar Chart — Filtered
    st.subheader("Total Sales by Region (Filtered)")
    region_sales_f = df_filtered.groupby("Region", as_index=False)["Sales_Volume"].sum()
    fig_bar_f = px.bar(
        region_sales_f,
        x="Region",
        y="Sales_Volume",
        color="Region",
        text="Sales_Volume",
        title="Total BMW Sales by Region (Filtered)"
    )
    st.plotly_chart(fig_bar_f, use_container_width=True)

    # Trend Chart — Filtered
    st.subheader("Sales Trend by Region (Filtered)")
    trend_df_f = df[
        (df["Region"].isin(selected_region)) &
        (df["Model"].isin(selected_model)) &
        (df["Fuel_Type"].isin(selected_fuel))
    ].groupby(["Year", "Region"], as_index=False)["Sales_Volume"].sum()

    fig_trend_f = px.line(
        trend_df_f,
        x="Year",
        y="Sales_Volume",
        color="Region",
        markers=True,
        title="Sales Trend Over Time (Filtered)"
    )
    st.plotly_chart(fig_trend_f, use_container_width=True)

    # Area Chart
    st.subheader("Regional Contribution Over Time")
    fig_area = px.area(
        trend_df_f,
        x="Year",
        y="Sales_Volume",
        color="Region",
        title="Sales Volume Contribution by Region"
    )
    st.plotly_chart(fig_area, use_container_width=True)

    # Treemap
    st.subheader("Sales Breakdown by Region & Model")
    treemap_df = df_filtered.groupby(["Region", "Model"], as_index=False)["Sales_Volume"].sum()
    fig_treemap = px.treemap(
        treemap_df,
        path=["Region", "Model"],
        values="Sales_Volume",
        title="BMW Sales Breakdown by Region & Model"
    )
    st.plotly_chart(fig_treemap, use_container_width=True)

    # Filtered dataset
    st.subheader("Filtered Dataset Preview")
    st.dataframe(df_filtered, use_container_width=True)

# --------------------------------------------------
# TAB 3 — SALES FORECASTING
# --------------------------------------------------
with tab3:

    st.subheader(" Sales Forecasting")

    forecast_region = st.selectbox("Select Region", sorted(df["Region"].unique()))
    forecast_years = st.slider("Years to Forecast", 1, 10, 5)

    df_region = df[df["Region"] == forecast_region].groupby("Year", as_index=False)["Sales_Volume"].sum()

    st.markdown(f"### Historical Sales — {forecast_region}")
    st.dataframe(df_region)

    ts = df_region.rename(columns={"Year": "ds", "Sales_Volume": "y"})
    ts["ds"] = pd.to_datetime(ts["ds"], format="%Y")

    model = Prophet(yearly_seasonality=True)
    model.fit(ts)

    future = model.make_future_dataframe(periods=forecast_years, freq="Y")
    forecast = model.predict(future)

    forecast_clean = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
    forecast_clean["ds"] = forecast_clean["ds"].dt.year

    last_year = df_region["Year"].max()
    forecast_future = forecast_clean[forecast_clean["ds"] > last_year]

    st.markdown("### Forecasted Sales Table")
    st.dataframe(forecast_future)

    # Forecast Plot
    fig_forecast = go.Figure()

    fig_forecast.add_trace(go.Scatter(
        x=ts["ds"], y=ts["y"], mode="lines+markers", name="Actual"
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast", line=dict(width=3)
    ))

    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_upper"], line=dict(width=0), showlegend=False
    ))
    fig_forecast.add_trace(go.Scatter(
        x=forecast["ds"], y=forecast["yhat_lower"], fill="tonexty", name="Confidence Interval", line=dict(width=0)
    ))

    fig_forecast.update_layout(
        title=f" Forecast for {forecast_region} ({forecast_years} years ahead)",
        xaxis_title="Year",
        yaxis_title="Sales Volume",
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(fig_forecast, use_container_width=True)

    # Trend Component
    st.markdown("### Trend Component")
    fig_trend = px.line(forecast, x="ds", y="trend", title="Underlying Trend")
    st.plotly_chart(fig_trend, use_container_width=True)