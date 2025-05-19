
# groundwater.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

# ------------------- DATA LOADER -------------------
@st.cache_data
def load_data():
    """Load groundwater Excel workbook."""
    return pd.read_excel("Modified_filled.xlsx")

# ------------------- DATA PREPARATION -------------------
def prepare_groundwater_data(df):
    new_df = df.copy()
    if "Years" in new_df.columns:
        new_df = new_df.rename(columns={"Years": "Year"})
    possible_cols = ["pH", "Conductivity", "Temperature", "BOD", "Fecal_Coliform"]
    for col in possible_cols:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
    return new_df

# ------------------- POTABILITY -------------------
def determine_potability_by_year(df, year_col="Year"):
    rules = {
        "pH": lambda x: 6.5 <= x <= 8.5,
        "Conductivity": lambda x: x <= 2000,
        "Temperature": lambda x: x <= 35,
        "BOD": lambda x: x <= 5,
        "Fecal_Coliform": lambda x: x <= 1,
    }
    if year_col not in df.columns:
        return pd.DataFrame()
    grouped = df.groupby(year_col).mean(numeric_only=True).reset_index()
    rows = []
    for _, row in grouped.iterrows():
        y = int(row[year_col])
        potable = True
        measures = {}
        for param, cond in rules.items():
            val = row.get(param, float("nan"))
            measures[param] = val
            if pd.isna(val) or not cond(val):
                potable = False
        rows.append({"Year": y, **measures, "Potable": "Yes" if potable else "No"})
    return pd.DataFrame(rows).sort_values("Year")

# ------------------- HELPERS -------------------
def get_numeric_columns_excluding_year(df, year_col="Year"):
    return [c for c in df.columns if c != year_col and pd.api.types.is_numeric_dtype(df[c])]

def get_yearly_data(df, col_name, year_col="Year"):
    if year_col not in df.columns or col_name not in df.columns:
        return pd.DataFrame(columns=["Year", "Value"])
    grouped = (
        df.dropna(subset=[col_name, year_col])
        .groupby(year_col)[col_name]
        .mean()
        .reset_index(name="Value")
        .sort_values(year_col)
    )
    grouped[year_col] = grouped[year_col].astype(int)
    return grouped.rename(columns={year_col: "Year"})

def forecast_next_5_years(yearly_df, model_name="Decision Tree"):
    if yearly_df.empty:
        return pd.DataFrame(columns=["Year", "Value"])
    X = yearly_df[["Year"]].values
    y = yearly_df["Value"].values
    if model_name == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        model = LinearRegression()
    model.fit(X, y)
    last_year = int(yearly_df["Year"].max())
    future_years = np.arange(last_year + 1, last_year + 6)
    future_vals = model.predict(future_years.reshape(-1, 1))
    return pd.DataFrame({"Year": future_years, "Value": future_vals})

# ------------------- RENDER TAB -------------------
def render_tab(df):
    st.header("Groundwater Data Analysis")

    # Overview
    st.subheader("1) Overview")
    st.write("**Sample (first 5 rows):**")
    st.write(df.head())
    st.write(f"**Shape**: {df.shape[0]} rows x {df.shape[1]} columns")
    st.write("**Missing Values:**")
    st.write(df.isnull().sum().loc[lambda s: s > 0].sort_values(ascending=False))

    # Potability
    st.subheader("2) Potability by Year")
    pot_df = determine_potability_by_year(df)
    if pot_df.empty:
        st.warning("No potability data available.")
    else:
        st.dataframe(pot_df)

    # Yearly analysis
    st.subheader("3) Yearly Analysis for Selected Parameter")
    numeric_cols = get_numeric_columns_excluding_year(df)
    if not numeric_cols:
        st.warning("No numeric columns detected.")
    else:
        param = st.selectbox("Select a Groundwater parameter", numeric_cols, key="gw_parameter")
        yearly_df = get_yearly_data(df, param)
        if yearly_df.empty:
            st.write("No data available.")
        else:
            st.dataframe(yearly_df)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(yearly_df["Year"], yearly_df["Value"], marker="o")
            ax.set_title(f"Groundwater: {param} by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Value")
            st.pyplot(fig, use_container_width=False)

    # Forecast
    st.subheader("4) 5-Year Forecast")
    if not numeric_cols:
        st.warning("No numeric columns to forecast.")
        return
    fc_param = st.selectbox("Select parameter to forecast", numeric_cols, key="gw_forecast")
    model_name = st.selectbox(
        "Select forecast model",
        ["Decision Tree", "Random Forest", "Linear Regression"],
        key="ground_model",
    )
    yearly_df = get_yearly_data(df, fc_param)
    if yearly_df.empty:
        st.write("No data to forecast.")
    else:
        future_df = forecast_next_5_years(yearly_df, model_name=model_name)
        st.dataframe(future_df)
        st.download_button(
            "Download Forecast CSV",
            future_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{fc_param}_groundwater_forecast.csv",
            mime="text/csv",
        )
        combined = pd.concat([yearly_df, future_df])
        fig, ax = plt.subplots(figsize=(4, 3))
        last_hist = yearly_df["Year"].max()
        hist_mask = combined["Year"] <= last_hist
        fut_mask = combined["Year"] > last_hist
        ax.plot(combined.loc[hist_mask, "Year"], combined.loc[hist_mask, "Value"], marker="o", label="Historical")
        ax.plot(combined.loc[fut_mask, "Year"], combined.loc[fut_mask, "Value"], marker="o", label="Forecast")
        ax.set_title(f"{fc_param} Forecast ({model_name}) - Groundwater")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig, use_container_width=False)
