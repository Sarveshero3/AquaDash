
# river.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import re

# ------------------- DATA LOADER -------------------
@st.cache_data
def load_data():
    """Load river CSV."""
    return pd.read_csv("river.csv")

# ------------------- DATA PREPARATION -------------------
def create_mean_columns_river(df):
    """Create _Mean_YYYY columns from Min/Max."""
    pattern = r"^(.+?)_(Min|Max)_(\d{4})$"
    new_df = df.copy()
    pairs = {}
    for col in new_df.columns:
        m = re.match(pattern, col)
        if m:
            prefix, which, year = m.groups()
            pairs.setdefault((prefix, year), {})[which] = col

    for (prefix, year), mm in pairs.items():
        min_col, max_col = mm.get("Min"), mm.get("Max")
        if min_col and max_col:
            mean_col = f"{prefix}_Mean_{year}"
            new_df[min_col] = pd.to_numeric(new_df[min_col], errors="coerce")
            new_df[max_col] = pd.to_numeric(new_df[max_col], errors="coerce")
            new_df[mean_col] = (new_df[min_col] + new_df[max_col]) / 2.0
    return new_df

# ------------------- POTABILITY -------------------
def determine_river_potability_from_mean_cols(df):
    rules = {
        "pH": lambda x: 6.5 <= x <= 8.5,
        "Conductivity": lambda x: x <= 2000,
        "Temperature": lambda x: x <= 35,
        "BOD": lambda x: x <= 5,
        "Fecal_Coliform": lambda x: x <= 1,
    }
    pattern = r"^(.+?)_Mean_(\d{4})$"
    year_map = {}
    for col in df.columns:
        m = re.match(pattern, col)
        if m:
            param, yr = m.groups()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            year_map.setdefault(int(yr), {})[param] = df[col].mean(skipna=True)

    rows = []
    for year in sorted(year_map):
        param_vals = year_map[year]
        potable = True
        row = {"Year": year}
        for param, cond in rules.items():
            val = param_vals.get(param, float("nan"))
            row[param] = val
            if pd.isna(val) or not cond(val):
                potable = False
        row["Potable"] = "Yes" if potable else "No"
        rows.append(row)
    return pd.DataFrame(rows)

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

def get_river_yearly_data_from_mean_cols(df, prefix):
    pattern = rf"^{prefix}_Mean_(\d{{4}})$"
    data = []
    for col in df.columns:
        m = re.match(pattern, col)
        if m:
            yr = int(m.group(1))
            df[col] = pd.to_numeric(df[col], errors="coerce")
            data.append({"Year": yr, "Value": df[col].mean(skipna=True)})
    return pd.DataFrame(data).dropna(subset=["Value"]).sort_values("Year")

def get_river_prefixes(df):
    pattern = r"^(.+?)_Mean_(\d{4})$"
    return sorted({re.match(pattern, c).group(1) for c in df.columns if re.match(pattern, c)})

# ------------------- RENDER TAB -------------------
def render_tab(df):
    st.header("River Data Analysis")

    # Overview
    st.subheader("1) Overview")
    st.write("**Sample (first 5 rows):**")
    st.write(df.head())
    st.write(f"**Shape**: {df.shape[0]} rows x {df.shape[1]} columns")
    st.write("**Missing Values:**")
    st.write(df.isnull().sum().loc[lambda s: s > 0].sort_values(ascending=False))

    # Potability
    st.subheader("2) Potability by Year (Based on Mean Columns)")
    pot_df = determine_river_potability_from_mean_cols(df)
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
        param = st.selectbox("Select a River parameter", numeric_cols, key="river_parameter")
        yearly_df = get_yearly_data(df, param)
        if yearly_df.empty:
            st.write("No data available.")
        else:
            st.dataframe(yearly_df)
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(yearly_df["Year"], yearly_df["Value"], marker="o")
            ax.set_title(f"River: {param} by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Average Value")
            st.pyplot(fig, use_container_width=False)

    # Forecast
    st.subheader("4) 5-Year Forecast (Using Mean Columns)")
    prefixes = get_river_prefixes(df)
    if not prefixes:
        st.warning("No _Mean_ columns found.")
        return
    prefix = st.selectbox("Select parameter prefix", prefixes, key="river_prefix_forecast")
    model_name = st.selectbox(
        "Select forecast model", ["Decision Tree", "Random Forest", "Linear Regression"], key="river_model_forecast"
    )
    prefix_df = get_river_yearly_data_from_mean_cols(df, prefix)
    if prefix_df.empty:
        st.write("No data found for that prefix.")
    else:
        future_df = forecast_next_5_years(prefix_df, model_name=model_name)
        st.dataframe(future_df)
        st.download_button(
            "Download Forecast CSV",
            future_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{prefix}_river_forecast.csv",
            mime="text/csv",
        )
        combined = pd.concat([prefix_df, future_df])
        fig, ax = plt.subplots(figsize=(4, 3))
        last_hist = prefix_df["Year"].max()
        hist_mask = combined["Year"] <= last_hist
        fut_mask = combined["Year"] > last_hist
        ax.plot(combined.loc[hist_mask, "Year"], combined.loc[hist_mask, "Value"], marker="o", label="Historical")
        ax.plot(combined.loc[fut_mask, "Year"], combined.loc[fut_mask, "Value"], marker="o", label="Forecast")
        ax.set_title(f"{prefix} Forecast ({model_name}) - River")
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig, use_container_width=False)
