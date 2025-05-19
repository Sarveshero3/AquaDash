# main.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Local modules
import river as rv
import groundwater as gw
import comparative as comp   # ← NEW: comparative tab

# ──────────────────────────────────────────────────────────────
# Page configuration
st.set_page_config(
    page_title="💧 AquaDash – Local‑First Water‑Quality Explorer",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# Load the built-in datasets once and reuse
df_river_raw = rv.load_data()
df_river = rv.create_mean_columns_river(df_river_raw)

df_ground_raw = gw.load_data()
df_ground = gw.prepare_groundwater_data(df_ground_raw)

# ──────────────────────────────────────────────────────────────
# App layout – three tabs
tab_river, tab_ground, tab_compare = st.tabs(
    ["River Data", "Groundwater Data", "Comparative Analysis"]
)

with tab_river:
    rv.render_tab(df_river)

with tab_ground:
    gw.render_tab(df_ground)

with tab_compare:
    # Pass the default frames so users can start comparing
    comp.render_tab(df_river, df_ground)

# ──────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "_Note: River forecasting charts use the `_Mean_YYYY` columns derived "
    "from the raw river dataset._"
)
