# comparative.py  ───────────────────────────────────────────────
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional, Tuple

# ────── I/O helpers ────────────────────────────────────────────
@st.cache_data
def _read_upload(file) -> Optional[pd.DataFrame]:
    if file is None:
        return None
    try:
        if file.name.lower().endswith((".csv", ".txt")):
            return pd.read_csv(file)
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ {file.name}: {e}")
        return None


def _common_cols(dfs: List[pd.DataFrame]) -> List[str]:
    """Columns that appear in **all** dataframes."""
    cols = set(dfs[0].columns)
    for df in dfs[1:]:
        cols &= set(df.columns)
    return sorted(cols)


def _numeric_common(dfs: List[pd.DataFrame], exclude: List[str]) -> List[str]:
    out = []
    for col in _common_cols(dfs):
        if col in exclude:
            continue
        if all(pd.api.types.is_numeric_dtype(df[col]) for df in dfs):
            out.append(col)
    return out


# ────── MAIN TAB ───────────────────────────────────────────────
def render_tab(*default_dfs: pd.DataFrame) -> None:
    st.header("Comparative Analysis (1 – 5 datasets)")

    # — 1. UPLOADS —
    uploaded = st.file_uploader(
        "Upload up to 5 datasets (CSV or Excel):",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )
    dataframes, labels = [], []                                    # user files
    for f in uploaded:
        df = _read_upload(f)
        if df is not None:
            dataframes.append(df)
            labels.append(f.name)          # remember each file’s name

    if not dataframes:                                             # fall-back
        dataframes = list(default_dfs)[:5]
        labels      = [f"Dataset {i+1}" for i in range(len(dataframes))]
        st.info("Using built-in datasets – upload files to replace them.")

    dataframes = [df for df in dataframes if df is not None][:5]
    n_df = len(dataframes)
    if n_df == 0:
        st.stop()

    # — 2. QUICK INFO + OPTIONAL COLUMN CLEANUP —
    st.subheader("Basic information & column pruning")

    thr = st.slider(
        "Threshold to flag a column as ‘mostly missing / zeros’",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="A column is suggested for removal if > threshold fraction of "
             "its values are NaN or zero.",
    )

    cleaned_dfs: List[pd.DataFrame] = []
    for i, df in enumerate(dataframes, 1):
        with st.expander(f"{labels[i-1]}: {df.shape[0]} rows × {df.shape[1]} cols"):
            st.write(df.head())

            # Suggest columns to drop
            drop_cand = [
                c
                for c in df.columns
                if (df[c].isna().mean() > thr) or
                   ((df[c] == 0).mean() > thr)
            ]
            drop_cols = st.multiselect(
                f"Remove sparse/zero columns from Dataset {i}?",
                drop_cand,
                key=f"drop_{i}",
            )
            cleaned_dfs.append(df.drop(columns=drop_cols))

    # — 3. TREND GRAPH (single dataset) —
    st.subheader("Single-dataset trend")

    idx_trend = st.selectbox(
        "Choose dataset for trend plot",
        list(range(1, n_df + 1)),
        format_func=lambda i: labels[i-1],   # show filename
    )
    df_trend = cleaned_dfs[idx_trend - 1]

    year_cols = [c for c in df_trend.columns if "year" in c.lower()]
    if not year_cols:
        st.warning("Trend plot skipped – chosen dataset has no column containing ‘year’.")
    else:
        y_col = st.selectbox("Year column", year_cols, key="year_trend")
        num_cols = [
            c for c in df_trend.columns
            if c != y_col and pd.api.types.is_numeric_dtype(df_trend[c])
        ]
        if num_cols:
            param_trend = st.selectbox("Element / parameter", num_cols, key="param_trend")
            _plot_trend(df_trend, y_col, param_trend)
        else:
            st.warning("No numeric columns to plot a trend.")

    # — 4. MULTI-DATASET COMPARISON —
    st.subheader("Compare 2 – 5 datasets")

    # user picks which datasets to include
    idx_compare = st.multiselect(
        "Select datasets to compare",
        list(range(1, n_df + 1)),
        default=list(range(1, min(n_df, 2) + 1)),
        format_func=lambda i: labels[i-1],
        help="Pick at least two.",
    )
    if len(idx_compare) >= 2:
        dfs_cmp = [cleaned_dfs[i - 1] for i in idx_compare]
        year_cand = [c for c in _common_cols(dfs_cmp) if "year" in c.lower()]
        if not year_cand:
            st.warning("No common ‘year’ column across the selected datasets.")
            return
        y_col_cmp = st.selectbox("Year column (common)", year_cand, key="year_cmp")
        num_common = _numeric_common(dfs_cmp, exclude=[y_col_cmp])
        if not num_common:
            st.warning("No common numeric element across the selected datasets.")
            return
        param_cmp = st.selectbox("Element / parameter", num_common, key="param_cmp")

        chart_type = st.radio("Graph style", ["Line", "Scatter", "Bar"], horizontal=True)

        _plot_multi(
            dfs_cmp, y_col_cmp, param_cmp, chart_type,
            [labels[i-1] for i in idx_compare]   # pass names instead of numbers
        )


# ────── PLOTTING HELPERS ───────────────────────────────────────
def _plot_trend(df: pd.DataFrame, year: str, param: str) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    tmp = (
        df[[year, param]]
        .dropna()
        .groupby(year)[param]
        .mean()
        .reset_index()
        .sort_values(year)
    )
    ax.plot(tmp[year], tmp[param], marker="o")
    ax.set_title(f"{param} over Years")
    ax.set_xlabel("Year")
    ax.set_ylabel(param)
    st.pyplot(fig, use_container_width=False)


def _plot_multi(
    dfs: List[pd.DataFrame],
    year: str,
    param: str,
    style: str,
    labels: List[str],
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    palette = plt.cm.get_cmap("tab10", len(dfs))

    # Aggregate per dataset
    agg: List[Tuple[pd.Series, str]] = []
    for i, df in enumerate(dfs):
        tmp = (
            df[[year, param]]
            .dropna()
            .groupby(year)[param]
            .mean()
            .reset_index()
            .sort_values(year)
        )
        agg.append((tmp, labels[i]))

    if style == "Line":
        for i, (d, label) in enumerate(agg):
            ax.plot(d[year], d[param], marker="o", label=label, color=palette(i))
    elif style == "Scatter":
        for i, (d, label) in enumerate(agg):
            ax.scatter(d[year], d[param], label=label, color=palette(i))
    else:  # bar
        # Align years
        years = sorted(
            set.intersection(*[set(d[year]) for d, _ in agg])
        )
        x = np.arange(len(years))
        width = 0.8 / len(agg)
        for i, (d, label) in enumerate(agg):
            series = d.set_index(year).reindex(years)[param]
            ax.bar(x + i * width - 0.4 + width / 2, series, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(years)

    ax.set_title(f"{param} – Comparison")
    ax.set_xlabel("Year")
    ax.set_ylabel(param)
    ax.legend(fontsize="small", ncol=2)
    st.pyplot(fig, use_container_width=False)


# ───────────────────────────────────────────────────────────────
