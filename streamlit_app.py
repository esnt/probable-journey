
# streamlit_app.py — Baby Names EDA (optimized, no sidebar)
# Run: streamlit run streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import zipfile

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*observed=False.*")
# -------------------------
# Page config & light CSS
# -------------------------
st.set_page_config(page_title="Baby Names Explorer (EDA)", page_icon="👶", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 95% !important; padding-left: 2rem; padding-right: 2rem; }
      .stMetric { text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Paths (assumes repo contains NameData/namesbystate.zip)
# -------------------------
APP_DIR = Path(__file__).parent
STATE_ZIP = (APP_DIR / "NameData" / "namesbystate.zip").resolve()

# -------------------------
# Helpers: dtypes & caching
# -------------------------
def _optimize(df: pd.DataFrame) -> pd.DataFrame:
    if "year" in df: df["year"] = pd.to_numeric(df["year"], downcast="integer")
    if "count" in df: df["count"] = pd.to_numeric(df["count"], downcast="integer")
    for c in ("sex","state"):
        if c in df: df[c] = df[c].astype("category")
    if "name" in df:
        df["name"] = df["name"].astype("string")
    return df

@st.cache_resource(show_spinner=True)
def load_state_data(zip_path: Path) -> pd.DataFrame:
    if not zip_path.exists():
        raise FileNotFoundError(f"State ZIP not found at {zip_path}")
    dfs = []
    with zipfile.ZipFile(zip_path, "r") as z:
        files = [f for f in z.namelist() if f.endswith(".TXT")]
        for file in files:
            with z.open(file) as f:
                df = pd.read_csv(f, header=None, names=['state','sex','year','name','count'])
                dfs.append(df)
    state_df = pd.concat(dfs, ignore_index=True)
    state_df["state"] = state_df["state"].str.upper()
    return _optimize(state_df)

@st.cache_data(show_spinner=False)
def aggregate_national_from_state(state_df: pd.DataFrame) -> pd.DataFrame:
    nat = (state_df.groupby(["name","sex","year"], as_index=False, observed=True)["count"].sum())
    totals = nat.groupby(["year","sex"], observed=True)["count"].transform("sum")
    nat = nat.assign(pct = nat["count"] / totals)
    return _optimize(nat)

@st.cache_data(show_spinner=False, max_entries=64)
def add_per_10k(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pct" not in out.columns:
        totals = out.groupby(["year","sex"])["count"].transform("sum")
        out["pct"] = out["count"] / totals
    out["per_10k"] = out["pct"] * 10_000
    return out

def compute_ranks(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rank"] = df.groupby(["year","sex"])["count"].rank(method="dense", ascending=False).astype("int32")
    return df

def fm_pivot(focus: pd.DataFrame) -> pd.DataFrame:
    pv = (focus.set_index(["year","sex"])["count"]
                .unstack(fill_value=0)
                .reindex(columns=["F","M"], fill_value=0)
                .rename(columns={"F":"F_count","M":"M_count"}))
    pv.index = pv.index.astype(int)
    pv["total"] = pv[["F_count","M_count"]].sum(axis=1)
    pv["female_share"] = np.where(pv["total"]>0, pv["F_count"]/pv["total"], np.nan)
    return pv

# Letters heatmap pivot
@st.cache_data(show_spinner=False, max_entries=64)
def letters_pivot(nat_df: pd.DataFrame, sex: str, metric: str, start: int, end: int) -> pd.DataFrame:
    base = nat_df[(nat_df["year"]>=start) & (nat_df["year"]<=end)]
    if sex in ("F","M"):
        base = base[base["sex"]==sex]
    base = base.assign(first_letter = base["name"].str[0].str.upper())
    base = base[base["first_letter"].str.match(r"[A-Z]")]
    totals = base.groupby("year", observed=True)["count"].sum().rename("year_total")
    by_ly = base.groupby(["year","first_letter"], observed=True)["count"].sum().reset_index().merge(
        totals, on="year", how="left"
    )
    if metric == "Percent of births":
        by_ly["value"] = np.where(by_ly["year_total"]>0, by_ly["count"]/by_ly["year_total"], 0.0)
    else:
        by_ly["value"] = by_ly["count"]
    heat = by_ly.pivot(index="first_letter", columns="year", values="value").fillna(0)
    return heat

def thinking_prompts(title: str, questions: list[str], key: str):
    st.divider()
    with st.expander(title, expanded=False):
        st.markdown("\n".join([f"- {q}" for q in questions]))


# -------------------------
# Load data (state → national)
# -------------------------
try:
    st_df = load_state_data(STATE_ZIP)
    nat_df = aggregate_national_from_state(st_df)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

YR_MIN, YR_MAX = int(nat_df["year"].min()), int(nat_df["year"].max())

# -------------------------
# Tabs
# -------------------------


tab1, tab5, tab7, tab2, tab6 = st.tabs([
    "Name Trends", "Geographical Trends", "Gendered Names",
    "Compare Names", "First Letters"
])

# ------------------------
# Tab 1: Name Trends
# ------------------------



with tab1:
    st.subheader("Trends for a Name Over Time")

    # Controls inside the tab
    c1, c2 = st.columns([2, 1])
    with c1:
        name = st.text_input("Type a name (case-insensitive)", value="Emma").strip()
    with c2:
        sex_choice = st.selectbox("Sex", ["Both", "F", "M"], index=0)

    norm = st.selectbox("Normalize by:", ["raw counts", "per 10,000 births", "percent of births"])

    # Use your single national dataset here (rename nat_df to whatever your variable is)
    df = nat_df.copy()
    df["name_lower"] = df["name"].str.lower()
    focus = df[df["name_lower"] == name.lower()].copy()

    # Add per_10k (and pct if missing)
    focus = add_per_10k(focus)

    # Apply in-tab sex filter for plotting
    if sex_choice in ["F", "M"]:
        focus = focus[focus["sex"] == sex_choice]

    if focus.empty:
        st.warning("Name not found in selected filters.")
    else:
        if norm == "raw counts":
            y, label = "count", "Count"
        elif norm == "per 10,000 births":
            y, label = "per_10k", "Per 10,000"
        else:  # "percent of births"
            y, label = "pct", "Percent"

        fig = px.line(
            focus,
            x="year",
            y=y,
            color="sex",  # fine even if only one sex is present
            markers=True,
            labels={"year": "Year", y: label, "sex": "Sex"},
            title=f'Popularity of “{name.title()}” over time'
        )
        if y == "pct":
            fig.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig, use_container_width=True)

    thinking_prompts(
    "Digging Deeper",
    [
        "Does the shape suggest a name fad (sharp peak) or a long, steady climb/decline?",
        "How does the pattern differ by sex? Is one driving the trend?",
        "Is the change more striking in *percent* than in *raw counts*?",
        "What external events (pop culture, famous people) could explain inflection points?"
    ],
    key="t1_prompts",
    )
# ------------------------
# Tab 2: Compare Names
# ------------------------
with tab2:
    st.subheader("Compare Multiple Names and Ranks")

    c1, c2 = st.columns([2,1])
    with c1:
        names = st.text_input("Enter names (comma-separated)",
                              value="Emma, Olivia, Sophia, Isabella").strip()
    with c2:
        sex_choice = st.radio("Sex", ["F", "M"], horizontal=True, key='t2_sex')

    compare_mode = st.selectbox("Compare by",
                            ["count", "per 10,000", "percent", "rank"],
                            index=0)

    # Use your single national dataset
    df = nat_df.copy()
    # Restrict to ONE sex to keep charts clean
    df = df[df["sex"] == sex_choice].copy()

    # Add per_10k (and pct if missing), then ranks
    df = add_per_10k(df)
    df = compute_ranks(df)

    sel = [n.strip().lower() for n in names.split(",") if n.strip()]
    df["name_lower"] = df["name"].str.lower()
    plot_df = df[df["name_lower"].isin(sel)].copy()

    if plot_df.empty:
        st.info("No matches for these names with current settings.")
    else:
        if compare_mode == "count":
            y, title = "count", "Counts over time"
        elif compare_mode == "per 10,000":
            y, title = "per_10k", "Per 10,000 births over time"
        elif compare_mode == "percent":
            y, title = "pct", "Percent of births over time"
        else:  # rank
            y, title = "rank", "Rank over time (1 = most common)"

        fig = px.line(
            plot_df,
            x="year",
            y=y,
            color="name",           # color only by name; no line dashing by sex
            labels={"year": "Year", y: y.replace("_", " ").title(), "name": "Name"},
            title=title
        )
        if compare_mode == "percent":
            fig.update_yaxes(tickformat=".1%")
        if compare_mode == "rank":
            fig.update_yaxes(autorange="reversed")

        st.plotly_chart(fig, use_container_width=True)
    thinking_prompts(
    "Digging Deeper",
    [
        "Which name overtakes the others first? Around what year?",
        "Does the ranking change if you view *percent* vs *per 10,000*?",
        "Are the lines converging or diverging recently (signal of a new trend)?",
        "If you extended the lines, what would you predict for the next 5 years?"
    ],
    key="t2_prompts",
    )


# ------------------------
# Tab 5: Geo: Name Map — choropleth & animation + logging
# ------------------------
with tab5:
    st.subheader("Where is a name most popular? (State choropleth)")

    if st_df is None or st_df.empty:
        st.info("State-level data not found. Add `NameData/namesbystate.zip` to enable this view.")
    else:
        c1, c2, c3 = st.columns([2,1,1])
        with c1:
            g_name = st.text_input("Name", value="Olivia", key="geo_name").strip()
        with c2:
            g_sex = st.radio("Sex", options=["F", "M", "Both"], horizontal=True, key="geo_sex")
        with c3:
            g_metric = st.selectbox(
                "Metric",
                ["Raw count", "Percent of births (sex)", "Per 10,000 births (sex)"],
                index=1, key="geo_metric"
            )
        with st.expander("Advanced year options"):
            # Choose a year range FIRST so we can base the color scale on it
            st.markdown("#### Animate over years")
            g_range = st.slider(
                "Year range (animation)",
                min_value=int(st_df["year"].min()),
                max_value=int(st_df["year"].max()),
                value=(int(st_df["year"].min()), int(st_df["year"].max())),
                key="geo_anim_range"
            )

            # Color scale options
            c4, c5 = st.columns([1,1])
            with c4:
                lock_scale = st.checkbox("Lock color scale across years (recommended)", value=True, key="geo_lock")
            with c5:
                clip99 = st.checkbox("Clip to 99th percentile (reduce outlier effect)", value=True, key="geo_clip")

            # Build a DF over the chosen range (used both for animation + to compute vmin/vmax)
            if g_range[0] >= g_range[1]:
                st.info("Choose a wider year range for the animation.")
                st.stop()

            anim = st_df[st_df["year"].between(*g_range)].copy()
            if g_sex in ["F", "M"]:
                anim = anim[anim["sex"] == g_sex]

            # name counts per state-year
            name_ct = (
                anim[anim["name"].str.lower() == g_name.lower()]
                .groupby(["year", "state"], as_index=False)["count"].sum()
                .rename(columns={"count": "name_count"})
            )
            # totals per state-year (within selected sex or both)
            totals_anim = (
                anim.groupby(["year", "state"])["count"]
                .sum()
                .rename("state_sex_total")
                .reset_index()
            )

            anim_df = totals_anim.merge(name_ct, on=["year", "state"], how="left")
            anim_df["name_count"] = anim_df["name_count"].fillna(0)

            # Compute the metric
            if g_metric == "Raw count":
                anim_df["value"] = anim_df["name_count"]
            elif g_metric == "Percent of births (sex)":
                anim_df["value"] = np.where(anim_df["state_sex_total"] > 0,
                                            anim_df["name_count"] / anim_df["state_sex_total"], 0.0)
            else:  # Per 10000
                anim_df["value"] = np.where(anim_df["state_sex_total"] > 0,
                                            anim_df["name_count"] / anim_df["state_sex_total"] * 10_000, 0.0)

            # ---- FIXED COLOR SCALE CALCULATION ----
            if lock_scale:
                if g_metric == "Percent of births (sex)":
                    vmin = 0.0
                    vmax = float(anim_df["value"].quantile(0.99) if clip99 else anim_df["value"].max())
                    # guard if all zeros
                    if vmax <= 0:
                        vmax = 0.01
                else:
                    vmin = float(anim_df["value"].min())
                    vmax = float(anim_df["value"].quantile(0.99) if clip99 else anim_df["value"].max())
                    if vmax <= vmin:
                        vmax = vmin + (1 if g_metric == "Raw count" else 0.1)
                fixed_range = (vmin, vmax)
            else:
                fixed_range = None
            # ---------------------------------------


        # --- Single-year static map (use SAME range_color) ---
            st.markdown("#### Single-year map")
            g_year = st.selectbox("Year", sorted(st_df["year"].unique()), index=len(st_df["year"].unique())-1)

        base_year_df = st_df[st_df["year"] == g_year].copy()
        if g_sex in ["F", "M"]:
            work = base_year_df[base_year_df["sex"] == g_sex].copy()
        else:
            work = base_year_df.copy()

        name_mask = work["name"].str.lower() == g_name.lower()
        name_by_state = (
            work[name_mask]
            .groupby("state", as_index=False)["count"].sum()
            .rename(columns={"count": "name_count"})
        )

        totals_by_state_sex = (
            base_year_df.groupby(["state", "sex"])["count"].sum()
            .rename("state_sex_total").reset_index()
        )
        if g_sex in ["F", "M"]:
            totals = totals_by_state_sex[totals_by_state_sex["sex"] == g_sex].drop(columns="sex")
        else:
            totals = (
                totals_by_state_sex.groupby("state")["state_sex_total"]
                .sum().rename("state_sex_total").reset_index()
            )

        plot_df = totals.merge(name_by_state, on="state", how="left")
        plot_df["name_count"] = plot_df["name_count"].fillna(0)
        if g_metric == "Raw count":
            plot_df["value"] = plot_df["name_count"]
            color_title = f"{g_name.title()} — count"
        elif g_metric == "Percent of births (sex)":
            plot_df["value"] = np.where(plot_df["state_sex_total"] > 0,
                                        plot_df["name_count"] / plot_df["state_sex_total"], 0.0)
            color_title = f"{g_name.title()} — % of births ({g_sex.lower() if g_sex!='Both' else 'both sexes'})"
        else:
            plot_df["value"] = np.where(plot_df["state_sex_total"] > 0,
                                        plot_df["name_count"] / plot_df["state_sex_total"] * 1_000_000, 0.0)
            color_title = f"{g_name.title()} — per 10,000 births ({g_sex.lower() if g_sex!='Both' else 'both sexes'})"

        fig_static = px.choropleth(
            plot_df,
            locations="state",
            locationmode="USA-states",
            color="value",
            scope="usa",
            hover_data={"state": True, "name_count": True, "state_sex_total": True, "value": True},
            title=f"{color_title} in {g_year}",
            color_continuous_scale="Viridis",
            range_color=fixed_range  # <<< lock the scale here
        )
        if g_metric == "Percent of births (sex)":
            fig_static.update_coloraxes(colorbar_tickformat=".1%")
        st.plotly_chart(fig_static, use_container_width=True)
        st.caption("Locked color scale uses the selected year range above for fair comparisons.")

        # --- Animated map (reuse SAME range_color) ---
        fig_anim = px.choropleth(
            anim_df,
            locations="state",
            locationmode="USA-states",
            color="value",
            scope="usa",
            hover_data={"year": True, "state": True, "name_count": True, "state_sex_total": True, "value": True},
            animation_frame="year",
            title=f"{g_name.title()} — {g_metric} ({g_sex}) over years {g_range[0]}–{g_range[1]}",
            color_continuous_scale="Viridis",
            range_color=fixed_range  # <<< lock the scale here too
        )
        if g_metric == "Percent of births (sex)":
            fig_anim.update_coloraxes(colorbar_tickformat=".1%")
        st.plotly_chart(fig_anim, use_container_width=True)

    thinking_prompts(
    "Digging Deeper",
    [
        "Is the name concentrated in regions (e.g., coasts vs. heartland)?",
        "Does *percent of births* tell a different story than raw counts?",
        "When you animate, do waves move (south → north, east → west)?",
        "What hypotheses could explain these regional patterns?"
    ],
    key="t5_prompts",
    )

# ------------------------
# Tab 6: Letters & Sounds
# ------------------------
with tab6:
    st.subheader("A look at the first letter popularity of names")

    # ---------- Controls ----------
    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        ls_metric = st.radio("Metric", ["Percent of births", "Raw counts"], horizontal=False, key="t6_metric")
    with c2:
        yr_min_all, yr_max_all = int(nat_df["year"].min()), int(nat_df["year"].max())
        default_start = max(1950, yr_min_all)
        default_end = yr_max_all
        ls_start, ls_end = st.slider(
            "Year range",
            min_value=yr_min_all,
            max_value=yr_max_all,
            value=(default_start, default_end),
            key="t6_range"
        )

    order_mode = st.selectbox(
        "Sort letters by:",
        ["Alphabetical", "Total popularity in range", "Total popularity in last year of range"],
        index=1,
        key="t6_order"
    )

    # ---------- Prep data ----------
    base = nat_df[nat_df["year"].between(ls_start, ls_end)].copy()
    if base.empty:
        st.info("No data for this selection.")
        st.stop()

    base["first_letter"] = base["name"].str[0].str.upper()
    base = base[base["first_letter"].str.match(r"[A-Z]")]

    # Split by sex
    baseF = base[base["sex"] == "F"].copy()
    baseM = base[base["sex"] == "M"].copy()

    # Totals per year per sex
    totalsF = baseF.groupby("year")["count"].sum().rename("year_total").reset_index()
    totalsM = baseM.groupby("year")["count"].sum().rename("year_total").reset_index()

    def letter_year(df, totals):
        out = (df.groupby(["year", "first_letter"])["count"].sum().reset_index()
                 .merge(totals, on="year", how="left"))
        if ls_metric == "Percent of births":
            out["value"] = np.where(out["year_total"] > 0, out["count"] / out["year_total"], 0.0)
            fmt_percent = True
            color_title = "Share of births"
        else:
            out["value"] = out["count"]
            fmt_percent = False
            color_title = "Count"
        return out, fmt_percent, color_title

    byF, fmt_percent, color_title = letter_year(baseF, totalsF)
    byM, _, _ = letter_year(baseM, totalsM)

    # Pivot to heatmaps
    heatF = byF.pivot(index="first_letter", columns="year", values="value").fillna(0)
    heatM = byM.pivot(index="first_letter", columns="year", values="value").fillna(0)

    # Determine a COMMON letter order so rows line up
    all_letters = sorted(set(heatF.index).union(set(heatM.index)))
    # Build a combined matrix to rank letters (both sexes together)
    combined = (heatF.reindex(all_letters, fill_value=0) + heatM.reindex(all_letters, fill_value=0))

    if order_mode == "Alphabetical":
        letters_order = sorted(all_letters)
    elif order_mode == "Total popularity in range":
        letters_order = combined.sum(axis=1).sort_values(ascending=False).index.tolist()
    else:  # "By last year in range (both sexes)"
        last_year = combined.columns.max()
        if last_year in combined.columns:
            letters_order = combined[last_year].sort_values(ascending=False).index.tolist()
        else:
            letters_order = combined.sum(axis=1).sort_values(ascending=False).index.tolist()

    heatF = heatF.reindex(letters_order, fill_value=0)
    heatM = heatM.reindex(letters_order, fill_value=0)

    # ---------- Heatmaps side-by-side with SHARED color scale ----------
    st.markdown("### First-letter heatmaps (Female vs Male)")
    # Compute a stable color range across both
    if fmt_percent:
        vmin = 0.0
        vmax = float(np.nanpercentile(np.concatenate([heatF.values.ravel(), heatM.values.ravel()]), 99))
        if vmax <= 0: vmax = 0.01
        range_color = (vmin, vmax)
    else:
        vmin = float(min(heatF.values.min(), heatM.values.min()))
        vmax = float(np.nanpercentile(np.concatenate([heatF.values.ravel(), heatM.values.ravel()]), 99))
        if vmax <= vmin: vmax = vmin + 1.0
        range_color = (vmin, vmax)

    colF, colM = st.columns(2)
    with colF:
        figF = px.imshow(
            heatF.values,
            x=heatF.columns.astype(int),
            y=heatF.index,
            labels=dict(x="Year", y="First letter", color=color_title),
            aspect="auto",
            height=720,
            color_continuous_scale="Viridis",
            zmin=range_color[0], zmax=range_color[1]
        )
        if fmt_percent: figF.update_coloraxes(colorbar_tickformat=".0%")
        figF.update_layout(title="Female")
        st.plotly_chart(figF, use_container_width=True)

    with colM:
        figM = px.imshow(
            heatM.values,
            x=heatM.columns.astype(int),
            y=heatM.index,
            labels=dict(x="Year", y="First letter", color=color_title),
            aspect="auto",
            height=720,
            color_continuous_scale="Viridis",
            zmin=range_color[0], zmax=range_color[1]
        )
        if fmt_percent: figM.update_coloraxes(colorbar_tickformat=".0%")
        figM.update_layout(title="Male")
        st.plotly_chart(figM, use_container_width=True)

    # ---------- Top letters bars (selected year) ----------
    # ---------- Top letters bars (selected year) ----------
    st.markdown("### Top letters in a selected year (Female vs Male)")

    # Build the year options from both heatmaps and guard for empties
    years_opts = sorted(
        set(heatF.columns.astype(int).tolist()) |
        set(heatM.columns.astype(int).tolist())
    )

    if not years_opts:
        st.info("No years available for selection in this range.")
    else:
        # default to the latest available year
        last_idx = len(years_opts) - 1
        bc1, bc2 = st.columns([1, 1])
        with bc1:
            pick_year = st.selectbox(
                "Year",
                options=years_opts,
                index=last_idx,
                key="t6_pick_year_both"
            )
        with bc2:
            top_n = st.slider("Top-N letters", 3, 15, 10, step=1, key="t6_topn_both")

        # Helper to slice one sex's data for the selected year
        def year_slice(by_df, yr):
            # ensure 'year' column is int for matching
            tmp = by_df.copy()
            tmp["year"] = tmp["year"].astype(int)
            s = tmp[tmp["year"] == int(yr)][["first_letter", "value"]].copy()
            return s.sort_values("value", ascending=False).head(top_n)

        yf = year_slice(byF, pick_year)
        ym = year_slice(byM, pick_year)

        # Compute common y max for aligned scales (if both empty, y_max=0)
        y_max = max(
            yf["value"].max() if not yf.empty else 0,
            ym["value"].max() if not ym.empty else 0
        )

        bb1, bb2 = st.columns(2)
        with bb1:
            if yf.empty:
                st.info("No female data for the selected year.")
            else:
                fig_bf = px.bar(
                    yf, x="first_letter", y="value",
                    labels={"first_letter": "Letter", "value": color_title},
                    title=f"Female — Top {top_n} letters in {pick_year}"
                )
                if fmt_percent:
                    fig_bf.update_yaxes(tickformat=".0%")
                fig_bf.update_yaxes(range=[0, y_max])
                st.plotly_chart(fig_bf, use_container_width=True)

        with bb2:
            if ym.empty:
                st.info("No male data for the selected year.")
            else:
                fig_bm = px.bar(
                    ym, x="first_letter", y="value",
                    labels={"first_letter": "Letter", "value": color_title},
                    title=f"Male — Top {top_n} letters in {pick_year}"
                )
                if fmt_percent:
                    fig_bm.update_yaxes(tickformat=".0%")
                fig_bm.update_yaxes(range=[0, y_max])
                st.plotly_chart(fig_bm, use_container_width=True)

    
    thinking_prompts(
    "Digging Deeper",
    [
        "Which letters dominate *today* for girls vs boys?",
        "Which letters have risen or fallen across decades?",
        "Are male and female patterns converging or diverging lately?",
        "If you change the year range, which letters are most sensitive?"
    ],
    key="t6_prompts",
    )
# ------------------------
# Tab 7: Gendered Names (female share + unisex + flipped)
# ------------------------

with tab7:
    st.subheader("How \"male\" or \"female\" is a name?")

    # Inputs
    gname = st.text_input("Type a name", value="Taylor", key="t7_name").strip()
    

    # We'll build the UI in an expander, but make sure we compute in the right order.
    exp = st.expander("Advanced year options", expanded=False)
    with exp:
        smooth = st.checkbox("Smooth with 3-year rolling mean", value=True, key="t7_smooth")
        ec1, ec2 = st.columns(2)

        # Year range selector (defaults 1910–2023, clamped to dataset)
        with ec2:
            yr_min_all, yr_max_all = int(nat_df["year"].min()), int(nat_df["year"].max())
            default_start = max(1910, yr_min_all)
            default_end = min(2023, yr_max_all)
            sel_start, sel_end = st.slider(
                "Year range",
                min_value=yr_min_all,
                max_value=yr_max_all,
                value=(default_start, default_end),
                key="t7_range"
            )

        # Build data for the chosen range and name
        base = nat_df[nat_df["year"].between(sel_start, sel_end)].copy()
        focus = (
            base[base["name"].str.lower() == gname.lower()]
            .groupby(["year", "sex"], as_index=False)["count"].sum()
        )

        # If no data, stop early and show a hint
        if focus.empty:
            st.info("Name not found in the selected years.")
            st.stop()


        # Build female-share series (robust to missing F/M in the range)
        pv = (
            focus.set_index(["year", "sex"])["count"]
                .unstack(fill_value=0)                   # pivot to columns F/M; missing -> 0
                .reindex(columns=["F", "M"], fill_value=0)  # ensure both columns exist
                .rename(columns={"F": "F_count", "M": "M_count"})
        )

        pv.index = pv.index.astype(int)   # keep years as ints for widgets
        pv["total"] = pv[["F_count", "M_count"]].sum(axis=1)

        # If total is zero across the whole range, bail early
        if (pv["total"] == 0).all():
            st.info("No occurrences for this name in the selected years.")
            st.stop()

        pv["female_share"] = np.where(pv["total"] > 0, pv["F_count"] / pv["total"], np.nan)
        # KPI year selector (now that pv exists)
        with ec1:
            years_available = pv.index.tolist()
            if not years_available:
                st.info("No data points in the selected years.")
                st.stop()

            kpi_year = st.selectbox(
                "Pick a year for the single-year metric",
                options=sorted(years_available),
                index=len(years_available) - 1,
                key="t7_kpi_year"
            )
            kpi_share = pv.loc[kpi_year, "female_share"] if kpi_year in pv.index else np.nan

    # ---- KPIs & charts (outside the expander for a cleaner look) ----

    # Overall share across selected years (weighted)
    total_F = pv["F_count"].sum()
    total_all = pv["total"].sum()
    overall_share = (total_F / total_all) if total_all > 0 else np.nan

    # Compact gauges
    import plotly.graph_objects as go
    def gauge_share(value, title):
        no_data = not pd.notna(value)
        val = 0.0 if no_data else float(value)
        bar_color = "#d1d5db" if no_data else "#8b5cf6"  # gray if no data
        subtitle = " (no data)" if no_data else ""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"valueformat": ".0%"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": bar_color},
                "steps": [
                    {"range": [0, 0.33], "color": "#60a5fa"},
                    {"range": [0.33, 0.66], "color": "#a78bfa"},
                    {"range": [0.66, 1.0], "color": "#f472b6"}
                ]
            },
            title={"text": f"{title}{subtitle}"}
        ))
        fig.update_layout(height=180, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(gauge_share(kpi_share, f"Female share in {kpi_year}"), use_container_width=True)
    with g2:
        st.plotly_chart(gauge_share(overall_share, f"Overall female share ({sel_start}–{sel_end})"), use_container_width=True)

    # Smoothing toggle for the line
    if smooth:
        pv["female_share_smooth"] = pv["female_share"].rolling(3, center=True, min_periods=1).mean()
        ycol = "female_share_smooth"
    else:
        ycol = "female_share"

    fig = px.line(
        pv.reset_index(),
        x="year",
        y=ycol,
        title=f'Female share of “{gname.title()}” over time',
        labels={"year": "Year", ycol: "Female share (0–1)"},
    )
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

    st.caption("0 = entirely male, 1 = entirely female; ~0.5 ≈ unisex.")

    thinking_prompts(
    "Digging Deeper",
    [
        "Is the name trending more female or more male over time?",
        "Are shifts gradual or sudden (e.g., pop culture shocks)?",
        "Does the overall share match the last-year share, or are they different?",
        "In your selected range, is the name more unisex than you expected?"
    ],
    key="t7_prompts",
    )



