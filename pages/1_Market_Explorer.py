# =============================================================================
# Imports
# =============================================================================

from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

import math

from market_explorer.auth import require_auth
from market_explorer.discovery import list_datasets_v2
from market_explorer.data_io import load_dataset, to_csv_bytes

from market_explorer.analytics import (
    apply_filters,
    compute_kpis,
    top_by_country,
    top_companies,
    compute_insights,
)

from market_explorer.tiering import add_tier, filter_by_tier

from market_explorer.labels import titleize_slug

from market_explorer.market_explorer_helpers import (
    company_label,
    compute_bp_simple,
    fmt_money,
    get_company_context_row,
)

from market_explorer.notes import (
    load_notes,
    save_notes,
    company_key,
    upsert_note,
)

# =============================================================================
# Authentification sécurité
# =============================================================================

require_auth()
profile = st.session_state.get("profile")

# =============================================================================
# Trajectoire
# =============================================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "Data_Clean2"
datasets_all = list_datasets_v2(DATA_DIR)

if not datasets_all:
    st.error(f"Aucun CSV exploitable trouvé dans {DATA_DIR}")
    st.stop()


# =============================================================================
# Neat
# =============================================================================

C_FONCE = "#41072A"
C_ROSE = "#FF85C8"
C_WHITE = "#FFFFFF"
C_CLAIR = "#F2C5DA"

TIER_ORDER = ["All", "Tier 1", "Tier 2", "Tier 3"]

TIER_UI = {
    "All": "All Markets",
    "Tier 1": "Large Market",
    "Tier 2": "Mid-Market",
    "Tier 3": "Low-Market",
}
TIER_UI_INV = {v: k for k, v in TIER_UI.items()}

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {C_WHITE}; }}
    h1, h2, h3, h4 {{ color: {C_FONCE}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Market Explorer")

tab_explorer, tab_overview = st.tabs(["Market Explorer", "Market Overview"])

# =============================================================================
# Critères par défaut 
# =============================================================================

st.session_state.setdefault("vertical", None)
st.session_state.setdefault("subverticals", None)
st.session_state.setdefault("tier_filter", "All")
st.session_state.setdefault("top_n", 10)


# =============================================================================
# Première Page -- Market Explorer
# =============================================================================

with tab_explorer:
    
    # -----------------------
    # Sidebar
    # -----------------------
    with st.sidebar:
        
        st.header("Navigation")
        
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("pages/0_Home.py")
            
        st.button("🔎 Market Explorer", use_container_width=True, disabled=True)
        
        if st.button("🏨 BP Hotels", use_container_width=True):
            st.switch_page("pages/3_Account_Business_Plan_Hotels.py")
            
        st.divider()

        
        st.header("Scope")
        verticals = sorted({d.vertical for d in datasets_all})

        if not verticals:
            st.warning("Aucune verticale trouvée dans Data_Clean_V2.")
            st.stop()

        verticals_ui = ["All"] + verticals
        default_vertical = st.session_state.get("vertical") or "All"
        vertical_index = verticals_ui.index(default_vertical) if default_vertical in verticals_ui else 0

        vertical = st.selectbox(
            "Vertical",
            verticals_ui,
            key="vertical",
            index=vertical_index,
            format_func=lambda v: "All Verticals" if v == "All" else titleize_slug(v),
        )

        if vertical == "All":
            ds_vertical = datasets_all
        else:
            ds_vertical = [d for d in datasets_all if d.vertical == vertical]

        subverticals = sorted({d.subvertical for d in ds_vertical})
        if not subverticals:
            st.warning("Aucune sous-verticale trouvée pour cette verticale.")
            st.stop()

        all_subverticals_selected = st.checkbox("All sub-verticals", value=True)
        if all_subverticals_selected:
            selected_subverticals = subverticals
        else:
            stored_subverticals = st.session_state.get("subverticals") or []
            default_subverticals = (
                [s for s in stored_subverticals if s in subverticals]
                or subverticals[: min(len(subverticals), 5)]
            )
            selected_subverticals = st.multiselect(
                "Sous-verticales",
                subverticals,
                default=default_subverticals,
                key="subverticals",
                format_func=titleize_slug,
            )

        if not selected_subverticals:
            st.warning("Veuillez sélectionner au moins une sous-verticale.")
            st.stop()

        if all_subverticals_selected:
            ds_scope = ds_vertical
        else:
            ds_scope = [d for d in ds_vertical if d.subvertical in selected_subverticals]

        if all_subverticals_selected:
            subvertical_label = "All Sub-verticals"
        elif len(selected_subverticals) <= 3:
            subvertical_label = ", ".join(titleize_slug(s) for s in selected_subverticals)
        else:
            subvertical_label = f"{len(selected_subverticals)} Sub-verticals"

        if not ds_scope:
            st.warning("Aucun dataset trouvé pour ce couple verticale/sous-verticale.")
            st.stop()

        countries_scope = sorted({d.country for d in ds_scope})
        if not countries_scope:
            st.warning("Aucun pays trouvé pour ce couple verticale/sous-verticale.")
            st.stop()

        all_countries_selected = st.checkbox("All countries", value=True)
        
        if all_countries_selected:
            selected_countries = countries_scope
        else:
            selected_countries = st.multiselect(
                "Countries",
                countries_scope,
                default=countries_scope[: min(len(countries_scope), 5)],
                format_func=titleize_slug,
            )
    # -----------------------
    # Load dataset(s)
    # -----------------------
    
    if not selected_countries:
        st.warning("Veuillez sélectionner au moins un pays.")
        st.stop()

    match = [d for d in ds_scope if d.country in selected_countries]

    if not match:
        st.warning("Dataset introuvable pour ce scope verticale/sous-verticale/pays.")
        st.stop()

    dfs = []
    for d in match:
        tmp = load_dataset(d.path)
        if tmp is not None and not tmp.empty:
            dfs.append(tmp)

    if not dfs:
        st.warning("Dataset introuvable ou vide pour ce scope verticale/sous-verticale/pays.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)

    # Display sources
    
    if len(match) == 1:
        dataset_info = match[0]
        st.caption(
            f"Source: {titleize_slug(dataset_info.vertical)} / {titleize_slug(dataset_info.subvertical)} — "
            f"{titleize_slug(dataset_info.country)} — {dataset_info.path.name}"
        )
    else:
        files = ", ".join([f"{titleize_slug(d.country)}: {d.path.name}" for d in match])
        st.caption(
            f"Source: {titleize_slug(vertical) if vertical != 'All' else 'All Verticals'} / "
            f"{subvertical_label} — "
            f"{'All Countries' if len(selected_countries) == len(countries_scope) else 'Selected Countries'} — {files}"
        )
    
    # -----------------------
    # Sidebar: Country summary (independent from tiering)
    # -----------------------
    with st.sidebar:
        st.subheader("Pays")
        st.caption(
            f"{len(selected_countries)} pays sélectionné(s) sur {len(countries_scope)} disponibles."
        )

        st.divider()

        # -----------------------
        # Tiering
        # -----------------------
        st.header("Market Tiering")

        t1 = st.number_input("Large Market threshold (M$)", min_value=0.0, value=500.0, step=50.0)
        t2 = st.number_input("Mid-Market threshold (M$)", min_value=0.0, value=100.0, step=25.0)

        if t1 < t2:
            st.warning("Large Market threshold should be ≥ Mid-Market threshold. Adjusting automatically.")
            t1, t2 = t2, t1

        tier_ui_options = [TIER_UI[t] for t in TIER_ORDER]
        default_internal = st.session_state.get("tier_filter", "All")
        default_ui = TIER_UI.get(default_internal, TIER_UI["All"])
        default_index = tier_ui_options.index(default_ui) if default_ui in tier_ui_options else 0

        tier_ui = st.selectbox("Market Tier", tier_ui_options, index=default_index)
        tier_filter = TIER_UI_INV[tier_ui]
        st.session_state["tier_filter"] = tier_filter

        st.divider()

    # -----------------------
    # Prepare tiered df + apply tier filter
    # -----------------------
    df_t = add_tier(df, t1=t1, t2=t2)
    df_t = filter_by_tier(df_t, tier_filter)
    
    # -----------------------
    # Revenue bounds based on selected tier (TRUE bounds)
    # -----------------------
    df_t_rev = df_t.copy()
    df_t_rev["Revenue_M"] = pd.to_numeric(df_t_rev.get("Revenue_M"), errors="coerce")
    valid_rev = df_t_rev["Revenue_M"].dropna()
    
    if valid_rev.empty:
        rev_min_raw, rev_max_raw = 0.0, 0.0
    else:
        rev_min_raw = float(valid_rev.min())
        rev_max_raw = float(valid_rev.max())
    
    # Round nicely
    rev_min = float(math.floor(rev_min_raw / 10) * 10) if rev_min_raw > 0 else 0.0
    rev_max = float(math.ceil(rev_max_raw / 10) * 10) if rev_max_raw > 0 else 0.0
    
    # Sidebar debug caption
    st.sidebar.caption(f"Max revenue in selected tier: {rev_max_raw:,.1f} M$")
    
    # -----------------------
    # Sidebar: Filters
    # -----------------------
    company_types = sorted(
        [c for c in df_t.get("Company Type", pd.Series(dtype=object)).dropna().unique().tolist() if str(c).strip()]
    )
    sectors = sorted(
        [c for c in df_t.get("Sector", pd.Series(dtype=object)).dropna().unique().tolist() if str(c).strip()]
    )
    
    with st.sidebar:
        st.header("Filters")
    
        # Info caption based on tier
        if tier_filter in ("Tier 1", "Tier 2", "Tier 3"):
            st.caption(f"Revenue range constrained by {TIER_UI[tier_filter]}.")
        else:
            st.caption("Revenue range constrained by current scope (All Markets).")
    
        # Slider bounds ARE the tier bounds
        if rev_max <= rev_min:
            revenue_range = (0.0, 0.0)
        else:
            step = 10.0 if (rev_max - rev_min) >= 10 else 1.0
    
            revenue_range = st.slider(
                "Revenue range (M$)",
                min_value=float(rev_min),
                max_value=float(rev_max),
                value=(float(rev_min), float(rev_max)),
                step=float(step),
            )
            
        company_type = st.selectbox("Company Type", ["All"] + company_types, index=0)
        sector = st.selectbox("Sector", ["All"] + sectors, index=0)
    
        top_n = st.slider(
            "Top N (charts)",
            min_value=5,
            max_value=50,
            value=st.session_state.get("top_n", 10),
            step=5,
        )
        st.session_state["top_n"] = top_n

    # -----------------------
    # Apply filters
    # -----------------------
    # IMPORTANT: UI uses "All" as a sentinel meaning "no filter".
    # analytics.apply_filters expects None (or an empty iterable) to disable a filter.
    
    if len(selected_countries) == len(countries_scope):
        country_f = None
    else:
        country_f = list(selected_countries)
    company_type_f = None if company_type == "All" else [company_type]
    sector_f = None if sector == "All" else [sector]

    df_f = apply_filters(
        df_t,
        revenue_min_m=float(revenue_range[0]),
        revenue_max_m=float(revenue_range[1]),
        country=country_f,
        company_type=company_type_f,
        sector=sector_f,
    )

    # -----------------------
    # Scope banner
    # -----------------------
    
    scope_vertical = "All Verticals" if vertical == "All" else titleize_slug(vertical)
    scope_subvertical = subvertical_label
    if len(selected_countries) == len(countries_scope):
        scope_countries = "All Countries"
    elif len(selected_countries) <= 3:
        scope_countries = ", ".join(titleize_slug(c) for c in selected_countries)
    else:
        scope_countries = f"{len(selected_countries)} Countries"
    scope_tier = TIER_UI.get(tier_filter, tier_filter)

    st.markdown(
        f"""
        <div style="
            padding: 14px 18px;
            border-left: 5px solid {C_FONCE};
            margin: 8px 0 14px 0;
            background-color: #FAFAFA;
            border-radius: 4px;
        ">
            <div style="font-size: 18px; font-weight: 600; color: {C_FONCE};">
                {scope_vertical} / {scope_subvertical} — {scope_countries}
            </div>
            <div style="margin-top: 4px; font-size: 14px;">
                <strong>{scope_tier}</strong> · {len(df_f)} target companies
            </div>
            <div style="margin-top: 2px; font-size: 13px; color: #666;">
                Revenue scope: {revenue_range[0]:,.0f}–{revenue_range[1]:,.0f} M$
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -----------------------
    # KPIs
    # -----------------------
    kpis = compute_kpis(df_f)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{int(kpis['companies'])}")
    c2.metric("Total Revenue (M$)", f"{kpis['total_rev_m']:,.0f}")
    c3.metric("Median Revenue (M$)", f"{kpis['median_rev_m']:,.1f}")
    c4.metric("Countries", f"{int(kpis['countries'])}")

    st.divider()

    # Insights (support both dict-based and list-based contracts)
    insights = compute_insights(df_f)
    single_country_scope = len(selected_countries) == 1
    if isinstance(insights, dict) and insights:
        if single_country_scope:
            st.info(
                "📌 **Market insight** — "
                "Top 5 companies represent "
                f"{insights.get('top5_share_pct', 0)}% of total market revenue."
            )
        else:
            st.info(
                "📌 **Market insight** — "
                f"{insights.get('top_country', 'N/A')} represents "
                f"{insights.get('top_country_share_pct', 0)}% of total market revenue."
            )
    elif isinstance(insights, (list, tuple)) and len(insights) > 0:
        st.info("📌 **Market insight** — " + " ".join(str(x) for x in insights if str(x).strip()))

    # -----------------------
    # Charts
    # -----------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        if single_country_scope:
            st.subheader("Top Countries (by Revenue)")
            st.info("Geographic split is not relevant for a single-country scope.")
        else:
            st.subheader("Top Countries (by Revenue)")
            by_country = top_by_country(df_f)
            if "Revenue_M" in by_country.columns:
                by_country = by_country.head(10)

            if len(by_country) == 0:
                st.info("No country data available for the current filters.")
            elif len(by_country) == 1:
                st.info("Only one country available. Not enough diversity to plot Top countries.")
            else:
                bc = by_country[["Country", "Revenue_M"]].copy().sort_values("Revenue_M", ascending=True)
                fig = px.bar(
                    bc,
                    x="Revenue_M",
                    y="Country",
                    orientation="h",
                    labels={"Revenue_M": "Revenue (M$)", "Country": ""},
                    color_discrete_sequence=[C_ROSE],
                )
                fig.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font_color=C_FONCE,
                    margin=dict(l=10, r=10, t=10, b=10),
                    height=420,
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Top {top_n} Companies (Revenue)")
        top_df = top_companies(df_f, n=top_n)

        if len(top_df) == 0:
            st.info("No companies available to plot with the current filters.")
        else:
            tc = top_df[["Name", "Revenue_M"]].copy()
            tc["Name"] = tc["Name"].astype(str)
            tc = tc.sort_values("Revenue_M", ascending=True)
            fig = px.bar(
                tc,
                x="Revenue_M",
                y="Name",
                orientation="h",
                labels={"Revenue_M": "Revenue (M$)", "Name": ""},
                color_discrete_sequence=[C_FONCE],
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color=C_FONCE,
                margin=dict(l=10, r=10, t=10, b=10),
                height=420,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Market Revenue Concentration")

        df_conc = df_f.copy()
        df_conc["Revenue_M"] = pd.to_numeric(df_conc.get("Revenue_M"), errors="coerce")
        df_conc = df_conc.dropna(subset=["Revenue_M"])

        if df_conc.empty or float(df_conc["Revenue_M"].sum()) <= 0:
            st.info("Not enough revenue data to compute market concentration.")
        else:
            TOP_LEADERS_N = 20
            total_rev = float(df_conc["Revenue_M"].sum())
            top_rev = float(df_conc.nlargest(TOP_LEADERS_N, "Revenue_M")["Revenue_M"].sum())
            rest_rev = max(0.0, total_rev - top_rev)

            conc_df = pd.DataFrame(
                {
                    "Segment": [f"Top {TOP_LEADERS_N} Leaders", "Reste du Marché"],
                    "Revenue_M": [top_rev, rest_rev],
                }
            )

            fig = px.pie(conc_df, names="Segment", values="Revenue_M", hole=0.65)
            fig.update_traces(
                textinfo="percent",
                textposition="inside",
                sort=False,
                marker=dict(colors=[C_FONCE, C_ROSE]),
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_color=C_FONCE,
                margin=dict(l=10, r=10, t=10, b=10),
                height=420,
                showlegend=True,
            )
            st.plotly_chart(fig, use_container_width=True)

# -----------------------
# General BP 
# -----------------------

    is_airline_vertical = (
        len(selected_subverticals) == 1 and str(selected_subverticals[0]).lower() == "airline"
    )
    bp_vertical_label = "Airline" if is_airline_vertical else "Hotel"
    st.subheader(f"BP Général — Premium & Commission ({bp_vertical_label})")

    # Default base: your filtered revenue KPI (already in M$)
    
    base_hotel_rev_m = float(kpis.get("total_rev_m", 0.0))

    bp_col1, bp_col2, bp_col3 = st.columns(3)

    with bp_col1:
        
        st.subheader("Market")
        base_hotel_rev_m = st.number_input(
            f"{bp_vertical_label} revenue (Year 1, M$)",
            min_value=0.0,
            value=base_hotel_rev_m,
            step=1.0,    
        )
        
        ticket_share_pct = 100.0
        if is_airline_vertical:
            ticket_share_pct = st.number_input(
                "Ticket sales share (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
                step=1.0,
            )
            
        market_growth_pct = st.number_input(
            "Market growth / year (%)",
            min_value=0.0,
            max_value=200.0,
            value=8.0,
            step=1.0,
        )

    with bp_col2:
        
        st.subheader("Distribution")
        direct_rate_pct = st.number_input(
            "% Direct",
            min_value=0.0,
            max_value=100.0,
            value=30.0,
            step=1.0,
        )
        
        take_rate_pct = st.number_input(
            "Take Rate (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
        )

    with bp_col3:
        
        st.subheader("Economics")
        price_rate_pct = st.number_input(
            "Price (% of booking)",
            min_value=0.0,
            max_value=100.0,
            value=4.0,
            step=0.5,
        )
        neat_comm_pct = st.number_input(
            "Neat commission (%)",
            min_value=0.0,
            max_value=100.0,
            value=20.0,
            step=1.0,
        )

    # Convert % → decimals with caps
    
    market_growth = min(max(market_growth_pct / 100.0, 0.0), 2.0)
    direct_rate = min(max(direct_rate_pct / 100.0, 0.0), 1.0)
    ticket_share = min(max(ticket_share_pct / 100.0, 0.0), 1.0)
    take_rate = min(max(take_rate_pct / 100.0, 0.0), 1.0)
    price_rate = min(max(price_rate_pct / 100.0, 0.0), 1.0)
    neat_commission = min(max(neat_comm_pct / 100.0, 0.0), 1.0)

    if is_airline_vertical:
        assurable_rev_m = base_hotel_rev_m * ticket_share
        st.caption(f"Assurable revenue from ticket sales (Y1): {assurable_rev_m:,.1f} M$".replace(",", " "))
    else:
        assurable_rev_m = base_hotel_rev_m
        
    revenue_label = "Assurable revenue (M$)" if is_airline_vertical else f"{bp_vertical_label} revenue (M$)"


    df_bp = compute_bp_simple(
        assurable_rev_m,
        market_growth,
        direct_rate,
        take_rate,
        price_rate,
        neat_commission,
        years = 5,
        revenue_label=revenue_label
    )

    # KPI cards
    
    year5 = df_bp.iloc[-1]
    k1, k2, k3 = st.columns(3)
    k1.metric("Premium (Year 5)", fmt_money(float(year5["Premium (M$)"])))
    k2.metric("Neat revenue (Year 5)", fmt_money(float(year5["Neat revenue (M$)"])))
    year5_market_rev = base_hotel_rev_m * ((1 + market_growth) ** 4)
    k3.metric(f"{bp_vertical_label} revenue (Year 5)", fmt_money(float(year5_market_rev)))

    # Chart
    
    fig_bp = px.line(
        df_bp,
        x="Year",
        y=["Premium (M$)", "Neat revenue (M$)"],
        markers=True,
        labels={"value": "Amount (M$)", "variable": ""},
        color_discrete_sequence=[C_FONCE, C_ROSE],
    )

    fig_bp.update_traces(
        mode="lines+markers",
        line=dict(width=5),
        marker=dict(size=9),
    )

    fig_bp.update_layout(
        template="plotly_white",
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color=C_FONCE,
        margin=dict(l=10, r=10, t=10, b=10),
        height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig_bp, use_container_width=True)

    st.divider()


    # -----------------------
    # Qualified Target List
    # -----------------------
    
    st.subheader("Qualified Target List")

    # Notes for current profile
    
    notes = load_notes(profile)

    cols = [
        "Name",
        "Country",
        "Revenue_M",
        "Market Tier",
        "Sector",
        "LinkedIn URL",
        "Company Type",
        "Main Broker",
        "Main Insurer",
    ]
    for c in cols:
        if c not in df_f.columns:
            df_f[c] = None

    target = df_f[cols].sort_values("Revenue_M", ascending=False).copy()

    # Add tag column from notes
    
    def _tag_for_row(r):
        k = company_key(str(r["Name"]), str(r.get("Country", "")))
        return notes.get(k, {}).get("tag", "")

    target.insert(0, "My Tag", target.apply(_tag_for_row, axis=1))

    st.caption("Tip: click a row to open/edit notes below 👇")

    event = st.dataframe(
        target,
        use_container_width=True,
        height=520,
        hide_index=True,
        selection_mode="single-row",
        on_select="rerun",
    )

    subvertical_export = (
        "all_subverticals" if all_subverticals_selected else "-".join(selected_subverticals)
    )
    export_tag = f"{vertical}_{subvertical_export}"
    countries_tag = "all_countries" if len(selected_countries) == len(countries_scope) else "selected_countries"
    st.download_button(
        label="Download Target List (CSV)",
        data=to_csv_bytes(target),
        file_name=f"targets_{export_tag}_{countries_tag}.csv",
        mime="text/csv",
    )

    # -----------------------
    # Create Business Plan
    # -----------------------
    
    st.markdown("### 🚀 Créer un BP")

    selected_row_for_bp = None
    selected_label_for_bp = None

    if event is not None and getattr(event, "selection", None):
        rows = event.selection.get("rows", [])
        if rows:
            i = rows[0]
            selected_row_for_bp = target.iloc[i]
            selected_label_for_bp = company_label(selected_row_for_bp)
            st.session_state["bp_selected_company_option"] = selected_label_for_bp

    company_options = []
    option_to_row = {}
    for _, row in target.iterrows():
        label = company_label(row)
        company_options.append(label)
        option_to_row[label] = row

    if not company_options:
        st.info("Aucune entreprise disponible pour créer un BP avec les filtres actuels.")
    else:
        selected_label = st.selectbox(
            "Choisir une entreprise",
            company_options,
            key="bp_selected_company_option",
        )
        selected_row_for_bp = option_to_row.get(selected_label)

    can_create_bp = selected_row_for_bp is not None

    if st.button("🚀 Créer un BP", use_container_width=True, disabled=not can_create_bp):
        if not can_create_bp:
            st.warning("Veuillez sélectionner une entreprise avant de créer un BP")
        else:
            context = get_company_context_row(selected_row_for_bp)
            dataset_label = (
                match[0].path.name
                if len(match) == 1
                else ", ".join(sorted({d.path.name for d in match}))
            )
            st.session_state["bp_context"] = {
                **context,
                "dataset": dataset_label,
                "tiering": tier_filter,
                "zone": scope_countries,
                "market": vertical,
                "vertical": selected_subverticals,
                "subvertical": selected_subverticals,
                "countries_scope": scope_countries,
                "countries": list(selected_countries),
                "source": "market_explorer",
            }
            try:
                st.switch_page("pages/3_Account_Business_Plan_Hotels.py")
            except Exception:
                st.session_state["current_page"] = "bp_hotels"
                st.rerun()


    # -----------------------
    # Notes editor
    # -----------------------

    st.markdown("### 📝 Notes on a company")

    selected_name = None
    selected_country = ""
    linkedin_url = ""

    # 1) If a row is selected from the table
    
    if event is not None and getattr(event, "selection", None):
        rows = event.selection.get("rows", [])
        if rows:
            i = rows[0]
            selected_name = str(target.iloc[i]["Name"])
            selected_country = str(target.iloc[i].get("Country", ""))
            row = target.iloc[[i]]  # dataframe with 1 row

    # 2) Fallback: selectbox
    
    names = target["Name"].astype(str).tolist()

    if not names:
        st.info("No companies to annotate with current filters.")
        st.stop()

    if selected_name is None:
        selected_name = st.selectbox("Company", names)
        row = target[target["Name"].astype(str) == str(selected_name)].head(1)
        selected_country = str(row["Country"].iloc[0]) if not row.empty else ""
    else:
        st.markdown(f"**Selected:** {selected_name} ({selected_country})")

    # 3) Extract LinkedIn URL 
    
    try:
        linkedin_url = str(row["LinkedIn URL"].iloc[0] or "").strip()
    except Exception:
        linkedin_url = ""

    # 4) Notes logic
    
    key = company_key(selected_name, selected_country)
    existing = notes.get(key, {})

    tag_options = ["", "Hot", "Maybe", "No fit"]
    current_tag = existing.get("tag", "")
    tag_index = tag_options.index(current_tag) if current_tag in tag_options else 0

    tag = st.selectbox("Tag", tag_options, index=tag_index)
    note_text = st.text_area(
        "Why?",
        value=existing.get("note", ""),
        height=120,
        max_chars=300,
    )

    # 5) Actions
    
    cA, cB = st.columns([1, 1])

    with cA:
        if st.button("💾 Save note", use_container_width=True):
            display_vertical = "All Verticals" if vertical == "All" else titleize_slug(vertical)
            display_subvertical = subvertical_label
            notes = upsert_note(
                notes,
                key,
                tag,
                note_text,
                display_name=selected_name,
                country=selected_country,
                linkedin_url=linkedin_url,
                subvertical=display_subvertical,
                countries_scope=scope_countries,
                vertical=display_vertical,
            )
            save_notes(profile, notes)
            st.success("Saved ✅")
            st.rerun()

    with cB:
        if st.button("🗑️ Delete this note", use_container_width=True):
            if key in notes:
                notes.pop(key, None)
                save_notes(profile, notes)
                st.success("Deleted ✅")
                st.rerun()
            else:
                st.info("No note to delete.")

# =============================================================================
# 2eme Page -- Market Overview
# =============================================================================
    
with tab_overview:
    st.subheader("Market Share Overview")

    all_countries_overview = sorted({d.country for d in datasets_all})
    all_verticals_overview = sorted({d.vertical for d in datasets_all})

    c1, c2 = st.columns([1, 1])
    with c1:
        vertical_ms = st.selectbox(
            "Vertical",
            all_verticals_overview,
            index=0,
            format_func=titleize_slug,
        )
    
    with c2:
        include_all_countries = st.checkbox("All countries (overview)", value=True)
        if include_all_countries:
            countries_ms = all_countries_overview
        else:
            countries_ms = st.multiselect(
                "Countries",
                all_countries_overview,
                default=all_countries_overview[: min(len(all_countries_overview), 5)],
                format_func=titleize_slug,
            )
    # 1) Vertical shares (macro)

    ds_country = [d for d in datasets_all if d.country in countries_ms]
    if not ds_country:
        st.warning("No datasets found for the selected countries.")
        st.stop()

    rows_vertical = []
    for d in ds_country:
        try:
           tmp = load_dataset(d.path)
        except Exception as e:
            st.warning(f"Could not load {d.path.name}: {e}")
            continue
        if tmp.empty:
            continue
        rows_vertical.append({"Vertical": d.vertical, "Revenue_M": float(tmp["Revenue_M"].sum())})

    if not rows_vertical:
        st.warning("No usable revenue data found for the selected countries.")
        st.stop()

    df_vertical = pd.DataFrame(rows_vertical).groupby("Vertical", as_index=False)["Revenue_M"].sum()
    total_scope = float(df_vertical["Revenue_M"].sum())
    if total_scope <= 0:
        st.warning("Total revenue is 0 for the selected countries (nothing to display).")
        st.stop()

    df_vertical["Share_%"] = (df_vertical["Revenue_M"] / total_scope * 100).round(1)
    df_vertical["Vertical_label"] = df_vertical["Vertical"].map(titleize_slug)
    df_plot_vertical = df_vertical.sort_values("Revenue_M", ascending=True).copy()
    df_plot_vertical["Share_label"] = df_plot_vertical["Share_%"].astype(str) + "%"

    scope_label = (
        "All Countries"
        if len(countries_ms) == len(all_countries_overview)
        else ", ".join(titleize_slug(c) for c in countries_ms[:3])
    )
    if len(countries_ms) > 3 and len(countries_ms) != len(all_countries_overview):
        scope_label = f"{scope_label} (+{len(countries_ms) - 3})"

    st.caption(
        f"Countries: {scope_label} — Total revenue analyzed: {total_scope:,.0f} M$ "
        "(sum of all datasets in Data_Clean_V2 for the selected countries)."
    )

    fig = px.bar(
        df_plot_vertical,
        x="Revenue_M",
        y="Vertical_label",
        orientation="h",
        labels={"Revenue_M": "Revenue (M$)", "Vertical_label": ""},
        text="Share_label",
        color_discrete_sequence=[C_FONCE],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color=C_FONCE,
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # 2) Drill-down: sub-vertical shares within selected vertical

    st.subheader(f"{titleize_slug(vertical_ms)} — Breakdown by Sub-vertical")

    ds_vertical = [d for d in ds_country if d.vertical == vertical_ms]
    if not ds_vertical:
        st.info(f"No datasets found for vertical={titleize_slug(vertical_ms)} in this scope.")
        st.stop()

    rows_subvertical = []
    for d in ds_vertical:
        try:
            tmp = load_dataset(d.path)
        except Exception as e:
            st.warning(f"Could not load {d.path.name}: {e}")
            continue
        if tmp.empty:
            continue
        rows_subvertical.append({"Subvertical": d.subvertical, "Revenue_M": float(tmp["Revenue_M"].sum())})

    if not rows_subvertical:
        st.info("No usable revenue data for this vertical / country scope.")
        st.stop()

    df_sub = pd.DataFrame(rows_subvertical).groupby("Subvertical", as_index=False)["Revenue_M"].sum()
    total_vertical = float(df_sub["Revenue_M"].sum())
    if total_vertical <= 0:
        st.info("Total revenue is 0 for this vertical (nothing to display).")
        st.stop()

    df_sub["Share_%"] = (df_sub["Revenue_M"] / total_vertical * 100).round(1)
    df_sub["Subvertical_label"] = df_sub["Subvertical"].map(titleize_slug)
    df_sub = df_sub.sort_values("Revenue_M", ascending=True).copy()
    df_sub["Share_label"] = df_sub["Share_%"].astype(str) + "%"

    st.caption(f"{titleize_slug(vertical_ms)} total: {total_vertical:,.0f} M$")

    fig2 = px.bar(
        df_sub,
        x="Revenue_M",
        y="Subvertical_label",
        orientation="h",
        labels={"Revenue_M": "Revenue (M$)", "Subvertical_label": ""},
        text="Share_label",
        color_discrete_sequence=[C_ROSE],
    )
    fig2.update_traces(textposition="outside", cliponaxis=False)
    fig2.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color=C_FONCE,
        margin=dict(l=10, r=10, t=10, b=10),
        height=420,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Details")
    df_table = df_sub.sort_values("Revenue_M", ascending=False)[
        ["Subvertical_label", "Revenue_M", "Share_%"]
    ].copy()
    df_table.columns = ["Sub-vertical", "Revenue (M$)", "Share (%)"]
    st.dataframe(df_table, use_container_width=True, height=260)
