from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

import math

from market_explorer.auth import require_auth
from market_explorer.discovery import DatasetCatalog, list_datasets
from market_explorer.data_io import load_dataset, to_csv_bytes

from market_explorer.analytics import (
    apply_filters,
    compute_kpis,
    top_by_country,
    top_companies,
    compute_insights,
)

from market_explorer.tiering import add_tier, filter_by_tier

from market_explorer.labels import (
    titleize_slug,
    zone_label,
    market_label,
    zone_label_ui,
    zones_in_scope_from_ui,
)

from market_explorer.notes import (
    load_notes,
    save_notes,
    company_key,
    upsert_note,
)

# =============================================================================
# Auth guard
# =============================================================================

require_auth()
profile = st.session_state.get("profile")

# =============================================================================
# Paths / catalog
# =============================================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "Data_Clean"
catalog = DatasetCatalog.from_dir(DATA_DIR)

datasets_all = list_datasets(DATA_DIR)
if not datasets_all:
    st.error(f"Aucun CSV exploitable trouvé dans {DATA_DIR}")
    st.stop()


# =============================================================================
# Theme (Neat)
# =============================================================================

C_FONCE = "#41072A"
C_ROSE = "#FF85C8"
C_WHITE = "#FFFFFF"

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

st.title("Market Explorer (Internal)")

tab_explorer, tab_overview = st.tabs(["Market Explorer", "Market Overview"])

ZONE_LABELS = {
    "france": "France",
    "eu": "Europe",
    "eu_fr": "Europe + France",
}

def format_zone_option(value) -> str:
    normalized = str(value).strip().lower()
    return ZONE_LABELS.get(normalized, zone_label_ui(value))

def get_company_context_row(row: pd.Series) -> dict:
    company_id = row.get("Company ID")
    if pd.isna(company_id) or str(company_id).strip() == "":
        company_id = company_key(str(row["Name"]), str(row.get("Country", "")))
    return {
        "company_id": str(company_id),
        "company_name": str(row["Name"]),
        "country": str(row.get("Country", "")),
    }

def company_label(row: pd.Series) -> str:
    label_country = str(row.get("Country", "")).strip()
    return f"{row['Name']} ({label_country})" if label_country else str(row["Name"])

# =============================================================================
# Session state defaults
# =============================================================================

st.session_state.setdefault("zone", "france")
st.session_state.setdefault("market", "travel")
st.session_state.setdefault("vertical", "hotel")
st.session_state.setdefault("tier_filter", "All")
st.session_state.setdefault("top_n", 10)


# =============================================================================
# TAB 1 — MARKET EXPLORER
# =============================================================================

with tab_explorer:

    # -----------------------
    # Sidebar: Scope + Tiering
    # -----------------------
    with st.sidebar:
        
        st.markdown("### Navigation")
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("pages/0_Home.py")
        st.button("🔎 Market Explorer", use_container_width=True, disabled=True)
        if st.button("🏨 BP Hotels", use_container_width=True):
            st.switch_page("pages/3_Account_Business_Plan_Hotels.py")
        st.divider()

        
        st.header("Scope")

        zone = st.selectbox(
            "Zone",
            ["france", "eu", "eu_fr"],
            key="zone",
            format_func=format_zone_option,
        )

        zones_in_scope = zones_in_scope_from_ui(zone)
        ds_zone = [d for d in datasets_all if d.zone in zones_in_scope]

        if not ds_zone:
            st.warning("Aucun dataset trouvé pour cette zone.")
            st.stop()

        markets = sorted({d.market for d in ds_zone})
        markets_ui = ["All"] + markets
        default_market = st.session_state.get("market", "All")
        market_index = markets_ui.index(default_market) if default_market in markets_ui else 0

        market = st.selectbox(
            "Market",
            markets_ui,
            key="market",
            index=market_index,
            format_func=lambda m: "All Markets" if m == "All" else market_label(m),
        )

        # datasets in zone + market
        if market == "All":
            ds_scope = ds_zone
        else:
            ds_scope = [d for d in ds_zone if d.market == market]

        verticals = sorted({d.vertical for d in ds_scope})
        if not verticals:
            st.warning("Aucune verticale trouvée pour ce couple Zone/Market.")
            st.stop()

        verticals_ui = ["All"] + verticals
        default_vertical = st.session_state.get("vertical", "All")
        vertical_index = verticals_ui.index(default_vertical) if default_vertical in verticals_ui else 0

        vertical = st.selectbox(
            "Vertical",
            verticals_ui,
            key="vertical",
            index=vertical_index,
            format_func=lambda v: "All Verticals" if v == "All" else titleize_slug(v),
        )
    # -----------------------
    # Load dataset(s)
    # -----------------------
    if vertical == "All":
        match = ds_scope
    else:
        match = [d for d in ds_scope if d.vertical == vertical]

    if not match:
        st.warning("Dataset introuvable pour ce couple market/vertical/zone.")
        st.stop()

    dfs = []
    for d in match:
        tmp = load_dataset(d.path, zone=d.zone)
        if tmp is not None and not tmp.empty:
            dfs.append(tmp)

    if not dfs:
        st.warning("Dataset introuvable ou vide pour ce couple market/vertical/zone.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)

    # Display sources
    if len(match) == 1:
        dataset_info = match[0]
        st.caption(
            f"Source: {market_label(dataset_info.market)} / {titleize_slug(dataset_info.vertical)} — "
            f"{zone_label(dataset_info.zone)} — {dataset_info.path.name}"
        )
    else:
        files = ", ".join([f"{zone_label(d.zone)}: {d.path.name}" for d in match])
        st.caption(
            f"Source: {market_label(market) if market != 'All' else 'All Markets'} / "
            f"{titleize_slug(vertical) if vertical != 'All' else 'All Verticals'} — "
            f"{zone_label_ui(zone)} — {files}"
        )
        
    # -----------------------
    # Sidebar: Zone country filter (independent from tiering)
    # -----------------------
    countries_scope = sorted(
        [c for c in df.get("Country", pd.Series(dtype=object)).dropna().unique().tolist() if str(c).strip()]
    )

    with st.sidebar:
        st.subheader("Zone")

        if zone == "france":
            country = st.selectbox("Country", ["All"] + countries_scope, index=0)
        else:
            all_countries_selected = st.checkbox("All countries", value=True)
            if all_countries_selected:
                country = countries_scope
            else:
                country = st.multiselect(
                    "Countries",
                    countries_scope,
                    default=countries_scope[: min(len(countries_scope), 5)],
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
    
    if zone == "france":
        country_f = None if country == "All" else [country]
    else:
        country_f = None if not country else list(country)
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
    scope_market = "All Markets" if market == "All" else market_label(market)
    scope_vertical = "All Verticals" if vertical == "All" else titleize_slug(vertical)
    scope_zone = zone_label_ui(zone)
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
                {scope_market} / {scope_vertical} — {scope_zone}
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
    if isinstance(insights, dict) and insights:
        if zone == "france":
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
        if zone == "france":
            st.subheader("Top Countries (by Revenue)")
            st.info("Geographic split is not relevant for France (single-country scope).")
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

    st.divider()

    # -----------------------
    # Qualified Target List
    # -----------------------
    st.subheader("Qualified Target List")

    # Notes for current profile (for tags in table + editor)
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

    # Add tag column from notes (fast + sales-friendly)
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

    export_tag = f"{market}_{vertical}"
    st.download_button(
        label="Download Target List (CSV)",
        data=to_csv_bytes(target),
        file_name=f"targets_{export_tag}_{zone}.csv",
        mime="text/csv",
    )

    # -----------------------
    # Create Business Plan
    # -----------------------
    st.markdown("### 🚀 Créer un BP (Hôtel)")

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

    if st.button("🚀 Créer un BP (Hôtel)", use_container_width=True, disabled=not can_create_bp):
        if not can_create_bp:
            st.warning("Veuillez sélectionner une entreprise avant de créer un BP.")
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
                "zone": zone,
                "market": market,
                "vertical": vertical,
                "source": "market_explorer",
            }
            try:
                st.switch_page("pages/3_Account_Business_Plan_Hotels.py")
            except Exception:
                st.session_state["current_page"] = "bp_hotels"
                st.rerun()


    # -----------------------
    # Notes editor (click row -> edit)
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

    # 3) Extract LinkedIn URL (NOW row is correct)
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
            display_zone = zone_label_ui(zone)
            display_market = "All Markets" if market == "All" else market_label(market)
            display_vertical = "All Verticals" if vertical == "All" else titleize_slug(vertical)
            notes = upsert_note(
                notes,
                key,
                tag,
                note_text,
                display_name=selected_name,
                country=selected_country,
                linkedin_url=linkedin_url,
                zone=display_zone,
                market=display_market,
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
# TAB 2 — MARKET OVERVIEW
# =============================================================================
with tab_overview:
    st.subheader("Market Share Overview")

    c1, c2 = st.columns([1, 1])
    with c1:
        zone_ms = st.selectbox(
            "Zone (Overview)",
            ["france", "eu", "eu_fr"],
            index=0,
            format_func=format_zone_option,
        )

    zones_ms = zones_in_scope_from_ui(zone_ms)

    markets_focus = ["travel", "goods", "financial_services", "ticketing"]
    with c2:
        market_ms = st.selectbox(
            "Market",
            markets_focus,
            index=0,
            format_func=market_label,
        )

    # 1) Market shares (macro)
    ds_zone = [d for d in datasets_all if d.zone in zones_ms and d.market in markets_focus]
    if not ds_zone:
        st.warning(f"No datasets found for zone={zone_ms}.")
        st.stop()

    rows_market = []
    for d in ds_zone:
        try:
            tmp = load_dataset(d.path, zone=d.zone)
        except Exception as e:
            st.warning(f"Could not load {d.path.name}: {e}")
            continue
        if tmp.empty:
            continue
        rows_market.append({"Market": d.market, "Revenue_M": float(tmp["Revenue_M"].sum())})

    if not rows_market:
        st.warning("No usable revenue data found for the selected zone.")
        st.stop()

    df_market = pd.DataFrame(rows_market).groupby("Market", as_index=False)["Revenue_M"].sum()
    total_zone = float(df_market["Revenue_M"].sum())
    if total_zone <= 0:
        st.warning("Total revenue is 0 for the selected zone (nothing to display).")
        st.stop()

    df_market["Share_%"] = (df_market["Revenue_M"] / total_zone * 100).round(1)
    df_market["Market_label"] = df_market["Market"].map(market_label).fillna(df_market["Market"])
    df_plot_market = df_market.sort_values("Revenue_M", ascending=True).copy()
    df_plot_market["Share_label"] = df_plot_market["Share_%"].astype(str) + "%"

    st.caption(
        f"Zone: {zone_label_ui(zone_ms)} — Total revenue analyzed: {total_zone:,.0f} M$ "
        "(sum of all vertical datasets in Data_Clean for this zone)."
    )

    fig = px.bar(
        df_plot_market,
        x="Revenue_M",
        y="Market_label",
        orientation="h",
        labels={"Revenue_M": "Revenue (M$)", "Market_label": ""},
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

    # 2) Drill-down: vertical shares within selected market
    st.subheader(f"{market_label(market_ms)} · {zone_label_ui(zone_ms)} — Breakdown by Vertical")

    ds_market = [d for d in datasets_all if d.zone in zones_ms and d.market == market_ms]
    if not ds_market:
        st.info(f"No datasets found for market={market_ms} in zone={zone_ms}.")
        st.stop()

    rows_vertical = []
    for d in ds_market:
        try:
            tmp = load_dataset(d.path)
        except Exception as e:
            st.warning(f"Could not load {d.path.name}: {e}")
            continue
        if tmp.empty:
            continue
        rows_vertical.append({"Vertical": d.vertical, "Revenue_M": float(tmp["Revenue_M"].sum())})

    if not rows_vertical:
        st.info("No usable revenue data for this market / zone.")
        st.stop()

    df_vert = pd.DataFrame(rows_vertical).groupby("Vertical", as_index=False)["Revenue_M"].sum()
    total_market = float(df_vert["Revenue_M"].sum())
    if total_market <= 0:
        st.info("Total revenue is 0 for this market (nothing to display).")
        st.stop()

    df_vert["Share_%"] = (df_vert["Revenue_M"] / total_market * 100).round(1)
    df_vert["Vertical_label"] = df_vert["Vertical"].map(titleize_slug)
    df_vert = df_vert.sort_values("Revenue_M", ascending=True).copy()
    df_vert["Share_label"] = df_vert["Share_%"].astype(str) + "%"

    st.caption(f"{market_label(market_ms)} total (zone {zone_label_ui(zone_ms)}): {total_market:,.0f} M$")

    fig2 = px.bar(
        df_vert,
        x="Revenue_M",
        y="Vertical_label",
        orientation="h",
        labels={"Revenue_M": "Revenue (M$)", "Vertical_label": ""},
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
    df_table = df_vert.sort_values("Revenue_M", ascending=False)[["Vertical_label", "Revenue_M", "Share_%"]].copy()
    df_table.columns = ["Vertical", "Revenue (M$)", "Share (%)"]
    st.dataframe(df_table, use_container_width=True, height=260)
