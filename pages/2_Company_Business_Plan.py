from __future__ import annotations

from pathlib import Path
import math
import re

import pandas as pd
import streamlit as st
import plotly.express as px

from market_explorer.auth import require_auth
from market_explorer.discovery import list_datasets
from market_explorer.data_io import load_dataset, to_csv_bytes


# =============================================================================
# Guard: auth required
# =============================================================================
require_auth()
profile = st.session_state.get("profile")

# =============================================================================
# Page config + styling
# =============================================================================
C_FONCE = "#41072A"
C_ROSE = "#FF85C8"
C_WHITE = "#FFFFFF"

st.set_page_config(page_title="Account Business Plan — Hotels", layout="wide")

st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {C_WHITE}; }}
      h1, h2, h3, h4 {{ color: {C_FONCE}; }}
      .muted {{ color: rgba(0,0,0,0.55); font-size: 0.95rem; }}
      .card {{
        background: white;
        border: 1px solid rgba(65,7,42,0.10);
        border-radius: 18px;
        padding: 16px 16px 12px 16px;
        box-shadow: 0 8px 22px rgba(0,0,0,0.04);
      }}
      .kpi-title {{ color: rgba(0,0,0,0.55); font-size: 0.85rem; }}
      .kpi-value {{ color: {C_FONCE}; font-size: 1.35rem; font-weight: 750; }}
      .badge {{
        display:inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(255,133,200,0.18);
        color: {C_FONCE};
        font-weight: 750;
        font-size: 0.85rem;
      }}
      .small {{ font-size: 0.85rem; color: rgba(0,0,0,0.55); }}
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helpers
# =============================================================================
def _safe_float(x, default=0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _normalize_rate(value: float) -> float:
    """Accepts 0-1 or 0-100 inputs."""
    v = _safe_float(value, 0.0)
    if v > 1.0:
        return v / 100.0
    return v


def _fmt_money(x: float) -> str:
    return f"€{x:,.0f}".replace(",", " ")


def _fmt_money_m(x: float) -> str:
    return f"€{x/1_000_000:,.2f}M".replace(",", " ")


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _keyify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(value))


def _coerce_money_value(x) -> float | None:
    """Parse values like '1,2M', '800k', '1.1B', '€1 200 000' -> float in EUR."""
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None

    s = s.replace("€", "").replace("$", "").replace("£", "").replace(" ", "")

    mult = 1.0
    last = s[-1:].upper()
    if last == "B":
        mult = 1_000_000_000.0
        s = s[:-1]
    elif last == "M":
        mult = 1_000_000.0
        s = s[:-1]
    elif last == "K":
        mult = 1_000.0
        s = s[:-1]

    # Handle separators
    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    try:
        return float(s) * mult
    except Exception:
        return None


def _ensure_hotel_fields(df: pd.DataFrame, source_zone: str | None = None) -> pd.DataFrame:
    """
    Normalize df into:
    hotel_id, hotel_name, brand, account_name, rev_2024_eur, country, zone
    """
    df = df.copy()

    hotel_id_col = _pick_column(df, ["hotel_id", "Hotel ID", "ID", "id"])
    name_col = _pick_column(df, ["hotel_name", "Hotel Name", "Name", "Company Name", "company_name"])
    brand_col = _pick_column(df, ["brand", "Brand", "Chain", "Group", "Marque"])
    account_col = _pick_column(df, ["account_name", "Account Name", "Account", "Groupe", "ChainName", "GroupName"])
    country_col = _pick_column(df, ["country", "Country", "Pays"])

    revenue_col = _pick_column(
        df,
        [
            "rev_2024_eur",
            "rev_2024",
            "revenue_2024_eur",
            "Revenue_EUR",
            "Revenue",
            "Revenue_M",  # if your convention is in "M€" or "M$" we treat as millions
            "total_rev_m",
            "revenue_m",
        ],
    )

    # hotel_id
    if hotel_id_col:
        df["hotel_id"] = df[hotel_id_col].astype(str)
    else:
        df["hotel_id"] = df.index.astype(str)

    # hotel_name
    if name_col:
        df["hotel_name"] = df[name_col].astype(str)
    else:
        df["hotel_name"] = "Unnamed hotel"

    # brand
    if brand_col:
        df["brand"] = df[brand_col].fillna("Unknown").astype(str)
    else:
        df["brand"] = "Unknown"

    # account_name
    if account_col:
        df["account_name"] = df[account_col].fillna("Unknown").astype(str)
    else:
        df["account_name"] = df["brand"].fillna("Unknown").astype(str)

    # country
    if country_col:
        df["country"] = df[country_col].fillna("Unknown").astype(str)
    else:
        df["country"] = "Unknown"

    # revenue
    df["rev_2024_eur"] = 0.0
    if revenue_col:
        if revenue_col.lower().endswith("_m") or revenue_col in {"Revenue_M", "total_rev_m", "revenue_m"}:
            # convention: millions
            df["rev_2024_eur"] = pd.to_numeric(df[revenue_col], errors="coerce").fillna(0.0) * 1_000_000.0
        else:
            df["rev_2024_eur"] = (
                df[revenue_col].apply(_coerce_money_value).fillna(0.0).astype(float)
            )

    df["zone"] = source_zone or "unknown"

    return df[
        ["hotel_id", "hotel_name", "brand", "account_name", "rev_2024_eur", "country", "zone"]
    ]


def _traffic_light(
    loss_ratio: float,
    broker_rev_m: float,
    lr_green: float,
    lr_orange: float,
    rev_green_m: float,
    rev_orange_m: float,
) -> tuple[str, str]:
    lr_score = 2 if loss_ratio <= lr_green else (1 if loss_ratio <= lr_orange else 0)
    rev_score = 2 if broker_rev_m >= rev_green_m else (1 if broker_rev_m >= rev_orange_m else 0)
    score = min(lr_score, rev_score)

    if score == 2:
        return "🟢 Go", "Attractive (good S/P + sufficient broker revenue)."
    if score == 1:
        return "🟠 Review", "Potentially interesting but needs validation / negotiation."
    return "🔴 No-go", "Unattractive on risk or economics (as modeled)."


def _scenario_defaults(kind: str) -> dict:
    if kind == "Pessimistic":
        return dict(take_rate_mult=0.80, premium_mult=0.90, claim_rate_mult=1.20, claim_amt_mult=1.10)
    if kind == "Optimistic":
        return dict(take_rate_mult=1.20, premium_mult=1.10, claim_rate_mult=0.80, claim_amt_mult=0.90)
    return dict(take_rate_mult=1.00, premium_mult=1.00, claim_rate_mult=1.00, claim_amt_mult=1.00)


def _hotel_option_label(row: pd.Series) -> str:
    name = str(row.get("hotel_name", ""))
    brand = str(row.get("brand", ""))
    rev = _safe_float(row.get("rev_2024_eur", 0.0))
    return f"{name} — {brand} — {_fmt_money(rev)}"


# =============================================================================
# Header
# =============================================================================
st.title("Account Business Plan — Hotels")
st.markdown(
    '<div class="muted">Account-based Hotel BP for <b>Assurance Neat</b>. Scope is selected per account type (Groupe / Chaîne / Indépendant).</div>',
    unsafe_allow_html=True,
)
st.write("")


# =============================================================================
# Load HOTEL dataset(s)
# =============================================================================
DATA_DIR = Path(__file__).resolve().parents[1] / "Data_Clean"
datasets = list_datasets(DATA_DIR)

if not datasets:
    st.error(f"No datasets found in {DATA_DIR}.")
    st.stop()

hotel_ds = [d for d in datasets if (d.vertical or "").lower() == "hotel" or "hotel" in (d.vertical or "").lower()]
if not hotel_ds:
    st.error("No HOTEL dataset found (vertical 'hotel').")
    st.stop()

# =============================================================================
# Sidebar: navigation + zone + load df + scope settings
# =============================================================================

with st.sidebar:
    st.markdown("### Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("pages/0_Home.py")
    if st.button("🔎 Market Explorer", use_container_width=True):
        st.switch_page("pages/1_Market_Explorer.py")
    st.button("📈 BP Hôtellerie", use_container_width=True, disabled=True)
    if st.button("🏨 Account BP Hotels", use_container_width=True):
        st.switch_page("pages/3_Account_Business_Plan_Hotels.py")
    st.divider()


    ZONE_UI = ["France", "Europe", "Europe + France"]
    ZONE_MAP = {"France": "france", "Europe": "eu", "Europe + France": "eu_fr"}
    zone_ui = st.selectbox("Zone", ZONE_UI, index=0)
    zone = ZONE_MAP[zone_ui]

# Load df for selected zone
ds_zone = [d for d in hotel_ds if getattr(d, "zone", None) == zone]
if not ds_zone:
    st.warning(f"No hotel dataset for zone={zone_ui}.")
    st.stop()

frames: list[pd.DataFrame] = []
for d in ds_zone:
    try:
        tmp = load_dataset(d.path)
        if tmp is None or tmp.empty:
            continue
        frames.append(_ensure_hotel_fields(tmp, source_zone=getattr(d, "zone", None)))
    except Exception as e:
        st.warning(f"Could not load {d.path.name}: {e}")

if not frames:
    st.warning("Hotel dataset is empty / unreadable.")
    st.stop()

df = pd.concat(frames, ignore_index=True)

missing_brand = (df["brand"] == "Unknown").mean() > 0.5
missing_country = (df["country"] == "Unknown").mean() > 0.5
missing_revenue = (df["rev_2024_eur"] <= 0).mean() > 0.5

with st.sidebar:
    if missing_brand:
        st.warning("Brand missing/empty → brand-level scope may be limited.")
    if missing_country:
        st.warning("Country missing/empty → country filters may be limited.")
    if missing_revenue:
        st.warning("Revenue missing/empty → CA may be understated.")

    st.divider()
    st.header("Account scope")
    account_type = st.radio(
        "Account type",
        options=["Groupe", "Chaîne", "Indépendant"],
        key="bp_account_type",
    )

    st.caption("Optional: pick one hotel to display a reference card.")
    query = st.text_input("Search hotel", value="", placeholder="e.g. Accor, Marriott...").strip().lower()
    df_pick = df.copy()
    if query:
        df_pick = df_pick[df_pick["hotel_name"].astype(str).str.lower().str.contains(query, na=False)]
    df_pick = df_pick.sort_values("rev_2024_eur", ascending=False).head(250).reset_index(drop=True)

    if len(df_pick) > 0:
        idx = st.selectbox(
            "Reference hotel",
            options=list(range(len(df_pick))),
            format_func=lambda i: _hotel_option_label(df_pick.iloc[i]),
            index=0,
        )
        ref_hotel = df_pick.iloc[int(idx)]
    else:
        ref_hotel = None


# =============================================================================
# Assumptions (main page inputs)
# =============================================================================
st.subheader("Assumptions")
st.caption("Define the scope in the sidebar. Adjust business inputs here.")

tab_hotel, tab_market, tab_risk, tab_decision = st.tabs(
    ["Hotel setup", "Market & pricing", "Risk & broker", "Decision & scenarios"]
)

with tab_hotel:
    hotel_type = st.selectbox("Hotel type", ["Independent", "Chain", "Group"], index=0)
    st.caption("Hotel type impacts distribution, negotiation power, pricing and risk mix.")

    DEFAULT_TYPE_MULT = {
        "Independent": {"take_rate": 0.85, "commission": 1.10, "premium": 0.95, "claim_rate": 0.95},
        "Chain": {"take_rate": 1.00, "commission": 1.00, "premium": 1.00, "claim_rate": 1.00},
        "Group": {"take_rate": 1.15, "commission": 0.85, "premium": 0.98, "claim_rate": 1.05},
    }

    tm = DEFAULT_TYPE_MULT[hotel_type]
    with st.expander("Hotel type multipliers (editable)", expanded=False):
        st.caption("Applied on top of your base inputs and scenarios.")
        type_take_mult = st.number_input("Take rate multiplier", value=float(tm["take_rate"]), step=0.05)
        type_comm_mult = st.number_input("Broker commission multiplier", value=float(tm["commission"]), step=0.05)
        type_prem_mult = st.number_input("Premium multiplier", value=float(tm["premium"]), step=0.05)
        type_cr_mult = st.number_input("Claim rate multiplier", value=float(tm["claim_rate"]), step=0.05)

    type_mult = dict(
        take_rate=float(type_take_mult),
        commission=float(type_comm_mult),
        premium=float(type_prem_mult),
        claim_rate=float(type_cr_mult),
    )

    # Capacity / nights economics (kept MVP-style)
    if hotel_type in ["Independent", "Chain"]:
        hotel_count = st.number_input("Number of hotels", min_value=1, value=1, step=1)
        avg_rooms_per_hotel = st.number_input("Avg rooms per hotel", min_value=1, value=80, step=1)
        rooms = int(hotel_count * avg_rooms_per_hotel)
    else:
        brand_count = st.number_input("Number of brands", min_value=1, value=3, step=1)
        hotels_per_brand = st.number_input("Avg hotels per brand", min_value=1, value=20, step=1)
        avg_rooms_per_hotel = st.number_input("Avg rooms per hotel", min_value=1, value=120, step=1)
        rooms = int(brand_count * hotels_per_brand * avg_rooms_per_hotel)

    st.caption(f"Total rooms = {rooms:,}")
    avg_stay_nights = st.number_input("Avg stay length (nights)", min_value=1.0, value=2.0, step=0.5)

with tab_market:
    default_occ = {"France": 0.65, "Europe": 0.62, "Europe + France": 0.63}[zone_ui]
    occupancy = st.slider("Occupancy rate (annual average)", 0.10, 0.95, float(default_occ), 0.01)

    st.caption("Take rate = % of stays where customer adds insurance at checkout.")
    tr_cancel = st.slider("Base take rate — Cancellation", 0.00, 0.60, 0.10 if zone_ui == "France" else 0.12, 0.01)
    tr_interrupt = st.slider("Base take rate — Interruption", 0.00, 0.60, 0.07 if zone_ui == "France" else 0.08, 0.01)

    st.caption("Premium is priced as % of average stay price.")
    prem_cancel_pct = st.slider("Base premium % — Cancellation", 0.0, 0.25, 0.06, 0.005)
    prem_interrupt_pct = st.slider("Base premium % — Interruption", 0.0, 0.25, 0.04, 0.005)

with tab_risk:
    st.caption("Single average rates (MVP).")
    claim_rate_cancel = st.slider("Base claim rate — Cancellation", 0.0, 0.40, 0.06, 0.005)
    claim_rate_interrupt = st.slider("Base claim rate — Interruption", 0.0, 0.40, 0.03, 0.005)

    claim_amt_cancel = st.number_input("Avg claim amount — Cancellation (€)", min_value=0.0, value=350.0, step=50.0)
    claim_amt_interrupt = st.number_input("Avg claim amount — Interruption (€)", min_value=0.0, value=500.0, step=50.0)

    broker_commission = st.slider("Base broker commission on premium", 0.0, 0.50, 0.20, 0.01)

with tab_decision:
    lr_green = st.slider("S/P green (≤)", 0.0, 1.5, 0.55, 0.01)
    lr_orange = st.slider("S/P orange (≤)", 0.0, 2.5, 0.75, 0.01)

    rev_green_m = st.number_input("Broker rev green (M€) ≥", min_value=0.0, value=0.15, step=0.05)
    rev_orange_m = st.number_input("Broker rev orange (M€) ≥", min_value=0.0, value=0.05, step=0.01)

    with st.expander("Scenarios (edit multipliers)", expanded=False):
        st.caption("Multipliers applied on top of base inputs (and after hotel type).")
        pess = _scenario_defaults("Pessimistic")
        base = _scenario_defaults("Base")
        opti = _scenario_defaults("Optimistic")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Pessimistic**")
            pess_take = st.number_input("Take rate mult", value=float(pess["take_rate_mult"]), step=0.05, key="p_t")
            pess_prem = st.number_input("Premium mult", value=float(pess["premium_mult"]), step=0.05, key="p_p")
            pess_cr = st.number_input("Claim rate mult", value=float(pess["claim_rate_mult"]), step=0.05, key="p_cr")
            pess_ca = st.number_input("Claim amount mult", value=float(pess["claim_amt_mult"]), step=0.05, key="p_ca")
        with c2:
            st.markdown("**Base**")
            base_take = st.number_input("Take rate mult ", value=float(base["take_rate_mult"]), step=0.05, key="b_t")
            base_prem = st.number_input("Premium mult ", value=float(base["premium_mult"]), step=0.05, key="b_p")
            base_cr = st.number_input("Claim rate mult ", value=float(base["claim_rate_mult"]), step=0.05, key="b_cr")
            base_ca = st.number_input("Claim amount mult ", value=float(base["claim_amt_mult"]), step=0.05, key="b_ca")
        with c3:
            st.markdown("**Optimistic**")
            opt_take = st.number_input("Take rate mult  ", value=float(opti["take_rate_mult"]), step=0.05, key="o_t")
            opt_prem = st.number_input("Premium mult  ", value=float(opti["premium_mult"]), step=0.05, key="o_p")
            opt_cr = st.number_input("Claim rate mult  ", value=float(opti["claim_rate_mult"]), step=0.05, key="o_cr")
            opt_ca = st.number_input("Claim amount mult  ", value=float(opti["claim_amt_mult"]), step=0.05, key="o_ca")

        scenario_defs = {
            "Pessimistic": dict(take_rate_mult=pess_take, premium_mult=pess_prem, claim_rate_mult=pess_cr, claim_amt_mult=pess_ca),
            "Base": dict(take_rate_mult=base_take, premium_mult=base_prem, claim_rate_mult=base_cr, claim_amt_mult=base_ca),
            "Optimistic": dict(take_rate_mult=opt_take, premium_mult=opt_prem, claim_rate_mult=opt_cr, claim_amt_mult=opt_ca),
        }


# =============================================================================
# Scope builder (Assurance Neat)
# =============================================================================
st.subheader("Scope builder")

ca_total = float(df["rev_2024_eur"].sum())
ca_scope = 0.0
scope_rows: list[dict] = []
scoped_hotels = df.iloc[0:0].copy()

if account_type == "Groupe":
    brands = sorted(df["brand"].fillna("Unknown").unique().tolist())
    selected_brands = st.multiselect(
        "Select brands",
        options=brands,
        default=brands[:1] if brands else [],
        key="bp_group_brands",
    )

    for brand in selected_brands:
        brand_df = df[df["brand"] == brand]
        brand_total = float(brand_df["rev_2024_eur"].sum())

        mode_key = f"bp_group_mode_{_keyify(brand)}"
        mode = st.radio(
            f"Scope mode for {brand}",
            options=["Select hotels", "Take rate (%)"],
            horizontal=True,
            key=mode_key,
        )

        if mode == "Select hotels":
            hotel_ids = brand_df["hotel_id"].tolist()
            selected_ids = st.multiselect(
                f"Hotels for {brand}",
                options=hotel_ids,
                format_func=lambda hid, _bdf=brand_df: _hotel_option_label(_bdf.loc[_bdf["hotel_id"] == hid].iloc[0]),
                key=f"bp_group_hotels_{_keyify(brand)}",
            )
            brand_scope_df = brand_df[brand_df["hotel_id"].isin(selected_ids)]
            brand_scope = float(brand_scope_df["rev_2024_eur"].sum())
        else:
            take_rate_input = st.number_input(
                f"Take rate for {brand} (% of hotels)",
                min_value=0.0,
                max_value=100.0,
                value=50.0,
                step=1.0,
                key=f"bp_group_take_{_keyify(brand)}",
            )
            take_rate = _normalize_rate(take_rate_input)
            brand_scope = brand_total * take_rate
            brand_scope_df = brand_df.copy()

        ca_scope += brand_scope
        scope_rows.append(
            {"Brand": brand, "Hotels (scope)": int(len(brand_scope_df)), "CA brand total": brand_total, "CA brand scope": brand_scope}
        )
        scoped_hotels = pd.concat([scoped_hotels, brand_scope_df], ignore_index=True)

elif account_type == "Chaîne":
    scope_mode = st.radio(
        "Scope mode",
        options=["All hotels", "Filter by country-zone", "Select hotels", "Take rate (%)"],
        key="bp_chain_mode",
    )

    if scope_mode == "All hotels":
        scoped_hotels = df.copy()
        ca_scope = ca_total

    elif scope_mode == "Filter by country-zone":
        zones = sorted(df["zone"].dropna().unique().tolist())
        selected_zones = st.multiselect("Zones", options=zones, key="bp_chain_zones")
        zone_df = df if not selected_zones else df[df["zone"].isin(selected_zones)]

        countries = sorted(zone_df["country"].dropna().unique().tolist())
        selected_countries = st.multiselect("Countries", options=countries, key="bp_chain_countries")
        scoped_hotels = zone_df if not selected_countries else zone_df[zone_df["country"].isin(selected_countries)]
        ca_scope = float(scoped_hotels["rev_2024_eur"].sum())

        if scoped_hotels.empty:
            st.warning("No hotels match the selected zones/countries.")

    elif scope_mode == "Select hotels":
        hotel_ids = df["hotel_id"].tolist()
        selected_ids = st.multiselect(
            "Select hotels",
            options=hotel_ids,
            format_func=lambda hid: _hotel_option_label(df.loc[df["hotel_id"] == hid].iloc[0]),
            key="bp_chain_hotels",
        )
        scoped_hotels = df[df["hotel_id"].isin(selected_ids)]
        ca_scope = float(scoped_hotels["rev_2024_eur"].sum())

    else:
        take_rate_input = st.number_input(
            "Take rate (% of hotels)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=1.0,
            key="bp_chain_take",
        )
        take_rate = _normalize_rate(take_rate_input)
        ca_scope = ca_total * take_rate
        scoped_hotels = df.copy()

        scope_rows = (
            df.groupby("brand", as_index=False)
            .agg(**{"Hotels (scope)": ("hotel_id", "count"), "CA brand total": ("rev_2024_eur", "sum")})
            .assign(**{"CA brand scope": lambda x: x["CA brand total"] * take_rate})
            .to_dict("records")
        )

else:  # Indépendant
    hotel_ids = df["hotel_id"].tolist()
    selected_id = st.selectbox(
        "Select hotel",
        options=hotel_ids,
        format_func=lambda hid: _hotel_option_label(df.loc[df["hotel_id"] == hid].iloc[0]),
        key="bp_independent_hotel",
    )
    scoped_hotels = df[df["hotel_id"] == selected_id]
    ca_scope = float(scoped_hotels["rev_2024_eur"].sum())
    scope_rows = (
        scoped_hotels.groupby("brand", as_index=False)
        .agg(**{"Hotels (scope)": ("hotel_id", "count"), "CA brand total": ("rev_2024_eur", "sum")})
        .assign(**{"CA brand scope": lambda x: x["CA brand total"]})
        .to_dict("records")
    )

# fallback scope rows
if account_type in {"Groupe", "Chaîne"} and not scope_rows and not scoped_hotels.empty:
    scope_rows = (
        scoped_hotels.groupby("brand", as_index=False)
        .agg(**{"Hotels (scope)": ("hotel_id", "count"), "CA brand total": ("rev_2024_eur", "sum")})
        .assign(**{"CA brand scope": lambda x: x["CA brand total"]})
        .to_dict("records")
    )


# =============================================================================
# Reference hotel card (optional)
# =============================================================================
if ref_hotel is not None:
    st.markdown(
        f"""
        <div class="card">
          <div class="badge">Reference hotel</div>
          <h3 style="margin:10px 0 6px 0;">{ref_hotel.get("hotel_name","")}</h3>
          <div class="muted">
            Brand: <b>{ref_hotel.get("brand","")}</b> · Country: <b>{ref_hotel.get("country","")}</b> · Zone: <b>{ref_hotel.get("zone","")}</b><br/>
            Revenue (2024): <b>{_fmt_money(float(ref_hotel.get("rev_2024_eur",0.0)))}</b>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


# =============================================================================
# Derived booking economics (needed because premium is % of stay price)
# We use CA_scope and capacity parameters as a proxy.
# =============================================================================
room_nights = float(rooms) * 365.0 * float(occupancy)
stays = room_nights / float(avg_stay_nights) if avg_stay_nights > 0 else 0.0
adr = (ca_scope / room_nights) if room_nights > 0 else 0.0
avg_stay_price = adr * float(avg_stay_nights)


# =============================================================================
# Products (broker economics)
# =============================================================================
products = [
    dict(product="Cancellation", base_take_rate=tr_cancel, base_prem_pct=prem_cancel_pct, base_claim_rate=claim_rate_cancel, base_claim_amt=claim_amt_cancel),
    dict(product="Interruption", base_take_rate=tr_interrupt, base_prem_pct=prem_interrupt_pct, base_claim_rate=claim_rate_interrupt, base_claim_amt=claim_amt_interrupt),
]


# =============================================================================
# Compute projections by product & scenario
# =============================================================================
rows: list[dict] = []
for scen, mult in scenario_defs.items():
    for p in products:
        take = float(p["base_take_rate"]) * float(mult["take_rate_mult"]) * float(type_mult["take_rate"])
        take = max(0.0, min(1.0, take))

        prem_pct = float(p["base_prem_pct"]) * float(mult["premium_mult"]) * float(type_mult["premium"])
        prem_pct = max(0.0, prem_pct)

        claim_rate = float(p["base_claim_rate"]) * float(mult["claim_rate_mult"]) * float(type_mult["claim_rate"])
        claim_rate = max(0.0, min(1.0, claim_rate))

        claim_amt = float(p["base_claim_amt"]) * float(mult["claim_amt_mult"])
        claim_amt = max(0.0, claim_amt)

        policies = stays * take
        gwp = policies * avg_stay_price * prem_pct
        expected_claims = policies * claim_rate * claim_amt
        loss_ratio = (expected_claims / gwp) if gwp > 0 else 0.0

        eff_commission = float(broker_commission) * float(type_mult["commission"])
        eff_commission = max(0.0, min(0.80, eff_commission))
        broker_rev = gwp * eff_commission

        rows.append(
            dict(
                Scenario=scen,
                Product=p["product"],
                Rooms=int(rooms),
                Occupancy=float(occupancy),
                AvgStayNights=float(avg_stay_nights),
                StaysPerYear=float(stays),
                ADR=float(adr),
                AvgStayPrice=float(avg_stay_price),
                TakeRate=float(take),
                PremiumPct=float(prem_pct),
                ClaimRate=float(claim_rate),
                ClaimAmt=float(claim_amt),
                Policies=float(policies),
                GWP=float(gwp),
                ExpectedClaims=float(expected_claims),
                LossRatio=float(loss_ratio),
                BrokerCommission=float(eff_commission),
                BrokerRevenue=float(broker_rev),
            )
        )

proj = pd.DataFrame(rows)

agg = (
    proj.groupby("Scenario", as_index=False)
    .agg(GWP=("GWP", "sum"), ExpectedClaims=("ExpectedClaims", "sum"), BrokerRevenue=("BrokerRevenue", "sum"))
)
agg["LossRatio"] = agg.apply(lambda r: (r["ExpectedClaims"] / r["GWP"]) if r["GWP"] > 0 else 0.0, axis=1)

proj["GWP_M"] = proj["GWP"] / 1_000_000.0
proj["ExpectedClaims_M"] = proj["ExpectedClaims"] / 1_000_000.0
proj["BrokerRevenue_M"] = proj["BrokerRevenue"] / 1_000_000.0

agg["GWP_M"] = agg["GWP"] / 1_000_000.0
agg["ExpectedClaims_M"] = agg["ExpectedClaims"] / 1_000_000.0
agg["BrokerRevenue_M"] = agg["BrokerRevenue"] / 1_000_000.0


# =============================================================================
# Scope summary
# =============================================================================
base_row = agg[agg["Scenario"] == "Base"]
base_gwp_m = float(base_row["GWP_M"].iloc[0]) if not base_row.empty else 0.0
base_lr = float(base_row["LossRatio"].iloc[0]) if not base_row.empty else 0.0
base_brev_m = float(base_row["BrokerRevenue_M"].iloc[0]) if not base_row.empty else 0.0

brand_count = int(scoped_hotels["brand"].nunique()) if not scoped_hotels.empty else 0
hotel_count = int(scoped_hotels["hotel_id"].nunique()) if not scoped_hotels.empty else 0

decision, decision_msg = _traffic_light(base_lr, base_brev_m, lr_green, lr_orange, rev_green_m, rev_orange_m)

st.subheader("Scope summary")
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("CA total", _fmt_money(ca_total))
with k2:
    st.metric("CA scope", _fmt_money(ca_scope))
with k3:
    st.metric("Brands in scope", brand_count)
with k4:
    st.metric("Hotels in scope", hotel_count)

c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.4], gap="large")
with c1:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">Decision (Base)</div>
          <div class="kpi-value">{decision}</div>
          <div class="muted" style="margin-top:6px;">{decision_msg}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">GWP (Base)</div>
          <div class="kpi-value">{_fmt_money_m(base_gwp_m*1_000_000)}</div>
          <div class="muted" style="margin-top:6px;">Gross written premium</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">S/P (Base)</div>
          <div class="kpi-value">{base_lr:.1%}</div>
          <div class="muted" style="margin-top:6px;">Loss ratio = claims / premium</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f"""
        <div class="card">
          <div class="kpi-title">Broker net revenue (Base)</div>
          <div class="kpi-value">{_fmt_money_m(base_brev_m*1_000_000)}</div>
          <div class="muted" style="margin-top:6px;">Commission × GWP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if scope_rows:
    st.write("")
    scope_table = pd.DataFrame(scope_rows)
    for col in ["CA brand total", "CA brand scope"]:
        if col in scope_table.columns:
            scope_table[col] = scope_table[col].apply(_safe_float)
            scope_table[col] = scope_table[col].map(_fmt_money)
    st.dataframe(scope_table, use_container_width=True, hide_index=True)


# =============================================================================
# Scenario summary table
# =============================================================================
st.subheader("Scenario summary (all products)")
show = agg.copy()[["Scenario", "GWP_M", "ExpectedClaims_M", "LossRatio", "BrokerRevenue_M"]].sort_values("Scenario")
show.columns = ["Scenario", "GWP (M€)", "Expected claims (M€)", "S/P", "Broker net rev (M€)"]
show["S/P"] = show["S/P"].map(lambda x: f"{float(x):.1%}")
st.dataframe(show, use_container_width=True, hide_index=True)


# =============================================================================
# BP parameters (Assurance Neat)
# =============================================================================
st.subheader("BP parameters")

c1, c2, c3 = st.columns(3)
with c1:
    growth_yoy_input = st.number_input("Growth YoY (%)", value=5.0, step=0.5, key="bp_growth_yoy")
    eligible_rate_input = st.number_input("Eligible rate (%)", value=80.0, step=1.0, key="bp_eligible_rate")
    take_b2c_input = st.number_input("Take B2C (%)", value=20.0, step=1.0, key="bp_take_b2c")
with c2:
    premium_rate_input = st.number_input("Premium rate (%)", value=7.0, step=0.5, key="bp_premium_rate")
    sell_price_rate_input = st.number_input("Sell price rate (%)", value=10.0, step=0.5, key="bp_sell_price_rate")
    vat_rate_input = st.number_input("VAT rate (%)", value=20.0, step=1.0, key="bp_vat_rate")
with c3:
    marketing_y1 = st.number_input("Marketing budget Y1 (€)", value=0.0, step=10_000.0, key="bp_marketing_y1")
    marketing_y2 = st.number_input("Marketing budget Y2 (€)", value=0.0, step=10_000.0, key="bp_marketing_y2")
    marketing_y3 = st.number_input("Marketing budget Y3 (€)", value=0.0, step=10_000.0, key="bp_marketing_y3")

growth = _normalize_rate(growth_yoy_input)
eligible_rate = _normalize_rate(eligible_rate_input)
take_b2c = _normalize_rate(take_b2c_input)
premium_rate = _normalize_rate(premium_rate_input)
sell_price_rate = _normalize_rate(sell_price_rate_input)
vat_rate = _normalize_rate(vat_rate_input)

ca_y1 = ca_scope
ca_y2 = ca_y1 * (1 + growth)
ca_y3 = ca_y2 * (1 + growth)


def _bp_row(year_label: str, ca_value: float, marketing: float) -> dict:
    eligible = ca_value * eligible_rate
    covered = eligible * take_b2c
    premium_ttc = covered * premium_rate
    billed_ttc = covered * sell_price_rate
    commission_ttc = billed_ttc - premium_ttc
    net_ttc = commission_ttc - marketing
    billed_ht = billed_ttc / (1 + vat_rate) if vat_rate > 0 else billed_ttc

    return {
        "Year": year_label,
        "CA": ca_value,
        "Eligible": eligible,
        "Covered": covered,
        "Premium TTC": premium_ttc,
        "Billed TTC": billed_ttc,
        "Billed HT": billed_ht,
        "Commission TTC": commission_ttc,
        "Marketing": marketing,
        "Net TTC": net_ttc,
    }


bp_rows = [
    _bp_row("Y1", ca_y1, marketing_y1),
    _bp_row("Y2", ca_y2, marketing_y2),
    _bp_row("Y3", ca_y3, marketing_y3),
]
bp_df = pd.DataFrame(bp_rows)

st.write("")
st.subheader("3-year BP (based on CA scope)")
bp_display = bp_df.copy()
for col in ["CA", "Eligible", "Covered", "Premium TTC", "Billed TTC", "Billed HT", "Commission TTC", "Marketing", "Net TTC"]:
    bp_display[col] = bp_display[col].map(_fmt_money)

st.dataframe(bp_display, use_container_width=True, hide_index=True)

summary_net = float(bp_df["Net TTC"].sum())
st.markdown(f"**Total net (3Y): {_fmt_money(summary_net)}**")

if ca_scope <= 0:
    st.warning("CA scope is 0. Please refine the scope builder selections to compute the BP.")


# =============================================================================
# Charts
# =============================================================================
st.write("")
cA, cB = st.columns(2, gap="large")
with cA:
    st.subheader("Broker net revenue by scenario")
    fig = px.bar(
        agg.sort_values("BrokerRevenue_M", ascending=True),
        x="BrokerRevenue_M",
        y="Scenario",
        orientation="h",
        labels={"BrokerRevenue_M": "Broker net revenue (M€)", "Scenario": ""},
        color="Scenario",
    )
    fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10), height=320)
    st.plotly_chart(fig, use_container_width=True)

with cB:
    st.subheader("S/P (loss ratio) by scenario")
    fig2 = px.bar(
        agg.sort_values("LossRatio", ascending=True),
        x="LossRatio",
        y="Scenario",
        orientation="h",
        labels={"LossRatio": "S/P (loss ratio)", "Scenario": ""},
        color="Scenario",
    )
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10), height=320)
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)


# =============================================================================
# Detail view per product
# =============================================================================
st.write("")
st.subheader("Detail by product × scenario")

detail_view = proj[
    [
        "Scenario",
        "Product",
        "Policies",
        "GWP_M",
        "ExpectedClaims_M",
        "LossRatio",
        "BrokerRevenue_M",
        "TakeRate",
        "PremiumPct",
        "ClaimRate",
        "BrokerCommission",
    ]
].copy()

detail_view["LossRatio"] = detail_view["LossRatio"].map(lambda x: f"{float(x):.1%}")
detail_view["TakeRate"] = detail_view["TakeRate"].map(lambda x: f"{float(x):.1%}")
detail_view["PremiumPct"] = detail_view["PremiumPct"].map(lambda x: f"{float(x):.2%}")
detail_view["ClaimRate"] = detail_view["ClaimRate"].map(lambda x: f"{float(x):.2%}")
detail_view["BrokerCommission"] = detail_view["BrokerCommission"].map(lambda x: f"{float(x):.1%}")

detail_view.columns = [
    "Scenario",
    "Product",
    "Policies (est.)",
    "GWP (M€)",
    "Expected claims (M€)",
    "S/P",
    "Broker net rev (M€)",
    "Take rate (eff.)",
    "Premium % (eff.)",
    "Claim rate (eff.)",
    "Broker commission (eff.)",
]

st.dataframe(detail_view, use_container_width=True, hide_index=True, height=420)


# =============================================================================
# Exports
# =============================================================================
st.write("")
st.subheader("Exports")

assumptions = {
    "Zone": zone_ui,
    "AccountType": account_type,
    "HotelType": hotel_type,
    "Rooms": rooms,
    "Occupancy": occupancy,
    "AvgStayNights": avg_stay_nights,
    "BaseBrokerCommission": broker_commission,
    "TypeMult_TakeRate": type_mult["take_rate"],
    "TypeMult_Premium": type_mult["premium"],
    "TypeMult_ClaimRate": type_mult["claim_rate"],
    "TypeMult_Commission": type_mult["commission"],
    "CA_total": ca_total,
    "CA_scope": ca_scope,
    "Growth": growth,
    "EligibleRate": eligible_rate,
    "TakeB2C": take_b2c,
    "PremiumRate": premium_rate,
    "SellPriceRate": sell_price_rate,
    "VAT": vat_rate,
    "Marketing_Y1": marketing_y1,
    "Marketing_Y2": marketing_y2,
    "Marketing_Y3": marketing_y3,
}
assump_df = pd.DataFrame([assumptions])

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Download assumptions (CSV)",
        data=to_csv_bytes(assump_df),
        file_name=f"bp_hotels_assumptions_{zone}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    st.download_button(
        "Download projections (detail) (CSV)",
        data=to_csv_bytes(proj),
        file_name=f"bp_hotels_projections_detail_{zone}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c3:
    st.download_button(
        "Download scenario summary (CSV)",
        data=to_csv_bytes(agg),
        file_name=f"bp_hotels_projections_summary_{zone}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("BP computed on CA_scope (not CA_total), with account-level scope selection for Assurance Neat.")
