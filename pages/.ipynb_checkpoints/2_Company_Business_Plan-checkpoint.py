from __future__ import annotations

from pathlib import Path
import math
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

st.set_page_config(page_title="Company Business Plan — Hotels", layout="wide")

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
      .ok {{ color: #1f7a3f; font-weight: 700; }}
      .warn {{ color: #a46a00; font-weight: 700; }}
      .bad {{ color: #b42318; font-weight: 700; }}
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


def _fmt_money_m(x: float) -> str:
    return f"{x:,.2f} M$"


def _fmt_money(x: float) -> str:
    return f"${x:,.0f}"


def _traffic_light(
    loss_ratio: float,
    broker_rev_m: float,
    lr_green: float,
    lr_orange: float,
    rev_green_m: float,
    rev_orange_m: float,
) -> tuple[str, str]:
    """
    Combined decision:
    - Needs acceptable loss ratio AND enough broker revenue
    """
    lr_score = 2 if loss_ratio <= lr_green else (1 if loss_ratio <= lr_orange else 0)
    rev_score = 2 if broker_rev_m >= rev_green_m else (1 if broker_rev_m >= rev_orange_m else 0)

    score = min(lr_score, rev_score)

    if score == 2:
        return "🟢 Go", "Attractive (good S/P + sufficient broker revenue)."
    if score == 1:
        return "🟠 Review", "Potentially interesting but needs validation / negotiation."
    return "🔴 No-go", "Unattractive on risk or economics (as modeled)."


def _scenario_defaults(kind: str) -> dict:
    """
    Multipliers applied on top of base inputs:
    - take_rate_mult: conversion at checkout
    - premium_mult: premium % of stay price
    - claim_rate_mult: frequency
    - claim_amt_mult: severity
    """
    if kind == "Pessimistic":
        return dict(take_rate_mult=0.80, premium_mult=0.90, claim_rate_mult=1.20, claim_amt_mult=1.10)
    if kind == "Optimistic":
        return dict(take_rate_mult=1.20, premium_mult=1.10, claim_rate_mult=0.80, claim_amt_mult=0.90)
    return dict(take_rate_mult=1.00, premium_mult=1.00, claim_rate_mult=1.00, claim_amt_mult=1.00)


def _coerce_money_to_m(x):
    """
    Convert messy revenue values into "millions" (M$).
    Accepts numbers, '1,234', '$1.2M', '0.8B', etc.
    If already a plain number, assume it's already in M$ (dataset convention).
    """
    if x is None:
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None

    s = s.replace("€", "").replace("$", "").replace("£", "").replace(" ", "")

    mult = 1.0
    if s[-1:].upper() == "B":
        mult = 1000.0
        s = s[:-1]
    elif s[-1:].upper() == "M":
        mult = 1.0
        s = s[:-1]
    elif s[-1:].upper() == "K":
        mult = 0.001
        s = s[:-1]

    if "," in s and "." in s:
        s = s.replace(",", "")
    else:
        s = s.replace(",", ".")

    try:
        return float(s) * mult
    except Exception:
        return None


def _ensure_revenue_m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has a usable Revenue_M column.
    If Revenue_M is missing/mostly zeros, try to infer from other revenue-like columns.
    """
    if "Revenue_M" not in df.columns:
        df["Revenue_M"] = 0.0

    df["Revenue_M"] = pd.to_numeric(df["Revenue_M"], errors="coerce").fillna(0.0)

    if len(df) == 0:
        return df

    filled_ratio = (df["Revenue_M"] > 0).mean()
    if filled_ratio >= 0.20:
        return df

    candidates = []
    for c in df.columns:
        cl = c.lower().strip()
        if ("rev" in cl) or ("revenue" in cl) or ("turnover" in cl) or (cl == "ca") or ("chiffre" in cl):
            if c != "Revenue_M":
                candidates.append(c)

    candidates = sorted(
        candidates,
        key=lambda c: (
            0 if ("_m" in c.lower() or "(m" in c.lower() or "million" in c.lower()) else 1,
            len(c),
        ),
    )

    for c in candidates:
        tmp = df[c].apply(_coerce_money_to_m)
        tmp_num = pd.to_numeric(tmp, errors="coerce")
        if tmp_num.notna().mean() >= 0.50 and tmp_num.fillna(0).sum() > 0:
            df["Revenue_M"] = tmp_num.fillna(0.0)
            break

    return df


# =============================================================================
# Header
# =============================================================================
st.title("Company Business Plan — Hotels")
st.markdown(
    '<div class="muted">For an <b>insurance broker</b>: decide whether targeting this hotel is economically attractive (GWP, S/P, broker net revenue).</div>',
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

hotel_ds = [d for d in datasets if d.vertical == "hotel" or "hotel" in (d.vertical or "").lower()]
if not hotel_ds:
    st.error("No HOTEL dataset found (vertical 'hotel').")
    st.stop()


# =============================================================================
# Sidebar: selection + inputs
# =============================================================================
with st.sidebar:
    st.markdown("### Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("pages/0_Home.py")
    st.divider()
    
    st.header("Scope (Hotels)")

    ZONE_UI = ["France", "Europe", "Europe + France"]
    ZONE_MAP = {"France": "france", "Europe": "eu", "Europe + France": "eu_fr"}
    zone_ui = st.selectbox("Zone", ZONE_UI, index=0)
    zone = ZONE_MAP[zone_ui]

    ds_zone = [d for d in hotel_ds if d.zone == zone]
    if not ds_zone:
        st.warning(f"No hotel dataset for zone={zone_ui}.")
        st.stop()

    dfs = []
    for d in ds_zone:
        try:
            tmp = load_dataset(d.path)
            if tmp is not None and not tmp.empty:
                dfs.append(tmp)
        except Exception as e:
            st.warning(f"Could not load {d.path.name}: {e}")

    if not dfs:
        st.warning("Hotel dataset is empty / unreadable.")
        st.stop()

    df = pd.concat(dfs, ignore_index=True)

    # Basic columns fallback
    for col in ["Name", "Country", "Company Type"]:
        if col not in df.columns:
            df[col] = None

    df = _ensure_revenue_m(df)

    st.divider()
    st.header("Select a hotel")

    query = st.text_input("Search (name contains)", value="", placeholder="e.g. Accor, Marriott...").strip().lower()
    df_pick = df.copy()
    if query:
        df_pick = df_pick[df_pick["Name"].astype(str).str.lower().str.contains(query, na=False)]
    df_pick = df_pick.sort_values("Revenue_M", ascending=False)

    if df_pick.empty:
        st.info("No hotels match your search.")
        st.stop()

    df_pick_small = df_pick.head(300).reset_index(drop=True)

    def _opt_label(r) -> str:
        name = str(r.get("Name", ""))
        country = str(r.get("Country", "") or "")
        rev = float(r.get("Revenue_M", 0.0) or 0.0)
        return f"{name} — {country} — {rev:,.0f} M$"

    idx = st.selectbox(
        "Hotel",
        options=list(range(len(df_pick_small))),
        format_func=lambda i: _opt_label(df_pick_small.iloc[i]),
        index=0,
    )
    sel = df_pick_small.iloc[int(idx)].to_dict()

# =============================================================================
# Assumptions (main page inputs)
# =============================================================================
st.subheader("Assumptions")
st.caption("Keep the selection in the sidebar. Adjust the business inputs directly here.")

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

    with st.expander("Hotel type multipliers (editable)", expanded=False):
        st.caption("Applied on top of your base inputs and scenarios.")
        tm = DEFAULT_TYPE_MULT[hotel_type]
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

    if hotel_type in ["Independent", "Chain"]:
        hotel_count = st.number_input("Number of hotels", min_value=1, value=1, step=1)
        avg_rooms_per_hotel = st.number_input("Avg rooms per hotel", min_value=1, value=80, step=1)
        rooms = int(hotel_count * avg_rooms_per_hotel)
        st.caption(f"Total rooms = {rooms:,}")
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

    st.divider()

    st.caption("Premium is priced as % of average stay price.")
    prem_cancel_pct = st.slider("Base premium % — Cancellation", 0.0, 0.25, 0.06, 0.005)
    prem_interrupt_pct = st.slider("Base premium % — Interruption", 0.0, 0.25, 0.04, 0.005)
    
with tab_risk:
    st.caption("Single average rates (MVP).")
    claim_rate_cancel = st.slider("Base claim rate — Cancellation", 0.0, 0.40, 0.06, 0.005)
    claim_rate_interrupt = st.slider("Base claim rate — Interruption", 0.0, 0.40, 0.03, 0.005)

    claim_amt_cancel = st.number_input("Avg claim amount — Cancellation ($)", min_value=0.0, value=350.0, step=50.0)
    claim_amt_interrupt = st.number_input("Avg claim amount — Interruption ($)", min_value=0.0, value=500.0, step=50.0)

    broker_commission = st.slider("Base broker commission on premium", 0.0, 0.50, 0.20, 0.01)
with tab_decision:
    lr_green = st.slider("S/P green (≤)", 0.0, 1.5, 0.55, 0.01)
    lr_orange = st.slider("S/P orange (≤)", 0.0, 2.5, 0.75, 0.01)

    rev_green_m = st.number_input("Broker rev green (M$) ≥", min_value=0.0, value=0.15, step=0.05)
    rev_orange_m = st.number_input("Broker rev orange (M$) ≥", min_value=0.0, value=0.05, step=0.01)

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
# Selected hotel info
# =============================================================================
name = str(sel.get("Name", "")) if sel else ""
country = str(sel.get("Country", "") or "") if sel else ""
rev_m = float(sel.get("Revenue_M", 0.0) or 0.0) if sel else 0.0
company_type_db = str(sel.get("Company Type", "") or "") if sel else ""

st.markdown(
    f"""
    <div class="card">
      <div class="badge">Selected hotel</div>
      <h3 style="margin:10px 0 6px 0;">{name}</h3>
      <div class="muted">
        Zone: <b>{zone_ui}</b> · Country: <b>{country}</b> · DB Company Type: <b>{company_type_db or "—"}</b><br/>
        Hotel revenue (DB): <b>{rev_m:,.0f} M$</b>
      </div>
      <div class="small" style="margin-top:10px;">
        Hotel type chosen: <b>{hotel_type}</b> —
        multipliers applied:
        take rate × <b>{type_mult["take_rate"]:.2f}</b>,
        premium × <b>{type_mult["premium"]:.2f}</b>,
        claim rate × <b>{type_mult["claim_rate"]:.2f}</b>,
        broker commission × <b>{type_mult["commission"]:.2f}</b>.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


if rev_m <= 0:
    st.error(
        "Revenue for this hotel is 0 (or not mapped). "
        "That makes ADR/stay price 0 → GWP 0 → S/P may display 0%. "
        "Check your dataset revenue column mapping."
    )
    with st.expander("Debug: dataset columns", expanded=False):
        st.write(list(df.columns))
    st.stop()


# =============================================================================
# Derive booking economics from revenue (needed because premium is % of stay price)
# =============================================================================
rev = rev_m * 1_000_000.0  # dollars

room_nights = float(rooms) * 365.0 * float(occupancy)
stays = room_nights / float(avg_stay_nights) if avg_stay_nights > 0 else 0.0

adr = (rev / room_nights) if room_nights > 0 else 0.0
avg_stay_price = adr * float(avg_stay_nights)


# =============================================================================
# Products
# =============================================================================
products = [
    dict(
        product="Cancellation",
        base_take_rate=tr_cancel,
        base_prem_pct=prem_cancel_pct,
        base_claim_rate=claim_rate_cancel,
        base_claim_amt=claim_amt_cancel,
    ),
    dict(
        product="Interruption",
        base_take_rate=tr_interrupt,
        base_prem_pct=prem_interrupt_pct,
        base_claim_rate=claim_rate_interrupt,
        base_claim_amt=claim_amt_interrupt,
    ),
]


# =============================================================================
# Compute projections
# =============================================================================
rows = []
for scen, mult in scenario_defs.items():
    for p in products:
        # Take rate: base market * scenario * hotel type
        take = float(p["base_take_rate"]) * float(mult["take_rate_mult"]) * float(type_mult["take_rate"])
        take = max(0.0, min(1.0, take))

        # Premium %: base * scenario * hotel type
        prem_pct = float(p["base_prem_pct"]) * float(mult["premium_mult"]) * float(type_mult["premium"])
        prem_pct = max(0.0, prem_pct)

        # Claim rate: base * scenario * hotel type
        claim_rate = float(p["base_claim_rate"]) * float(mult["claim_rate_mult"]) * float(type_mult["claim_rate"])
        claim_rate = max(0.0, min(1.0, claim_rate))

        # Claim amount: base * scenario (leave hotel type out for MVP)
        claim_amt = float(p["base_claim_amt"]) * float(mult["claim_amt_mult"])
        claim_amt = max(0.0, claim_amt)

        policies = stays * take
        gwp = policies * avg_stay_price * prem_pct
        expected_claims = policies * claim_rate * claim_amt
        loss_ratio = (expected_claims / gwp) if gwp > 0 else 0.0

        # Broker commission: base * hotel type (negotiation power)
        eff_commission = float(broker_commission) * float(type_mult["commission"])
        eff_commission = max(0.0, min(0.80, eff_commission))  # safety cap
        broker_rev = gwp * eff_commission

        rows.append(
            dict(
                Scenario=scen,
                Product=p["product"],
                Hotel=name,
                Zone=zone_ui,
                Country=country,
                Revenue_M=rev_m,
                Rooms=int(rooms),
                Occupancy=float(occupancy),
                AvgStayNights=float(avg_stay_nights),
                StaysPerYear=float(stays),
                ADR=float(adr),
                AvgStayPrice=float(avg_stay_price),
                BaseTakeRate=float(p["base_take_rate"]),
                TakeRate=float(take),
                BasePremiumPct=float(p["base_prem_pct"]),
                PremiumPct=float(prem_pct),
                BaseClaimRate=float(p["base_claim_rate"]),
                ClaimRate=float(claim_rate),
                ClaimAmt=float(claim_amt),
                Policies=float(policies),
                GWP=float(gwp),
                ExpectedClaims=float(expected_claims),
                LossRatio=float(loss_ratio),
                BaseBrokerCommission=float(broker_commission),
                BrokerCommission=float(eff_commission),
                BrokerRevenue=float(broker_rev),
            )
        )

proj = pd.DataFrame(rows)

# Aggregate per scenario (all products)
agg = (
    proj.groupby("Scenario", as_index=False)
    .agg(
        GWP=("GWP", "sum"),
        ExpectedClaims=("ExpectedClaims", "sum"),
        BrokerRevenue=("BrokerRevenue", "sum"),
    )
)
agg["LossRatio"] = agg.apply(lambda r: (r["ExpectedClaims"] / r["GWP"]) if r["GWP"] > 0 else 0.0, axis=1)

# Convert to M$
proj["GWP_M"] = proj["GWP"] / 1_000_000.0
proj["ExpectedClaims_M"] = proj["ExpectedClaims"] / 1_000_000.0
proj["BrokerRevenue_M"] = proj["BrokerRevenue"] / 1_000_000.0

agg["GWP_M"] = agg["GWP"] / 1_000_000.0
agg["ExpectedClaims_M"] = agg["ExpectedClaims"] / 1_000_000.0
agg["BrokerRevenue_M"] = agg["BrokerRevenue"] / 1_000_000.0


# =============================================================================
# Top KPIs + decision light (Base scenario)
# =============================================================================
base_row = agg[agg["Scenario"] == "Base"]
base_gwp_m = float(base_row["GWP_M"].iloc[0]) if not base_row.empty else 0.0
base_lr = float(base_row["LossRatio"].iloc[0]) if not base_row.empty else 0.0
base_brev_m = float(base_row["BrokerRevenue_M"].iloc[0]) if not base_row.empty else 0.0

decision, decision_msg = _traffic_light(base_lr, base_brev_m, lr_green, lr_orange, rev_green_m, rev_orange_m)

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
          <div class="kpi-value">{_fmt_money_m(base_gwp_m)}</div>
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
          <div class="kpi-value">{_fmt_money_m(base_brev_m)}</div>
          <div class="muted" style="margin-top:6px;">Commission × GWP</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# =============================================================================
# Explain metrics + derived econ
# =============================================================================
with st.expander("How metrics are calculated (MVP)", expanded=False):
    st.markdown(
        f"""
**From DB**
- Hotel revenue (annual): **{rev_m:,.0f} M$**

**Sales inputs**
- Rooms: **{rooms}**
- Occupancy (market): **{occupancy:.0%}**
- Avg stay length: **{avg_stay_nights:.1f} nights**

**Derived hotel economics**
- Room-nights/year = rooms × 365 × occupancy  
  → **{room_nights:,.0f}**
- Stays/year = room-nights / avg stay length  
  → **{stays:,.0f}**
- ADR (average daily rate) = revenue / room-nights  
  → **{_fmt_money(adr)}**
- Avg stay price = ADR × avg stay length  
  → **{_fmt_money(avg_stay_price)}**

**Hotel type effect (multipliers applied)**
- Take rate × **{type_mult["take_rate"]:.2f}**
- Premium % × **{type_mult["premium"]:.2f}**
- Claim rate × **{type_mult["claim_rate"]:.2f}**
- Broker commission × **{type_mult["commission"]:.2f}**

**Insurance economics (per product & scenario)**
- Policies = stays × take rate
- GWP = policies × avg stay price × premium %
- Expected claims = policies × claim rate × avg claim amount
- S/P (loss ratio) = expected claims / GWP
- Broker net revenue = GWP × broker commission
        """
    )


# =============================================================================
# Scenario summary table
# =============================================================================
st.subheader("Scenario summary (all products)")
show = agg.copy()[["Scenario", "GWP_M", "ExpectedClaims_M", "LossRatio", "BrokerRevenue_M"]].sort_values("Scenario")
show.columns = ["Scenario", "GWP (M$)", "Expected claims (M$)", "S/P", "Broker net rev (M$)"]
show["S/P"] = show["S/P"].map(lambda x: f"{float(x):.1%}")

st.dataframe(show, use_container_width=True, hide_index=True)
st.write("")


# =============================================================================
# Charts
# =============================================================================
cA, cB = st.columns(2, gap="large")

with cA:
    st.subheader("Broker net revenue by scenario")
    fig = px.bar(
        agg.sort_values("BrokerRevenue_M", ascending=True),
        x="BrokerRevenue_M",
        y="Scenario",
        orientation="h",
        labels={"BrokerRevenue_M": "Broker net revenue (M$)", "Scenario": ""},
        color="Scenario",
        color_discrete_sequence=[C_FONCE, C_ROSE, "#6D5A64"],
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
        color_discrete_sequence=[C_FONCE, C_ROSE, "#6D5A64"],
    )
    fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white", margin=dict(l=10, r=10, t=10, b=10), height=320)
    fig2.update_xaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

st.write("")


# =============================================================================
# Detail view per product
# =============================================================================
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
    "GWP (M$)",
    "Expected claims (M$)",
    "S/P",
    "Broker net rev (M$)",
    "Take rate (eff.)",
    "Premium % (eff.)",
    "Claim rate (eff.)",
    "Broker commission (eff.)",
]

st.dataframe(detail_view, use_container_width=True, hide_index=True, height=420)
st.write("")


# =============================================================================
# Exports
# =============================================================================
st.subheader("Exports")

assumptions = {
    "Hotel": name,
    "Zone": zone_ui,
    "Country": country,
    "Revenue_M": rev_m,
    "HotelType": hotel_type,
    "Rooms": rooms,
    "Occupancy": occupancy,
    "AvgStayNights": avg_stay_nights,
    "BaseBrokerCommission": broker_commission,
    "TypeMult_TakeRate": type_mult["take_rate"],
    "TypeMult_Premium": type_mult["premium"],
    "TypeMult_ClaimRate": type_mult["claim_rate"],
    "TypeMult_Commission": type_mult["commission"],
    "BaseTakeRate_Cancellation": tr_cancel,
    "BaseTakeRate_Interruption": tr_interrupt,
    "BasePremiumPct_Cancellation": prem_cancel_pct,
    "BasePremiumPct_Interruption": prem_interrupt_pct,
    "BaseClaimRate_Cancellation": claim_rate_cancel,
    "BaseClaimRate_Interruption": claim_rate_interrupt,
    "AvgClaimAmt_Cancellation": claim_amt_cancel,
    "AvgClaimAmt_Interruption": claim_amt_interrupt,
    "Threshold_LR_Green": lr_green,
    "Threshold_LR_Orange": lr_orange,
    "Threshold_BrokerRev_Green_M": rev_green_m,
    "Threshold_BrokerRev_Orange_M": rev_orange_m,
}
assump_df = pd.DataFrame([assumptions])

c1, c2, c3 = st.columns(3)
with c1:
    st.download_button(
        "Download assumptions (CSV)",
        data=to_csv_bytes(assump_df),
        file_name=f"bp_hotels_assumptions_{zone}_{name[:30].replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    st.download_button(
        "Download projections (detail) (CSV)",
        data=to_csv_bytes(proj),
        file_name=f"bp_hotels_projections_detail_{zone}_{name[:30].replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c3:
    st.download_button(
        "Download scenario summary (CSV)",
        data=to_csv_bytes(agg),
        file_name=f"bp_hotels_projections_summary_{zone}_{name[:30].replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption(
    "Next upgrades: portfolio mode (multiple hotels), market presets per country, and an export-ready PDF for brokers."
)
