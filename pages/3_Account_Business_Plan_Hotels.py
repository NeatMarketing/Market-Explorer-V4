from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from market_explorer.auth import require_auth
from market_explorer.tiering import add_tier, filter_by_tier
from market_explorer.bp_state import get_bp_state, load_bp_state, save_bp_state

# =============================================================================
# Auth guard
# =============================================================================

require_auth()
profile = st.session_state.get("profile")

# =============================================================================
# Page config
# =============================================================================

st.set_page_config(
    page_title="Business Plan Hotels — Groupe",
    layout="wide",
)

st.title("Business Plan Hotels — Groupe (MVP)")

profile = st.session_state.get("profile", "user")

bp_context = st.session_state.get("bp_context") if isinstance(st.session_state.get("bp_context"), dict) else {}
bp_context_company_name = bp_context.get("company_name")
bp_context_company_id = bp_context.get("company_id")

if bp_context_company_name:
    banner_col, banner_action = st.columns([4, 1])
    with banner_col:
        st.info(f"Entreprise sélectionnée depuis Market Explorer: **{bp_context_company_name}**")
    with banner_action:
        if st.button("Changer", use_container_width=True):
            st.session_state.pop("bp_context", None)
            st.session_state.pop("bp_context_key", None)
            st.session_state.pop("bp_context_filters_key", None)
            st.session_state.pop("selected_hotel_name", None)
            st.rerun()
            
bp_store = load_bp_state(profile) if profile else {"bps": {}, "last_opened_id": None}
active_bp_id = st.session_state.get("bp_active_id") or bp_store.get("last_opened_id")
bp_state = get_bp_state(profile, active_bp_id) if profile and active_bp_id else {}
if bp_state is None:
    bp_state = {}
if st.session_state.pop("bp_restore_request", False) and active_bp_id:
    restored = get_bp_state(profile, active_bp_id)
    if restored:

        for key, value in (restored.get("session_state") or {}).items():
            st.session_state[key] = value

        st.session_state["bp_active_id"] = active_bp_id
        bp_state = restored
        st.success("BP rechargé.")
    else:
        bp_state = {}

# =============================================================================
# Sidebar
# =============================================================================

DATA_DIR = Path(__file__).resolve().parents[1] / "Data_Clean"

TIER_ORDER = ["All", "Tier 1", "Tier 2", "Tier 3"]
TIER_UI = {
    "All": "All Markets",
    "Tier 1": "Large Market",
    "Tier 2": "Mid-Market",
    "Tier 3": "Low-Market",
}
TIER_UI_INV = {v: k for k, v in TIER_UI.items()}

ZONE_LABELS = {
    "france": "France",
    "eu": "Europe",
}
ZONE_DATASET_MAP = {
    "france": "france",
    "eu": "europe",
}

if bp_context_company_name:
    context_key = (bp_context_company_id, bp_context_company_name)
    if st.session_state.get("bp_context_filters_key") != context_key:
        context_zone = bp_context.get("zone")
        if context_zone in ZONE_LABELS:
            st.session_state["hotel_zone"] = context_zone
        context_tier = bp_context.get("tiering")
        if context_tier in TIER_ORDER:
            st.session_state["hotel_tier_filter"] = context_tier
        st.session_state["bp_context_filters_key"] = context_key

def format_zone_option(value: str) -> str:
    return ZONE_LABELS.get(str(value).strip().lower(), str(value))

@st.cache_data(show_spinner=False)

def load_hotels(zone_key: str) -> pd.DataFrame:
    dataset_key = ZONE_DATASET_MAP.get(str(zone_key).strip().lower(), zone_key)
    dataset_path = DATA_DIR / f"travel_hotel_{dataset_key}_cleaned.csv"
    if not dataset_path.exists():
        return pd.DataFrame()
    return pd.read_csv(dataset_path)
    
with st.sidebar:
    
    st.markdown("### Navigation")
    if st.button("🏠 Home", use_container_width=True):
        st.switch_page("pages/0_Home.py")
    if st.button("🔎 Market Explorer", use_container_width=True):
        st.switch_page("pages/1_Market_Explorer.py")
    st.button("🏨 BP Hotels", use_container_width=True, disabled=True)
    st.divider()
    
    st.header("Sélection hôtel")
    zone = st.selectbox(
        "Zone",
        ["france", "eu"],
        key="hotel_zone",
        format_func=format_zone_option,
    )

    hotels_df = load_hotels(zone)
    if hotels_df.empty:
        st.warning("Aucun dataset hôtel disponible pour cette zone.")
    else:
        t1 = st.number_input(
            "Large Market threshold (M$)",
            min_value=0.0,
            value=500.0,
            step=50.0,
            key="hotel_tier_t1",
        )
        t2 = st.number_input(
            "Mid-Market threshold (M$)",
            min_value=0.0,
            value=100.0,
            step=25.0,
            key="hotel_tier_t2",
        )

        if t1 < t2:
            st.warning("Large Market threshold should be ≥ Mid-Market threshold. Adjusting automatically.")
            t1, t2 = t2, t1

        tier_ui_options = [TIER_UI[t] for t in TIER_ORDER]
        default_internal = st.session_state.get("hotel_tier_filter", "All")
        default_ui = TIER_UI.get(default_internal, TIER_UI["All"])
        default_index = tier_ui_options.index(default_ui) if default_ui in tier_ui_options else 0

        tier_ui = st.selectbox("Market Tier", tier_ui_options, index=default_index)
        tier_filter = TIER_UI_INV[tier_ui]
        st.session_state["hotel_tier_filter"] = tier_filter

        hotels_tiered = add_tier(hotels_df, t1=t1, t2=t2, revenue_col="Revenue_M")
        hotels_filtered = filter_by_tier(hotels_tiered, tier_filter)
        hotel_names = sorted(hotels_filtered["Name"].dropna().unique().tolist())

        if not hotel_names:
            st.warning("Aucun hôtel disponible pour ce tiering.")
        else:
            context_match = None
            if bp_context_company_name:
                normalized_map = {str(name).strip().lower(): name for name in hotel_names}
                context_match = normalized_map.get(str(bp_context_company_name).strip().lower())
                context_key = (bp_context_company_id, str(bp_context_company_name))
                if st.session_state.get("bp_context_key") != context_key:
                    st.session_state["bp_context_key"] = context_key
                    if context_match:
                        st.session_state["selected_hotel_name"] = context_match

            if bp_context_company_name and not context_match:
                st.warning(
                    "L'entreprise sélectionnée depuis Market Explorer n'est pas disponible avec les filtres actuels."
                )
            selected_hotel = st.selectbox(
                "Hôtel à étudier",
                hotel_names,
                key="selected_hotel_name",
            )
            selected_row = hotels_filtered.loc[hotels_filtered["Name"] == selected_hotel].head(1)
            if not selected_row.empty:
                revenue_value = selected_row["Revenue"].iloc[0]
                if pd.notna(revenue_value):
                    st.session_state["ca_total"] = float(revenue_value)
                hq_location = selected_row.get("HQ Location", pd.Series([""])).iloc[0]
                st.caption(f"Siège: {hq_location}" if hq_location else "Siège: n/a")
        if st.button("🔄 Réinitialiser la sélection", use_container_width=True):
            st.session_state.pop("bp_context", None)
            st.session_state.pop("bp_context_key", None)
            st.session_state.pop("bp_context_filters_key",None)
            st.session_state.pop("selected_hotel_name", None)
            st.rerun()

# =============================================================================
# Helpers
# =============================================================================

def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def normalize_shares(raw_shares: list[float]) -> list[float]:
    total = sum(raw_shares)
    if total == 0:
        if not raw_shares:
            return []
        return [100.0 / len(raw_shares)] * len(raw_shares)
    return [(share / total) * 100.0 for share in raw_shares]


def fmt_money(value: float) -> str:
    return f"{value:,.0f} €".replace(",", " ")

SAVE_STATE_PREFIXES = ("bp_group_",)
SAVE_STATE_KEYS = {
    "hotel_zone",
    "hotel_tier_t1",
    "hotel_tier_t2",
    "hotel_tier_filter",
    "selected_hotel_name",
    "ca_total",
    "CA_total",
}


def capture_bp_session_state() -> dict:
    session_snapshot = {
        key: value
        for key, value in st.session_state.items()
        if key.startswith(SAVE_STATE_PREFIXES) or key in SAVE_STATE_KEYS
    }
    return session_snapshot

# =============================================================================
# Inputs — Type d'entreprise
# =============================================================================

st.subheader("🏢 Type d'entreprise")
company_type = st.selectbox("Type d'entreprise", ["Groupe"], index=0)

st.divider()

# =============================================================================
# CA total (reuse if available)
# =============================================================================

ca_total_state = st.session_state.get("ca_total")
if ca_total_state is None:
    ca_total_state = st.session_state.get("CA_total")

if ca_total_state is None:
    ca_total = st.number_input(
        "CA total groupe (€)",
        min_value=0.0,
        value=0.0,
        step=100000.0,
        help="Utilisé si aucun CA n'est disponible depuis Market Explorer.",
        key="bp_group_ca_total_manual",
    )
else:
    ca_total = float(ca_total_state)
    st.metric("CA total groupe (Market Explorer)", fmt_money(ca_total))

if ca_total <= 0:
    st.warning("CA total nul ou manquant : les calculs seront à 0.")

st.divider()

# =============================================================================
# Marques du groupe
# =============================================================================

st.subheader("🏷️ Marques du groupe")

nb_marques = st.number_input(
    "Nombre de marques",
    min_value=1,
    max_value=50,
    value=int(st.session_state.get("bp_group_nb_marques", 3)),
    step=1,
    key="bp_group_nb_marques",
)

brand_inputs = []
raw_shares = []

for idx in range(1, nb_marques + 1):
    st.markdown(f"**Marque {idx}**")
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        name_key = f"bp_group_brand_name_{idx}"
        brand_name = st.text_input(
            "Nom de la marque",
            value=st.session_state.get(name_key, f"Marque {idx}"),
            key=name_key,
        )

    with c2:
        hotels_key = f"bp_group_brand_hotels_{idx}"
        nb_hotels = st.number_input(
            "Nb hôtels",
            min_value=0,
            value=int(st.session_state.get(hotels_key, 0)),
            step=1,
            key=hotels_key,
        )

    with c3:
        share_key = f"bp_group_brand_share_{idx}"
        part_ca = st.number_input(
            "Part CA (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get(share_key, 0.0)),
            step=1.0,
            key=share_key,
        )

    brand_inputs.append({
        "index": idx,
        "name": brand_name.strip() or f"Marque {idx}",
        "nb_hotels": int(nb_hotels),
        "part_ca": float(part_ca),
    })
    raw_shares.append(float(part_ca))

normalized_shares = normalize_shares(raw_shares)

if raw_shares and abs(sum(raw_shares) - 100.0) > 0.01:
    st.info("Les parts de CA ont été normalisées automatiquement pour totaliser 100%.")

st.divider()

# =============================================================================
# Revenus hébergement
# =============================================================================

st.subheader("🏨 Revenus hébergement")

pct_ca_hebergement = st.number_input(
    "% CA hébergement (global)",
    min_value=0.0,
    max_value=100.0,
    value=float(st.session_state.get("bp_group_pct_hebergement", 70.0)),
    step=1.0,
    key="bp_group_pct_hebergement",
)

override_hebergement = st.checkbox("Différencier par marque", value=False, key="bp_group_heberg_override")

per_brand_hebergement = {}
if override_hebergement:
    for brand in brand_inputs:
        key = f"bp_group_pct_hebergement_{brand['index']}"
        per_brand_hebergement[brand["index"]] = st.number_input(
            f"% hébergement — {brand['name']}",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get(key, pct_ca_hebergement)),
            step=1.0,
            key=key,
        )

st.divider()

# =============================================================================
# Hypothèses hôtelières
# =============================================================================

st.subheader("🛏️ Hypothèses hôtelières")

c1, c2, c3 = st.columns(3)

with c1:
    nb_chambres_moyen = st.number_input(
        "Chambres moyennes / hôtel (global)",
        min_value=0,
        value=int(st.session_state.get("bp_group_nb_chambres", 80)),
        step=5,
        key="bp_group_nb_chambres",
    )

with c2:
    taux_remplissage_annuel = st.number_input(
        "Taux de remplissage annuel % (global)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("bp_group_taux_remplissage", 65.0)),
        step=1.0,
        key="bp_group_taux_remplissage",
    )

with c3:
    duree_moyenne_sejour = st.number_input(
        "Durée moyenne séjour (nuits) (global)",
        min_value=1.0,
        value=float(st.session_state.get("bp_group_duree_sejour", 2.0)),
        step=0.5,
        key="bp_group_duree_sejour",
    )

override_hotellerie = st.checkbox("Différencier les hypothèses par marque", value=False, key="bp_group_hotellerie_override")

per_brand_hotel = {}
if override_hotellerie:
    for brand in brand_inputs:
        st.markdown(f"**Hypothèses — {brand['name']}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            key = f"bp_group_nb_chambres_{brand['index']}"
            nb_chambres = st.number_input(
                "Chambres moyennes / hôtel",
                min_value=0,
                value=int(st.session_state.get(key, nb_chambres_moyen)),
                step=5,
                key=key,
            )
        with c2:
            key = f"bp_group_taux_remplissage_{brand['index']}"
            taux = st.number_input(
                "Taux de remplissage annuel %",
                min_value=0.0,
                max_value=100.0,
                value=float(st.session_state.get(key, taux_remplissage_annuel)),
                step=1.0,
                key=key,
            )
        with c3:
            key = f"bp_group_duree_sejour_{brand['index']}"
            duree = st.number_input(
                "Durée moyenne séjour (nuits)",
                min_value=1.0,
                value=float(st.session_state.get(key, duree_moyenne_sejour)),
                step=0.5,
                key=key,
            )
        per_brand_hotel[brand["index"]] = {
            "nb_chambres": int(nb_chambres),
            "taux_remplissage": float(taux),
            "duree_sejour": float(duree),
        }

st.divider()

# =============================================================================
# Assurance
# =============================================================================

st.subheader("🛡️ Assurance")

c1, c2 = st.columns(2)
with c1:
    taux_prime_assurance = st.number_input(
        "Taux prime assurance (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("bp_group_taux_prime", 6.0)),
        step=0.5,
        key="bp_group_taux_prime",
    )
with c2:
    taux_de_prise = st.number_input(
        "Taux de prise (%)",
        min_value=0.0,
        max_value=100.0,
        value=float(st.session_state.get("bp_group_taux_prise", 25.0)),
        step=1.0,
        key="bp_group_taux_prise",
    )

st.divider()

# =============================================================================
# Calculations
# =============================================================================

warnings = []
rows = []

for brand, share_norm in zip(brand_inputs, normalized_shares):
    index = brand["index"]
    brand_name = brand["name"]
    nb_hotels = brand["nb_hotels"]

    pct_hebergement = per_brand_hebergement.get(index, pct_ca_hebergement)

    if override_hotellerie and index in per_brand_hotel:
        hotel_params = per_brand_hotel[index]
        nb_chambres = hotel_params["nb_chambres"]
        taux_remplissage = hotel_params["taux_remplissage"]
        duree_sejour = hotel_params["duree_sejour"]
    else:
        nb_chambres = int(nb_chambres_moyen)
        taux_remplissage = float(taux_remplissage_annuel)
        duree_sejour = float(duree_moyenne_sejour)

    ca_marque = ca_total * (share_norm / 100.0)
    ca_hotel_total = safe_div(ca_marque, nb_hotels) if nb_hotels > 0 else 0.0
    
    #if nb_hotels == 0:
        #warnings.append(f"{brand_name}: nombre d'hôtels = 0 → CA/hôtel à 0.")

    #if nb_chambres <= 0:
        #warnings.append(f"{brand_name}: chambres moyennes ≤ 0 → nuitées/ADR à 0.")

    #if taux_remplissage < 0 or taux_remplissage > 100:
        #warnings.append(f"{brand_name}: taux de remplissage hors [0,100].")

    #if duree_sejour < 1:
        #warnings.append(f"{brand_name}: durée séjour < 1 nuit.")
    
    ca_hebergement_hotel = ca_hotel_total * (pct_hebergement / 100.0)

    nuitees = max(nb_chambres, 0) * 365 * (max(taux_remplissage, 0) / 100.0)
    adr = safe_div(ca_hebergement_hotel, nuitees) if nuitees > 0 else 0.0

    if nuitees == 0:
        warnings.append(f"{brand_name}: nuitées = 0 → ADR à 0.")

    if adr < 20 or adr > 1500:
        warnings.append(f"{brand_name}: ADR atypique ({adr:,.0f} €).")

    prix_sejour = adr * duree_sejour
    nb_sejours = safe_div(nuitees, duree_sejour) if duree_sejour > 0 else 0.0

    prime_sejour = prix_sejour * (taux_prime_assurance / 100.0)
    prime_brute_annuelle = nb_sejours * prime_sejour * (taux_de_prise / 100.0)

    rows.append({
        "Marque": brand_name,
        "Nb hôtels": nb_hotels,
        "Part CA": share_norm,
        "CA marque": ca_marque,
        "CA/hôtel total": ca_hotel_total,
        "% hébergement": pct_hebergement,
        "CA hébergement/hôtel": ca_hebergement_hotel,
        "Chambres moy.": nb_chambres,
        "Occup. %": taux_remplissage,
        "Nuitées/an": nuitees,
        "ADR": adr,
        "Durée séjour": duree_sejour,
        "Prix séjour": prix_sejour,
        "Nb séjours/an": nb_sejours,
        "Prime %": taux_prime_assurance,
        "Prise %": taux_de_prise,
        "Prime brute/an": prime_brute_annuelle,
    })

if warnings:
    for msg in warnings:
        st.warning(msg)

st.subheader("📊 Résultats groupe")

result_df = pd.DataFrame(rows)

sum_primes = float(result_df["Prime brute/an"].sum()) if not result_df.empty else 0.0
sum_nuitees = float(result_df["Nuitées/an"].sum()) if not result_df.empty else 0.0
sum_ca_hebergement = float(result_df["CA hébergement/hôtel"].sum()) if not result_df.empty else 0.0

adr_moyen_pondere = safe_div(sum_ca_hebergement, sum_nuitees) if sum_nuitees > 0 else 0.0

k1, k2, k3 = st.columns(3)

k1.metric("CA total groupe", fmt_money(ca_total))
k2.metric("Prime brute annuelle", fmt_money(sum_primes))
k3.metric("ADR moyen pondéré", f"{adr_moyen_pondere:,.0f} €".replace(",", " "))

st.divider()

# =============================================================================
# Table détaillée
# =============================================================================

st.subheader("📋 Détail par marque")

display_df = result_df.copy()

if not display_df.empty:
    numeric_cols = [
        "CA marque",
        "CA/hôtel total",
        "CA hébergement/hôtel",
        "Nuitées/an",
        "ADR",
        "Prix séjour",
        "Nb séjours/an",
        "Prime brute/an",
    ]
    for col in numeric_cols:
        display_df[col] = display_df[col].round(0)

    st.dataframe(display_df, use_container_width=True, hide_index=True)
else:
    st.info("Aucune marque à afficher.")

st.divider()

# =============================================================================
# Graph
# =============================================================================

st.subheader("📈 Prime brute annuelle par marque")

if not result_df.empty:
    fig = px.bar(
        result_df,
        x="Marque",
        y="Prime brute/an",
        title="Prime brute annuelle par marque",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Graphique indisponible (pas de données).")

st.divider()

st.subheader("💾 Enregistrer le BP en cours")

if bp_state.get("name"):

    summary_parts = [f"**BP en cours :** {bp_state['name']}"]

    if bp_state.get("account"):
   
        summary_parts.append(f"**Compte :** {bp_state['account']}")
       
    if bp_state.get("updated_at"):
            
        summary_parts.append(f"**Dernière mise à jour :** {bp_state['updated_at']}")
   
    st.info("\n\n".join(summary_parts))
else:
    st.caption("Aucun BP enregistré. Utilisez le formulaire ci-dessous pour en sauvegarder un.")

with st.form("bp_state_form_account"):
    bp_name = st.text_input("Nom du BP", value=bp_state.get("name", ""))
    bp_account = st.text_input(
        "Compte / groupe / hôtel (optionnel)",
        value=bp_state.get("account", ""),
    )
    update_current = st.checkbox(
        "Mettre à jour le BP en cours",
        value=False,
        disabled=not active_bp_id,
        help="Laissez décoché pour enregistrer un nouveau BP.",
    )
    submitted_bp = st.form_submit_button("Enregistrer ce BP", use_container_width=True)

if submitted_bp:
    if not bp_name.strip():
        st.error("Merci de renseigner un nom de BP.")
    else:
        session_snapshot = capture_bp_session_state()
        target_bp_id = active_bp_id if update_current else None
        bp_state = save_bp_state(
            profile,
            {
                "name": bp_name.strip(),
                "account": bp_account.strip(),
                "session_state": session_snapshot,
            },
            bp_id=target_bp_id,
        )
        st.session_state["bp_active_id"] = bp_state["id"]
        st.success("BP mis à jour." if update_current else "BP enregistré.")