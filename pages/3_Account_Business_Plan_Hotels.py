import streamlit as st
import pandas as pd
import plotly.express as px

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Business Plan Hotels — Groupe",
    layout="wide",
)

st.title("🏨 Business Plan Hotels — Groupe (MVP)")
st.caption("MVP sales-friendly pour groupes multi-marques. Calculs simples, robustes et transparents.")

st.divider()

# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    st.markdown("### Navigation")
    st.button("🏠 Home", use_container_width=True, disabled=True)
    if st.button("🔎 Market Explorer", use_container_width=True):
        st.switch_page("pages/1_Market_Explorer.py")
    if st.button("📈 BP Hôtellerie", use_container_width=True):
        st.switch_page("pages/2_Company_Business_Plan.py")
    if st.button("🏨 Account BP Hotels", use_container_width=True):
        st.switch_page("pages/3_Account_Business_Plan_Hotels.py")
    st.divider()

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
    max_value=20,
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

    if nb_hotels == 0:
        warnings.append(f"{brand_name}: nombre d'hôtels = 0 → CA/hôtel à 0.")

    if nb_chambres <= 0:
        warnings.append(f"{brand_name}: chambres moyennes ≤ 0 → nuitées/ADR à 0.")

    if taux_remplissage < 0 or taux_remplissage > 100:
        warnings.append(f"{brand_name}: taux de remplissage hors [0,100].")

    if duree_sejour < 1:
        warnings.append(f"{brand_name}: durée séjour < 1 nuit.")

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
