import streamlit as st
import pandas as pd
import plotly.express as px

# =============================================================================
# Page config
# =============================================================================
st.set_page_config(
    page_title="Company Business Plan — Hotels (MVP Insurance)",
    layout="wide",
)

st.title("🏨 Hotel Insurance Business Plan (MVP)")
st.caption("Sales-friendly 3-year revenue projection based on a few simple inputs.")

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
# INPUTS — SALES FRIENDLY
# =============================================================================
st.subheader("🔢 Inputs (keep it simple)")

c1, c2, c3 = st.columns(3)

with c1:
    nb_hotels = st.number_input("Number of hotels", min_value=1, value=50, step=5)
    rooms_per_hotel = st.number_input("Average rooms per hotel", min_value=10, value=80, step=10)

with c2:
    price_insurance_per_room_per_year = st.number_input(
        "Insurance price (€/room/year)",
        min_value=1.0,
        value=60.0,
        step=5.0,
        help="Average annual premium per room."
    )
    take_rate = st.slider(
        "Take rate (% of rooms insured)",
        min_value=1,
        max_value=100,
        value=30,
        step=1,
        help="Adoption rate among eligible rooms."
    )

with c3:
    annual_growth = st.slider(
        "Annual growth (%)",
        min_value=-50,
        max_value=200,
        value=20,
        step=5,
        help="Year-over-year growth of the covered base (expansion of hotels/rooms under contract)."
    )
    start_year = st.selectbox("Start year", [2026, 2027, 2028, 2029, 2030], index=0)

st.divider()

# =============================================================================
# CALCULATIONS — TRANSPARENT
# =============================================================================
# Base inventory
total_rooms_y1 = nb_hotels * rooms_per_hotel

# Convert % to multipliers
take_mult = take_rate / 100.0
growth_mult = 1 + (annual_growth / 100.0)

# Projection over 3 years
rows = []
rooms_total = total_rooms_y1

for i in range(3):
    year = start_year + i

    rooms_insured = rooms_total * take_mult
    annual_premium = rooms_insured * price_insurance_per_room_per_year

    rows.append({
        "Year": year,
        "Hotels": nb_hotels,  # MVP: we keep hotels constant; growth drives covered base globally
        "Total Rooms": round(rooms_total),
        "Insured Rooms": round(rooms_insured),
        "Insurance Price (€/room/year)": price_insurance_per_room_per_year,
        "Take Rate (%)": take_rate,
        "Annual Premium (€)": annual_premium,
    })

    rooms_total = rooms_total * growth_mult

df = pd.DataFrame(rows)
df["Annual Premium (€)"] = df["Annual Premium (€)"].round(0).astype(int)

# =============================================================================
# OUTPUT — KPI CARDS
# =============================================================================
st.subheader("💰 Key results")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Year 1 insured rooms", f"{int(df.loc[0, 'Insured Rooms']):,}")
k2.metric("Year 1 annual premium", f"{int(df.loc[0, 'Annual Premium (€)']):,} €")
k3.metric("Year 3 insured rooms", f"{int(df.loc[2, 'Insured Rooms']):,}")
k4.metric("Year 3 annual premium", f"{int(df.loc[2, 'Annual Premium (€)']):,} €")

st.divider()

# =============================================================================
# TABLE + PLOT
# =============================================================================
c_left, c_right = st.columns([1.2, 1])

with c_left:
    st.subheader("📋 3-year plan")
    st.dataframe(
        df[["Year", "Total Rooms", "Insured Rooms", "Annual Premium (€)"]],
        use_container_width=True,
        hide_index=True
    )

with c_right:
    st.subheader("📈 Premium over time")
    fig = px.line(
        df,
        x="Year",
        y="Annual Premium (€)",
        markers=True,
        title="Annual Premium (3-year)"
    )
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# =============================================================================
# SALES PITCH — 1 paragraph
# =============================================================================
y1 = df.loc[0, "Annual Premium (€)"]
y3 = df.loc[2, "Annual Premium (€)"]
rooms = int(total_rooms_y1)

st.subheader("🗣️ Talk track (copy/paste)")
st.markdown(
    f"""
> *On a base of **{nb_hotels} hotels** (~**{rooms:,} rooms**), with a **{take_rate}% take rate**  
> and an average premium of **{price_insurance_per_room_per_year:.0f} €/room/year**,  
> we generate about **{y1:,} €** premium in Year 1.  
> With **{annual_growth}% annual growth**, this reaches about **{y3:,} €** by Year 3.*
"""
)
