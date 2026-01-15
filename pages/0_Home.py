import streamlit as st
import pandas as pd

from market_explorer.auth import (
    attempt_login,
    create_user,
    merged_credentials,
    login_required,
    logout,
    signup_enabled,
)
from market_explorer.bp_state import clear_bp_state, list_bp_states,load_bp_state
from market_explorer.notes import load_notes, reset_notes


# -----------------------
# Page config
# -----------------------
st.set_page_config(page_title="Market Explorer – Home", page_icon="🏠", layout="wide")


# -----------------------
# Minimal UI polish
# -----------------------
st.markdown(
    """
<style>
.block-container { padding-top: 2rem; max-width: 1100px; }
.small-muted { color: rgba(0,0,0,.55); font-size: 0.9rem; }
.badge { display:inline-block; padding: 4px 10px; border-radius: 999px;
         background: rgba(255, 0, 128, 0.08); border: 1px solid rgba(255, 0, 128, 0.18);
         font-weight: 600; font-size: 12px; }
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------
# Sidebar navigation (Home is current)
# -----------------------

with st.sidebar:
    st.markdown("### Navigation")
    st.button("🏠 Home", use_container_width=True, disabled=True)
    if st.button("🔎 Market Explorer", use_container_width=True):
        st.switch_page("pages/1_Market_Explorer.py")
    if st.button("🏨 BP Hotels", use_container_width=True):
        st.switch_page("pages/3_Account_Business_Plan_Hotels.py")
    st.divider()

st.title("Market Explorer")
st.caption("Internal market intelligence tool — explore markets, shortlist accounts, export target lists, and build business plans.")

# -----------------------
# Auth gate (robust)
# -----------------------

if not login_required():
    creds = merged_credentials()

    tab_names = ["Sign in"]
    if signup_enabled():
        tab_names.append("Create account")

    tabs = st.tabs(tab_names)

    # ---- Sign in tab
    
    with tabs[0]:
        st.subheader("🔐 Sign in")

        with st.form("signin_form", clear_on_submit=False):
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")
            remember = st.checkbox("Rester connecté", value=True, key="remember_me")
            submitted = st.form_submit_button("Sign in", use_container_width=True)

        if submitted:
            ok, msg = attempt_login(u, p, creds, remember=remember)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    # ---- Create account tab
    
    if signup_enabled() and len(tabs) > 1:
        with tabs[1]:
            st.subheader("🆕 Create account")
            st.caption("Create a new account (MVP).")
            st.caption("Username format: Nom Prénom.")
            st.caption("Password must be at least 10 characters, with upper/lowercase, a number, and a symbol.")

            with st.form("signup_form", clear_on_submit=True):
                nu = st.text_input("New username", key="signup_user", help="Format: Nom Prénom (letters only).")
                npw = st.text_input("New password", type="password", key="signup_pass")
                npw2 = st.text_input("Confirm password", type="password", key="signup_pass2")

                role = st.selectbox(
                    "Role",
                    ["Stagiaire", "Sales", "Ops", "Manager"],
                    index=0,
                    key="signup_role",
                )

                submitted_signup = st.form_submit_button("Create account", use_container_width=True)

            if submitted_signup:
                if npw != npw2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = create_user(nu, npw, role=role, credentials=creds)
                    if ok:
                        st.success(msg)
                        st.info("You can now switch back to the Sign in tab.")
                    else:
                        st.error(msg)

    st.stop()


# -----------------------
# Authenticated Home (simple & clean)
# -----------------------

profile = st.session_state.get("profile", "user").upper()
role = st.session_state.get("role", "—")

h1, h2 = st.columns([4, 1])
with h1:
    st.markdown(
        f"## 🏠 Home — Welcome back, **{profile}**  <span class='badge'>{role}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='small-muted'>Pick up where you left off.</div>", unsafe_allow_html=True)

with h2:
    if st.button("Sign out", use_container_width=True):
        logout()
        st.rerun()

st.divider()

# Notes summary
notes = load_notes(profile) or {}
companies_with_notes = len(notes)

def _note_text(value):
    if isinstance(value, dict):
        return str(value.get("note", "")).strip()
    return str(value).strip()

notes_count = sum(1 for v in notes.values() if _note_text(v))

k1, k2, k3 = st.columns(3)
k1.metric("Companies with notes", companies_with_notes)
k2.metric("Total notes", notes_count)
k3.metric("Status", "Authenticated")

st.divider()

st.subheader("🚀 Quick actions")
a1, a2 = st.columns(2)

with a1:
    st.write("**🔎 Market Explorer**")
    st.caption("Explore markets, filter, shortlist, export target lists.")
    if st.button("Open Market Explorer →", use_container_width=True):
        st.switch_page("pages/1_Market_Explorer.py")

with a2:
    st.write("🏨 Account BP Hotels")
    st.caption("Build assumptions and compute business plan outputs for hotels.")
    if st.button("Open Account BP Hotels ->", use_container_width = True):
        st.switch_page("pages/3_Account_Business_Plan_Hotels.py")

st.divider()

st.subheader("💾 BP en cours")
bp_page_path = "pages/3_Account_Business_Plan_Hotels.py"

bp_entries = list_bp_states(profile)

if bp_entries:
    for entry in bp_entries:
        summary_parts = [f"**BP enregistré :** {entry.get('name', 'Sans nom')}"]
        if entry.get("account"):
            summary_parts.append(f"**Compte :** {entry['account']}")
        if entry.get("updated_at"):
            summary_parts.append(f"**Dernière mise à jour :** {entry['updated_at']}")
        st.info("\n\n".join(summary_parts))
        
        open_col, delete_col = st.columns(2)
        with open_col:
            if st.button("Réouvrir ce BP", key=f"open_bp_{entry['id']}", use_container_width=True):
                st.session_state["bp_active_id"] = entry["id"]

                st.session_state["bp_restore_request"] = True
                st.switch_page(bp_page_path)
        with delete_col:
            if st.button("Effacer le BP enregistré", key=f"delete_bp_{entry['id']}", use_container_width=True):
                clear_bp_state(profile, entry["id"])
                st.success("BP supprimé.")
                st.rerun()
        st.divider()

else:
    st.caption("Aucun BP enregistré pour le moment. Enregistrez-le depuis l'onglet BP correspondant.")

st.divider()

st.subheader("🗒️ Your latest notes")
if notes:
    rows = []
    for key, value in notes.items():
        if isinstance(value, dict):
            rows.append(
                {
                    "Company": value.get("display_name", key),
                    "Zone": value.get("zone", ""),
                    "Market": value.get("market", ""),
                    "Vertical": value.get("vertical", ""),
                    "Tag": value.get("tag", ""),
                    "Note": value.get("note", ""),
                    "LinkedIn": value.get("linkedin_url", ""),
                }
            )
        else:
            rows.append(
                {
                    "Company": key,
                    "Zone": "",
                    "Market": "",
                    "Vertical": "",
                    "Tag": "",
                    "Note": value,
                    "LinkedIn": "",
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("Company", key=lambda s: s.str.lower()).head(12)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "LinkedIn": st.column_config.LinkColumn("LinkedIn", display_text="Open profile"),
        },
    )

    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("Reset my notes", use_container_width=True):
            reset_notes(profile)
            st.success("Notes cleared.")
            st.rerun()
    with colB:
        st.caption("Notes are stored locally (folder `notes/`).")
else:
    st.info("No notes yet. Add notes from the Market Explorer page.")
