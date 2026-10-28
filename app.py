"""
app.py — Main entry point for the DBSCAN Algorithm Visualizer.
Configures the Streamlit page and routes to sub-pages.
"""

import streamlit as st

# ── Page Config (must be the first Streamlit command) ───────────
st.set_page_config(
    page_title="DBSCAN Algorithm Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from views.dashboard import render as render_dashboard
from views.visualizer import render as render_visualizer
from views.education import render as render_education

# ── Navigation ──────────────────────────────────────────────────
PAGES = {
    "Dashboard": render_dashboard,
    "Visualizer": render_visualizer,
    "Education": render_education,
}

# Initialize session state
if "page" not in st.session_state:
    st.session_state["page"] = "Dashboard"

# Sidebar navigation
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:1rem 0 0.5rem;">
            <span style="font-size:2rem;">🔬</span>
            <h2 style="margin:0.3rem 0 0; font-size:1.15rem !important;
                background: linear-gradient(135deg, #4f8ff7, #a855f7);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                DBSCAN Visualizer
            </h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    selected = st.radio(
        "Navigate",
        list(PAGES.keys()),
        index=list(PAGES.keys()).index(st.session_state["page"]),
        key="nav_radio",
        label_visibility="collapsed",
    )
    if selected != st.session_state["page"]:
        st.session_state["page"] = selected
        st.rerun()

# ── Render Selected Page ────────────────────────────────────────
PAGES[st.session_state["page"]]()
