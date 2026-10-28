"""
dashboard.py — Landing / hero page for the DBSCAN Algorithm Visualizer.
"""

import streamlit as st
from components.styles import inject_custom_css


def render():
    """Render the dashboard / landing page."""
    inject_custom_css()

    # ── Hero Section ────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 2.5rem 1rem 1.5rem;">
            <span class="badge badge-purple" style="margin-bottom:0.75rem; display:inline-block;">
                Advanced Machine Learning Practical
            </span>
            <h1 style="font-size:2.8rem !important; margin-bottom:0.5rem;
                background: linear-gradient(135deg, #4f8ff7, #a855f7, #22c55e);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                DBSCAN Algorithm Visualizer
            </h1>
            <p style="color:#94a3b8; font-size:1.1rem; max-width:700px; margin:0 auto 2rem;">
                Explore density-based spatial clustering interactively.
                Generate datasets, tune parameters, and watch DBSCAN discover
                clusters of arbitrary shape — powered by real scikit-learn computations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Navigation Buttons ──────────────────────────────────────
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("🔬  Open Visualizer", use_container_width=True, key="nav_viz"):
            st.session_state["page"] = "Visualizer"
            st.rerun()
    with col2:
        if st.button("📚  Learn DBSCAN", use_container_width=True, key="nav_edu"):
            st.session_state["page"] = "Education"
            st.rerun()
    with col3:
        if st.button("🔄  Refresh", use_container_width=True, key="nav_refresh"):
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Cards ───────────────────────────────────────────
    cards = [
        {
            "icon": "📊",
            "title": "Interactive Visualization",
            "desc": (
                "Generate moons, circles, blobs, and custom datasets. "
                "Watch DBSCAN cluster them in real time with color-coded "
                "core, border, and noise points."
            ),
            "color": "#4f8ff7",
        },
        {
            "icon": "⚙️",
            "title": "Live Parameter Tuning",
            "desc": (
                "Adjust Epsilon (ε) and MinPts with sliders and instantly "
                "see how clustering results change. Understand parameter "
                "sensitivity through real experimentation."
            ),
            "color": "#a855f7",
        },
        {
            "icon": "🧠",
            "title": "Educational Content",
            "desc": (
                "Step-by-step algorithm walkthrough, point classification, "
                "advantages, limitations, real-world applications, and a "
                "side-by-side DBSCAN vs K-Means comparison."
            ),
            "color": "#22c55e",
        },
        {
            "icon": "📈",
            "title": "Real ML Metrics",
            "desc": (
                "Cluster counts, noise percentages, core-point statistics, "
                "cluster-size breakdowns, and silhouette scores — all "
                "computed from actual scikit-learn DBSCAN runs."
            ),
            "color": "#f97316",
        },
    ]

    c1, c2 = st.columns(2)
    for i, card in enumerate(cards):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(
                f"""
                <div class="custom-card" style="border-top: 3px solid {card['color']};">
                    <div style="font-size:2rem; margin-bottom:0.5rem;">{card['icon']}</div>
                    <h3 style="margin:0 0 0.5rem;">{card['title']}</h3>
                    <p style="color:#94a3b8; font-size:0.92rem; margin:0;">{card['desc']}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── What is DBSCAN? ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="info-box">
            <strong>What is DBSCAN?</strong><br>
            DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
            is an unsupervised machine learning algorithm that groups together
            points that are closely packed in space, marking low-density points
            as noise. Unlike K-Means, DBSCAN does <em>not</em> require
            specifying the number of clusters in advance and can discover
            clusters of arbitrary shape.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Footer ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#64748b; font-size:0.82rem; padding:0.5rem 0 1rem;">
            Built with Streamlit · scikit-learn · Plotly &nbsp;|&nbsp;
            Advanced Learner Practical — Machine Learning
        </div>
        """,
        unsafe_allow_html=True,
    )
