"""
education.py — Educational content page: DBSCAN concepts, step-by-step
walkthrough, advantages/limitations, applications, and DBSCAN vs K-Means.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons

from config import CLUSTER_COLORS, NOISE_COLOR, PLOTLY_LAYOUT
from components.styles import inject_custom_css
from components.plot_builder import build_cluster_plot
from components.dbscan_engine import run_dbscan
import plotly.graph_objects as go


def render():
    """Render the educational content page."""
    inject_custom_css()

    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h1 style="margin-bottom:0.2rem;">📚 Learn DBSCAN</h1>
            <p style="color:#94a3b8; margin:0;">
                A comprehensive guide to the DBSCAN algorithm — concepts,
                step-by-step walkthrough, and comparison with K-Means.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Tabs ────────────────────────────────────────────────────
    (
        tab_concepts,
        tab_steps,
        tab_pros_cons,
        tab_apps,
        tab_compare,
    ) = st.tabs([
        "🧩 Key Concepts",
        "🔄 Algorithm Steps",
        "⚖️ Pros & Cons",
        "🌍 Applications",
        "🆚 DBSCAN vs K-Means",
    ])

    # ────────────────────────────────────────────────────────────
    #  Tab 1 — Key Concepts
    # ────────────────────────────────────────────────────────────
    with tab_concepts:
        st.markdown("### The Two Parameters")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                """
                <div class="custom-card" style="border-top:3px solid #4f8ff7;">
                    <h3 style="color:#4f8ff7;">Epsilon (ε)</h3>
                    <p style="color:#94a3b8; font-size:0.92rem;">
                        The <b>maximum distance</b> between two points for them
                        to be considered neighbors. Think of it as drawing a
                        circle of radius ε around every point — any other point
                        inside that circle is a neighbor.
                    </p>
                    <ul style="color:#cbd5e1; font-size:0.88rem;">
                        <li>Too small → most points become noise</li>
                        <li>Too large → distinct clusters merge into one</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                """
                <div class="custom-card" style="border-top:3px solid #a855f7;">
                    <h3 style="color:#a855f7;">MinPts</h3>
                    <p style="color:#94a3b8; font-size:0.92rem;">
                        The <b>minimum number of neighbors</b> (including the
                        point itself) required within the ε-radius for a point
                        to qualify as a <em>core point</em>. It controls how
                        dense a region must be to start a cluster.
                    </p>
                    <ul style="color:#cbd5e1; font-size:0.88rem;">
                        <li>Higher MinPts → stricter clustering, more noise</li>
                        <li>Lower MinPts → more inclusive clusters</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Point Classification")

        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(
                """
                <div class="custom-card" style="border-top:3px solid #22c55e; text-align:center;">
                    <div style="font-size:2.2rem;">●</div>
                    <h3 style="color:#22c55e;">Core Point</h3>
                    <p style="color:#94a3b8; font-size:0.88rem;">
                        Has at least <b>MinPts</b> neighbors within distance ε.
                        Core points form the backbone of clusters.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p2:
            st.markdown(
                """
                <div class="custom-card" style="border-top:3px solid #f97316; text-align:center;">
                    <div style="font-size:2.2rem;">◆</div>
                    <h3 style="color:#f97316;">Border Point</h3>
                    <p style="color:#94a3b8; font-size:0.88rem;">
                        Within ε of a core point but does <b>not</b> have
                        MinPts neighbors itself. Sits on the edge of a cluster.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with p3:
            st.markdown(
                """
                <div class="custom-card" style="border-top:3px solid #6b7280; text-align:center;">
                    <div style="font-size:2.2rem;">✕</div>
                    <h3 style="color:#6b7280;">Noise Point</h3>
                    <p style="color:#94a3b8; font-size:0.88rem;">
                        Not within ε of any core point. These are outliers
                        that do not belong to any cluster.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────
    #  Tab 2 — Algorithm Steps
    # ────────────────────────────────────────────────────────────
    with tab_steps:
        st.markdown("### How DBSCAN Works — Step by Step")
        steps = [
            (
                "1️⃣",
                "Select an unvisited point",
                "The algorithm picks any unvisited point from the dataset at random.",
            ),
            (
                "2️⃣",
                "Query the ε-neighborhood",
                "Find all points within distance ε from the selected point.",
            ),
            (
                "3️⃣",
                "Core point check",
                "If the neighborhood has ≥ MinPts points, the selected point is "
                "a **core point** and a new cluster begins.",
            ),
            (
                "4️⃣",
                "Expand the cluster",
                "Add all ε-neighbors to the cluster. For each new neighbor that "
                "is also a core point, recursively add *its* ε-neighbors too. "
                "This process continues until no more density-reachable points remain.",
            ),
            (
                "5️⃣",
                "Label noise or border",
                "If the selected point has < MinPts neighbors, mark it as **noise** "
                "(for now). It may later be claimed as a **border point** if it falls "
                "within ε of a core point from another cluster.",
            ),
            (
                "6️⃣",
                "Repeat",
                "Move to the next unvisited point and repeat until every point "
                "has been visited.",
            ),
        ]
        for icon, title, desc in steps:
            st.markdown(
                f"""
                <div class="custom-card" style="padding:1rem 1.25rem;">
                    <strong style="font-size:1.05rem;">{icon} {title}</strong>
                    <p style="color:#94a3b8; margin:0.4rem 0 0; font-size:0.9rem;">{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────
    #  Tab 3 — Advantages & Limitations
    # ────────────────────────────────────────────────────────────
    with tab_pros_cons:
        a1, a2 = st.columns(2)
        with a1:
            st.markdown(
                """
                <div class="custom-card" style="border-top:3px solid #22c55e;">
                    <h3 style="color:#22c55e;">✅ Advantages</h3>
                    <ul style="color:#cbd5e1; font-size:0.9rem; line-height:1.8;">
                        <li>Does <b>not</b> require specifying the number of clusters</li>
                        <li>Can find <b>arbitrarily shaped</b> clusters</li>
                        <li>Robust to <b>outliers</b> — labels them as noise</li>
                        <li>Deterministic for core points (order-independent)</li>
                        <li>Works well when clusters have <b>similar density</b></li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with a2:
            st.markdown(
                """
                <div class="custom-card" style="border-top:3px solid #f43f5e;">
                    <h3 style="color:#f43f5e;">⚠️ Limitations</h3>
                    <ul style="color:#cbd5e1; font-size:0.9rem; line-height:1.8;">
                        <li>Struggles with clusters of <b>varying densities</b></li>
                        <li>Sensitive to the <b>choice of ε and MinPts</b></li>
                        <li>Not ideal for <b>very high-dimensional</b> data</li>
                        <li>Border points may be assigned non-deterministically</li>
                        <li>Requires distance metric — not suitable for all data types</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────
    #  Tab 4 — Real-world Applications
    # ────────────────────────────────────────────────────────────
    with tab_apps:
        st.markdown("### Where DBSCAN Is Used")
        apps = [
            ("🗺️", "Geospatial Analysis",
             "Identifying clusters of events (earthquakes, crime hotspots, delivery zones) on maps."),
            ("🖼️", "Image Segmentation",
             "Grouping similar pixels for object detection and scene understanding."),
            ("🌐", "Network Security",
             "Detecting anomalous traffic patterns and intrusion attempts."),
            ("🛒", "Customer Segmentation",
             "Grouping customers by purchasing behavior without pre-defining segment counts."),
            ("🧬", "Bioinformatics",
             "Clustering gene expression data and protein structures."),
            ("🚗", "Autonomous Vehicles",
             "Clustering LiDAR point clouds to detect obstacles and pedestrians."),
        ]
        cols = st.columns(2)
        for i, (icon, title, desc) in enumerate(apps):
            with cols[i % 2]:
                st.markdown(
                    f"""
                    <div class="custom-card">
                        <div style="font-size:1.8rem; margin-bottom:0.3rem;">{icon}</div>
                        <h3>{title}</h3>
                        <p style="color:#94a3b8; font-size:0.9rem; margin:0;">{desc}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ────────────────────────────────────────────────────────────
    #  Tab 5 — DBSCAN vs K-Means (with live plots)
    # ────────────────────────────────────────────────────────────
    with tab_compare:
        st.markdown("### Side-by-Side Comparison")
        st.markdown(
            """
            <div class="info-box">
                Below is a <b>live comparison</b> using a synthetic moons dataset.
                DBSCAN discovers the two crescent shapes while K-Means (k=2)
                splits the data along a linear boundary — failing on non-convex shapes.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Generate moon data for comparison
        X_moon, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
        df_moon = pd.DataFrame(X_moon, columns=["Feature 1", "Feature 2"])

        cmp1, cmp2 = st.columns(2)

        # DBSCAN result
        with cmp1:
            dbscan_result = run_dbscan(X_moon, eps=0.2, min_samples=5, scale=False)
            fig_db = build_cluster_plot(
                df_moon, dbscan_result,
                title="DBSCAN (ε=0.2, MinPts=5)",
            )
            fig_db.update_layout(height=420)
            st.plotly_chart(fig_db, width="stretch")

        # K-Means result
        with cmp2:
            km = KMeans(n_clusters=2, random_state=42, n_init=10)
            km_labels = km.fit_predict(X_moon)

            fig_km = go.Figure()
            for label in sorted(set(km_labels)):
                mask = km_labels == label
                fig_km.add_trace(go.Scatter(
                    x=X_moon[mask, 0],
                    y=X_moon[mask, 1],
                    mode="markers",
                    marker=dict(
                        size=8,
                        color=CLUSTER_COLORS[label % len(CLUSTER_COLORS)],
                        line=dict(width=1, color="white"),
                    ),
                    name=f"Cluster {label} ({int(mask.sum())})",
                    hovertemplate=(
                        f"<b>K-Means Cluster {label}</b><br>"
                        f"Feature 1: %{{x:.3f}}<br>"
                        f"Feature 2: %{{y:.3f}}<extra></extra>"
                    ),
                ))
            # Plot centroids
            fig_km.add_trace(go.Scatter(
                x=km.cluster_centers_[:, 0],
                y=km.cluster_centers_[:, 1],
                mode="markers",
                marker=dict(size=14, color="#eab308", symbol="star", line=dict(width=2, color="white")),
                name="Centroids",
            ))
            layout_km = dict(PLOTLY_LAYOUT)
            layout_km.update(
                title=dict(text="K-Means (k=2)", font=dict(size=18)),
                xaxis_title="Feature 1",
                yaxis_title="Feature 2",
                height=420,
            )
            fig_km.update_layout(**layout_km)
            st.plotly_chart(fig_km, width="stretch")

        # Comparison table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
            | Feature | DBSCAN | K-Means |
            |---------|--------|---------|
            | **Number of clusters** | Discovered automatically | Must be specified (k) |
            | **Cluster shape** | Arbitrary (non-convex) | Convex / spherical only |
            | **Noise handling** | Built-in (labels outliers) | No noise concept |
            | **Scalability** | O(n log n) with spatial index | O(n · k · iterations) |
            | **Parameters** | ε, MinPts | k (number of clusters) |
            | **Determinism** | Core points are deterministic | Depends on initialization |
            | **Varying density** | Struggles | Handles equally |
            """
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
