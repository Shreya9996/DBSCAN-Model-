"""
visualizer.py — Main DBSCAN visualizer page.
Combines dataset controls, DBSCAN parameter sliders, interactive plot,
neighborhood explorer, and results panel into one cohesive layout.
"""

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DEFAULT_EPS,
    DEFAULT_MIN_PTS,
    DEFAULT_N_SAMPLES,
    MIN_SAMPLES,
    MAX_SAMPLES,
)
from components.styles import inject_custom_css, metric_card
from components.dataset_generator import generate_dataset, parse_uploaded_csv
from components.dbscan_engine import run_dbscan, find_neighbors
from components.plot_builder import (
    build_cluster_plot,
    build_raw_scatter,
    build_neighborhood_plot,
)


def render():
    """Render the full DBSCAN visualizer page."""
    inject_custom_css()

    # ── Header ──────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom:1rem;">
            <h1 style="margin-bottom:0.2rem;">🔬 DBSCAN Visualizer</h1>
            <p style="color:#94a3b8; margin:0;">
                Generate or upload data, adjust parameters, and explore DBSCAN results interactively.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ══════════════════════════════════════════════════════════════
    #  SIDEBAR — Dataset & Parameters
    # ══════════════════════════════════════════════════════════════
    with st.sidebar:
        st.markdown("### 📂 Dataset")

        data_source = st.radio(
            "Data source",
            ["Synthetic", "Upload CSV"],
            horizontal=True,
            key="data_source",
        )

        df = None
        x_col = "Feature 1"
        y_col = "Feature 2"

        if data_source == "Synthetic":
            dataset_type = st.selectbox(
                "Dataset shape",
                ["Moons", "Circles", "Blobs", "Anisotropic", "Varied Density", "Random"],
                key="ds_type",
            )
            n_samples = st.slider(
                "Number of samples",
                min_value=MIN_SAMPLES,
                max_value=MAX_SAMPLES,
                value=DEFAULT_N_SAMPLES,
                step=50,
                key="n_samples",
            )
            noise_level = st.slider(
                "Dataset noise",
                min_value=0.01,
                max_value=0.30,
                value=0.05,
                step=0.01,
                key="ds_noise",
            )
            random_seed = st.number_input(
                "Random seed", value=42, min_value=0, max_value=9999, step=1, key="seed"
            )
            df = generate_dataset(dataset_type, n_samples, noise_level, int(random_seed))

        else:  # Upload CSV
            uploaded = st.file_uploader(
                "Upload a CSV file", type=["csv"], key="csv_upload"
            )
            if uploaded is not None:
                df_parsed, err = parse_uploaded_csv(uploaded)
                if err:
                    st.error(f"❌ {err}")
                else:
                    df = df_parsed
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) < 2:
                        st.error(
                            f"❌ Need at least 2 numeric columns, found {len(numeric_cols)}."
                        )
                        df = None
                    else:
                        x_col = st.selectbox("X axis feature", numeric_cols, index=0, key="x_feat")
                        y_idx = min(1, len(numeric_cols) - 1)
                        y_col = st.selectbox("Y axis feature", numeric_cols, index=y_idx, key="y_feat")
                        if x_col == y_col:
                            st.warning("⚠️ X and Y features are the same. Select different columns for a meaningful plot.")
            else:
                st.info("Upload a CSV file to get started.")

        st.markdown("---")
        st.markdown("### ⚙️ DBSCAN Parameters")

        eps = st.slider(
            "Epsilon (ε) — neighborhood radius",
            min_value=0.05,
            max_value=5.0,
            value=DEFAULT_EPS,
            step=0.05,
            key="eps",
            help="Maximum distance between two points to be considered neighbors.",
        )
        min_pts = st.slider(
            "MinPts — minimum neighbors",
            min_value=2,
            max_value=50,
            value=DEFAULT_MIN_PTS,
            step=1,
            key="min_pts",
            help="Minimum number of points required to form a dense region (core point).",
        )

        scale_data = st.checkbox("Standardize features", value=True, key="scale")

        show_epsilon = st.checkbox(
            "Show ε-radius circles", value=False, key="show_eps",
            help="Draw epsilon circles around a few core points to visualize the neighborhood."
        )

        # ── Reset Parameters Button ────────────────────────────
        st.markdown("---")
        if st.button("🔄 Reset Parameters", use_container_width=True, key="reset_params"):
            st.session_state["eps"] = DEFAULT_EPS
            st.session_state["min_pts"] = DEFAULT_MIN_PTS
            st.session_state["scale"] = True
            st.session_state["show_eps"] = False
            st.session_state["ds_type"] = "Moons"
            st.session_state["n_samples"] = DEFAULT_N_SAMPLES
            st.session_state["ds_noise"] = 0.05
            st.session_state["seed"] = 42
            st.rerun()

    # ══════════════════════════════════════════════════════════════
    #  MAIN AREA
    # ══════════════════════════════════════════════════════════════
    if df is None:
        st.warning("⬅️ Choose a dataset from the sidebar to begin.")
        return

    if df.empty:
        st.error("❌ The dataset is empty. Please select a different data source or upload a valid CSV.")
        return

    # Validate columns exist
    if x_col not in df.columns or y_col not in df.columns:
        st.error(f"❌ Selected columns '{x_col}' or '{y_col}' not found in the dataset.")
        return

    # Check for sufficient data
    X = df[[x_col, y_col]].values
    if len(X) < 2:
        st.error("❌ Need at least 2 data points for clustering.")
        return

    # Run DBSCAN
    result = run_dbscan(X, eps=eps, min_samples=min_pts, scale=scale_data)

    # ── Tabs: Clustered View | Raw Data | Neighborhood | Data Table ──
    tab_cluster, tab_raw, tab_neighbor, tab_data = st.tabs(
        ["📊 Clustered View", "🔵 Raw Data", "🔍 Neighborhood Explorer", "📋 Data Table"]
    )

    with tab_cluster:
        fig = build_cluster_plot(
            df, result,
            x_col=x_col, y_col=y_col,
            show_epsilon=show_epsilon,
            eps_value=eps,
            title="DBSCAN Clustering Result",
        )
        st.plotly_chart(fig, width="stretch")

        # Legend explanation
        st.markdown(
            """
            <div class="info-box" style="font-size:0.88rem;">
                <strong>Reading the chart:</strong> &nbsp;
                <b>● Circle</b> = Core point &nbsp;|&nbsp;
                <b>◆ Diamond</b> = Border point &nbsp;|&nbsp;
                <b>✕ Cross</b> = Noise point &nbsp;|&nbsp;
                Each color represents a distinct cluster.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_raw:
        fig_raw = build_raw_scatter(df, x_col, y_col, title="Raw Dataset (before clustering)")
        st.plotly_chart(fig_raw, width="stretch")

    # ── Neighborhood Explorer ───────────────────────────────────
    with tab_neighbor:
        st.markdown(
            """
            <div class="info-box">
                <strong>🔍 Neighborhood Explorer:</strong>
                Select a point by its index to see its ε-neighborhood circle
                and all neighbors highlighted in yellow. This helps you
                understand why DBSCAN classifies each point as core, border, or noise.
            </div>
            """,
            unsafe_allow_html=True,
        )

        nc1, nc2 = st.columns([1, 3])
        with nc1:
            point_idx = st.number_input(
                "Point index",
                min_value=0,
                max_value=len(df) - 1,
                value=0,
                step=1,
                key="neighbor_point_idx",
                help=f"Select a point index between 0 and {len(df) - 1}.",
            )

            # Find neighbors on the raw (or scaled) data — use same scaling as DBSCAN
            from sklearn.preprocessing import StandardScaler
            if scale_data:
                X_query = StandardScaler().fit_transform(X)
            else:
                X_query = X.copy()

            neighbor_indices, n_neighbors = find_neighbors(X_query, int(point_idx), eps)

            pt_type = result.point_types[int(point_idx)]
            pt_cluster = result.labels[int(point_idx)]
            cluster_display = f"Cluster {pt_cluster}" if pt_cluster != -1 else "Noise"

            st.markdown(f"**Type:** `{pt_type}`")
            st.markdown(f"**Cluster:** `{cluster_display}`")
            st.markdown(f"**Neighbors:** `{n_neighbors}`")
            st.markdown(f"**MinPts needed:** `{min_pts}`")

            if pt_type == "core":
                st.success(f"✅ Core point — has {n_neighbors} ≥ {min_pts} neighbors.")
            elif pt_type == "border":
                st.warning(f"⚡ Border point — has {n_neighbors} < {min_pts} neighbors, but is near a core point.")
            else:
                st.error(f"❌ Noise point — has {n_neighbors} < {min_pts} neighbors and is not near any core point.")

        with nc2:
            # Plot uses original (unscaled) coordinates for display
            fig_nb = build_neighborhood_plot(
                df, result,
                point_idx=int(point_idx),
                neighbor_indices=neighbor_indices,
                eps_value=eps,
                x_col=x_col,
                y_col=y_col,
            )
            st.plotly_chart(fig_nb, width="stretch")

    with tab_data:
        show_df = df[[x_col, y_col]].copy()
        show_df["Cluster"] = result.labels
        show_df["Point Type"] = result.point_types
        st.dataframe(show_df, width="stretch", height=400)

    # ── Results Panel ───────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Clustering Results")

    # Handle zero-cluster edge case
    if result.n_clusters == 0:
        st.markdown(
            """
            <div class="info-box" style="border-left-color: #f43f5e;">
                <strong>⚠️ No clusters found!</strong>
                DBSCAN classified all points as noise with the current parameters.
                Try <b>increasing Epsilon (ε)</b> or <b>decreasing MinPts</b> to allow
                denser groupings.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Metric cards row
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        metric_card("Clusters Found", result.n_clusters)
    with mc2:
        metric_card("Noise Points", f"{result.n_noise}  ({result.noise_pct}%)", "orange")
    with mc3:
        metric_card("Core Points", result.n_core, "green")
    with mc4:
        metric_card("Border Points", result.n_border, "purple")

    # Silhouette score (if valid)
    if result.silhouette is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        sc1, sc2, sc3 = st.columns([1, 2, 1])
        with sc2:
            metric_card("Silhouette Score", result.silhouette, "yellow")
            st.markdown(
                """
                <p style="text-align:center; color:#94a3b8; font-size:0.82rem; margin-top:0.3rem;">
                    Ranges from −1 to 1. Higher values indicate better-defined clusters.
                </p>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            """
            <div class="info-box" style="border-left-color: #f97316;">
                <strong>ℹ️ Silhouette score unavailable:</strong>
                requires at least 2 clusters with non-noise points.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Cluster size breakdown
    if result.cluster_sizes:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Cluster Size Breakdown")
        size_df = pd.DataFrame(
            [
                {"Cluster": f"Cluster {k}", "Points": v, "Percentage": f"{v / len(df) * 100:.1f}%"}
                for k, v in result.cluster_sizes.items()
            ]
        )
        st.dataframe(size_df, width="stretch", hide_index=True)

    # ── Parameter Explanation ───────────────────────────────────
    st.markdown("---")
    with st.expander("💡 Understanding the current parameters", expanded=False):
        st.markdown(
            f"""
            | Parameter | Value | Meaning |
            |-----------|-------|---------|
            | **Epsilon (ε)** | `{eps}` | Two points are neighbors if their distance ≤ {eps}. Larger ε merges more points into clusters; smaller ε creates tighter, more granular clusters. |
            | **MinPts** | `{min_pts}` | A point needs at least {min_pts} neighbors within ε to be a **core point**. Higher MinPts makes the algorithm more conservative, producing fewer but denser clusters. |
            | **Standardize** | `{'Yes' if scale_data else 'No'}` | {'Features are scaled to zero mean and unit variance before clustering, ensuring equal contribution from each axis.' if scale_data else 'Raw feature values are used directly.'} |
            """
        )
