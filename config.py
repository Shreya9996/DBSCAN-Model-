"""
config.py — Theme colors, cluster palette, and shared constants.
"""

# ── Dark Navy Theme ──────────────────────────────────────────────
THEME = {
    "bg_primary":    "#0a0e27",
    "bg_card":       "#131a3e",
    "bg_card_hover": "#1a2350",
    "accent_blue":   "#4f8ff7",
    "accent_purple": "#a855f7",
    "accent_green":  "#22c55e",
    "accent_orange": "#f97316",
    "accent_yellow": "#eab308",
    "text_primary":  "#e2e8f0",
    "text_secondary":"#94a3b8",
    "noise_color":   "#6b7280",
    "border":        "#1e293b",
}

# Cluster colors (used in Plotly traces, up to 12 clusters)
CLUSTER_COLORS = [
    "#4f8ff7",  # blue
    "#a855f7",  # purple
    "#22c55e",  # green
    "#f97316",  # orange
    "#eab308",  # yellow
    "#ec4899",  # pink
    "#06b6d4",  # cyan
    "#f43f5e",  # rose
    "#84cc16",  # lime
    "#8b5cf6",  # violet
    "#14b8a6",  # teal
    "#d946ef",  # fuchsia
]

NOISE_COLOR = "#6b7280"

# ── Plotly layout defaults ───────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0a0e27",
    plot_bgcolor="#0f1535",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(
        bgcolor="rgba(19,26,62,0.85)",
        bordercolor="#1e293b",
        borderwidth=1,
        font=dict(size=12),
    ),
    xaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
    yaxis=dict(gridcolor="#1e293b", zerolinecolor="#1e293b"),
)

# ── Dataset defaults ─────────────────────────────────────────────
DEFAULT_N_SAMPLES = 300
MIN_SAMPLES = 50
MAX_SAMPLES = 2000
DEFAULT_EPS = 0.3
DEFAULT_MIN_PTS = 5
