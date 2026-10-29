# 🔬 DBSCAN Algorithm Visualizer

An interactive educational application that demonstrates how the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm works using real scikit-learn computations, colorful Plotly visualizations, and clear explanations.

Built for the **Advanced Learner Practical — Machine Learning** course.

---




# DBSCAN Interactive Walkthrough & Visualizer

[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![GitHub Pages](https://img.shields.io/badge/Demo-GitHub%20Pages-brightgreen.svg)](https://hajarekrushna.github.io/DBSCAN-Interactive-Walkthrough/)
[![Pure JavaScript](https://img.shields.io/badge/Vanilla-JavaScript-yellow.svg)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![Zero Dependencies](https://img.shields.io/badge/Dependencies-0-success.svg)](#technology-stack)

An intuitive, interactive, and beautifully designed visual simulator for **DBSCAN** (*Density-Based Spatial Clustering of Applications with Noise*). Built specifically for learners, students, and practitioners in machine learning to demystify how density-based clustering works under the hood.

> 🚀 **Live Demo:** [**https://hajarekrushna.github.io/DBSCAN-Interactive-Walkthrough/**](https://hajarekrushna.github.io/DBSCAN-Interactive-Walkthrough/)

---

## 🌟 Key Features

- **Point-by-Point Algorithmic Simulation:** Watch every discrete operation unfold step-by-step on a 10-point 2D dataset ($P_1$ through $P_{10}$).
- **Interactive Playback Controls:**
  - Play, Pause, Step Forward, Step Backward, and Restart.
  - Interactive slider scrubber to jump to any stage in the clustering timeline.
  - Adjustable playback speed ($0.75\times$, $1\times$, $1.5\times$, $2\times$).
- **Voice Narration & Live Captions:**
  - Integrated speech synthesis (`window.speechSynthesis`) explains each phase out loud.
  - Word-by-word synchronized caption box with toggleable captions and audio narration.
- **Dynamic Geometric SVG Visuals:**
  - Dynamic $\epsilon$-neighborhood circles with distance labels.
  - Animated neighbor rings (amber), cluster waiting queues (dashed purple), and status highlights.
  - Core points (filled colored circles), border points (translucent rings), and noise points (grey crosses).
- **Live Mathematical Formulas:** Displays actual Euclidean distance calculations in real time:
  $$\text{dist}(p, q) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2} \le \epsilon$$
- **Hyperparameter Sensitivity Experiments:**
  - **$\epsilon$ too small ($0.5$):** Under-clustering — every point is flagged as noise.
  - **$\epsilon$ too large ($5.0$):** Over-clustering — separate clusters and noise points merge into one giant cluster.
  - **Strict $MinPts$ ($4$):** Demonstrates how higher density requirements eliminate core points.
- **Zero External Dependencies:** Built with pure vanilla HTML5, CSS3, SVG, and modern ES6+ JavaScript. No build step, no npm packages, and no heavyweight frameworks needed.

---

## 🧠 How DBSCAN Works

DBSCAN classifies points based on local density within a specified radius $\epsilon$:

1. **Core Point:** A point $p$ is a core point if at least $MinPts$ points (including $p$) are located within its $\epsilon$-neighborhood:
   $$|\mathcal{N}_\epsilon(p)| \ge MinPts$$
2. **Border Point:** A point is a border point if its neighborhood has fewer than $MinPts$ points, but it is density-reachable from a core point:
   $$|\mathcal{N}_\epsilon(p)| < MinPts \quad \text{and} \quad p \in \mathcal{N}_\epsilon(c) \text{ where } c \text{ is Core}$$
3. **Noise / Outlier:** Any point that is neither a core point nor a border point:
   $$|\mathcal{N}_\epsilon(p)| < MinPts \quad \text{and} \quad \forall c \in \text{Core}, p \notin \mathcal{N}_\epsilon(c)$$

### Cluster Expansion Algorithm
```
For each unvisited point p in Dataset:
    Mark p as visited
    N = get_neighbors(p, epsilon)
    If |N| < MinPts:
        Mark p as NOISE (temporary)
    Else:
        Create new Cluster C
        Add p to C as CORE
        Initialize queue Q with N \ {p}
        While Q is not empty:
            Pop point q from Q
            If q was NOISE:
                Mark q as BORDER in C
            If q is not visited:
                Mark q as visited
                N_q = get_neighbors(q, epsilon)
                Add q to C
                If |N_q| >= MinPts:
                    Mark q as CORE in C
                    Add unseen points in N_q to Q
                Else:
                    Mark q as BORDER in C
```

---


## Features

- **Interactive Visualizer** — Generate synthetic datasets (moons, circles, blobs, etc.) or upload CSV files, tune ε and MinPts in real time, and watch clusters form.
- **Core / Border / Noise Classification** — Points are visually distinguished with different markers and colors.
- **Live Metrics** — Cluster count, noise percentage, core/border point counts, silhouette score (when valid), and per-cluster size breakdown.
- **Educational Content** — Step-by-step algorithm walkthrough, parameter explanations, advantages & limitations, real-world applications.
- **DBSCAN vs K-Means** — Side-by-side live comparison on a moons dataset.


<img width="1361" height="718" alt="image" src="https://github.com/user-attachments/assets/ca651778-4987-4d06-a1cc-e6a59fb8bf28" />


<img width="1365" height="640" alt="image" src="https://github.com/user-attachments/assets/3a65d2d7-0584-495e-8baf-5a6593c3f753" />


<img width="1365" height="649" alt="image" src="https://github.com/user-attachments/assets/3857706e-16c6-4122-9204-e2f07b330f0f" />



<img width="1362" height="651" alt="image" src="https://github.com/user-attachments/assets/581372f3-1424-48ca-969c-75bb8ce3fb0e" />




## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Streamlit | Web UI framework |
| scikit-learn | DBSCAN & K-Means computation |
| Plotly | Interactive charts |
| NumPy / pandas | Data processing |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py
```

The app will open at `http://localhost:8501`.

## Project Structure

```
├── app.py                      # Main entry point
├── config.py                   # Theme colors & constants
├── requirements.txt            # Dependencies
├── components/
│   ├── dataset_generator.py    # Synthetic data & CSV upload
│   ├── dbscan_engine.py        # scikit-learn DBSCAN wrapper
│   ├── plot_builder.py         # Plotly chart construction
│   └── styles.py               # Custom CSS theme
├── pages/
│   ├── dashboard.py            # Landing page
│   ├── visualizer.py           # Main DBSCAN visualizer
│   └── education.py            # Educational content
└── assets/
```

## License

Academic project — for educational purposes.
