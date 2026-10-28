# 🔬 DBSCAN Algorithm Visualizer

An interactive educational application that demonstrates how the **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** algorithm works using real scikit-learn computations, colorful Plotly visualizations, and clear explanations.

Built for the **Advanced Learner Practical — Machine Learning** course.

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
