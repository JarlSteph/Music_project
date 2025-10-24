# 🎶 Project Overview🎶

This project analyzes and optimizes DJ playlists based on tempo (BPM) and musical key, aiming to replicate or evaluate how professional DJs (e.g., David Guetta) order their tracks.

⸻

### 📁 Data Folder

Contains all datasets, audio files, and evaluation results used in the project:
	•	🎵 David Guetta Playlists (571, 590, 638) – Real DJ sets used for evaluation.
	•	🧠 KEY-BPM_train-val – Training data for BPM and key estimation.
	•	📊 .csv files – Contain computed BPM, key, and cost results per track.

⸻

### ⚙️ Source Code (src/utilities)

Core Python modules that implement the feature extraction, cost calculation, and optimization logic:
	•	BPM.py – Extracts tempo using Fourier- and autocorrelation-based tempogram analysis.
	•	predictKey.py – Detects musical key and maps it to the Camelot wheel.
	•	transition_cost.py – Defines transition cost functions for tempo and key differences.
	•	cost_matrix.py – Builds pairwise cost matrices between all tracks.
	•	playlist_optimizer.py – Finds the optimal song order that minimizes total transition cost.
⸻

### 💻 Main Scripts
	•	main.py – Runs the full optimization pipeline:
extracts BPM and key → computes transition costs → generates optimized playlist → evaluates results.

	•	Gui_main.py
 Provides a simple graphical interface for selecting tracks, running the optimization, and viewing output visually.

⸻

🧩 Summary

This system combines music signal processing (for BPM/key estimation) with optimization algorithms (for sequencing tracks).
It is designed to test whether DJs order songs in a way that minimizes tempo and key transition costs.

