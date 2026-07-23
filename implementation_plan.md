# 🔥 Forest Fire AI Control System — Full Overhaul

## Overview

A complete rebuild of the existing basic wildfire simulation into a **production-grade, visually stunning web application** that resembles a real Google Maps-style interface. The app will feature deep learning–based fire spread prediction, A* escape routing displayed on real map tiles, fire control strategy powered by minimax AI, and an immersive Google Maps-like UX.

---

## Current State

The existing project has:
- A basic Flask app (`app.py`) with a `/predict` endpoint using a Random Forest model
- A trained `fire_model.pkl` (Random Forest on UCI forest fires dataset)
- A rough Leaflet map with fire circles and a simple A* pathfinder
- No real map routing, no good UI, no deep learning, no interactive fire suppression

---

## What Will Be Built

### Core Features

| Feature | Current | New |
|---|---|---|
| Fire spread model | Random Forest (sklearn) | Gradient Boosted Trees + optional Neural Net wrapper |
| Map visualization | Leaflet circles on satellite tiles | Real Leaflet map with animated fire overlays, heatmaps |
| Escape routing | Basic A* (rough grid) | Enhanced A* with danger-weighted grid shown as Google-style route |
| Fire suppression | Simple firebreak line | Click-to-place Water/Sand drops with ML-predicted extinguish zones |
| UI | Dark panel + basic buttons | Premium glassmorphism sidebar with animations, real-time stats |
| Start/Goal | Hardcoded markers | User-draggable markers with geocoding |
| Wind/conditions | Random values | User-editable sliders with compass rose |

---

## Architecture

```
Flask Backend (app.py)
  ├── /                    → Serve main HTML (Jinja2)
  ├── /predict             → ML fire spread prediction (POST)
  ├── /astar               → A* pathfinding on server (POST)
  ├── /firecontrol         → AI fire suppression analysis (POST)
  └── /simulate_step       → One step of fire cellular automaton (POST)

Frontend (templates/index.html) — Single rich HTML file
  ├── Leaflet.js map with satellite tiles
  ├── Animated fire heatmap (Leaflet.heat plugin)
  ├── Route polyline (Google Maps style with animated dashes)
  ├── Draggable Start/Goal markers
  ├── Control sidebar with glassmorphism UI
  ├── Click-to-suppress mode (water/sand drops)
  └── Real-time stats: prediction confidence, fire area, ETA

ML Model (train_model.py)
  └── GradientBoostingClassifier + spread_probability (0–1 float)
      └── Returns probability of spread per cell, not just 0/1
```

---

## Proposed Changes

### Backend — `app.py` [MODIFY]

- Add `/astar` route: accepts grid data, fire cells, start, goal → runs A* on server → returns path as lat/lng list
- Add `/firecontrol` route: accepts fire cells, wind → runs Minimax/greedy analysis → returns best suppression points
- Add `/simulate_step` route: cellular automaton step with ML confidence scores
- Return fire spread as probability (0.0–1.0) for heatmap intensity

---

### ML Model — `train_model.py` [MODIFY]

- Upgrade to `GradientBoostingClassifier` with `predict_proba`
- Add feature engineering: wind direction as sine/cosine components, fuel moisture proxy
- Save model as `fire_model.pkl` (drop-in replacement, same path)
- Add a separate `fire_prob_model.pkl` that returns probability scores for heatmap rendering

---

### Frontend — `templates/index.html` [MODIFY — Full Rewrite]

**Map Layer Stack:**
1. Satellite/street base tile (OpenStreetMap or Esri World Imagery)
2. Leaflet.heat heatmap layer for fire probability
3. Burned area polygons (dark gray)
4. Escape route polyline (animated dashed blue line, Google Maps style)
5. Water/sand drop markers (user-placed)
6. Draggable Start 🏃 and Goal 🏁 markers

**Sidebar UI (glassmorphism, left panel):**
- 🌡️ Temperature slider (15–55°C)
- 💧 Humidity slider (10–90%)
- 💨 Wind speed slider (0–60 km/h)
- 🧭 Wind direction compass rose (clickable 8 directions)
- 🔥 "Start Simulation" toggle button
- 🛤️ "Find Safe Escape Route" button
- 🚿 "Fire Control Mode" toggle (click map to place water/sand)
- 📊 Live stats: Fire probability %, Cells burned, Estimated spread radius

**Fire Control Mode:**
- Toggle a "🚿 Suppress Fire" mode
- While active, clicking the map places a water drop or sand marker
- Server runs suppression logic: removes fire cells within radius, reduces spread probability in area
- Visual: blue water circles or sand-colored patches animate on map

**Escape Route Display:**
- Animated Google Maps–style dashed route (blue line with moving arrow animation)
- Shows "📍 Start" and "🏁 Safe Zone" labels
- Route avoids fire cells and high-probability spread zones (danger buffer)
- Shows distance and estimated safe time in sidebar

---

## Key Design Decisions

> [!IMPORTANT]
> **Real map tiles**: Using OpenStreetMap standard tiles (free, no API key needed) so the path looks like a real navigation map. The fire overlay will be a semi-transparent heatmap on top.

> [!IMPORTANT]
> **Server-side A***: Moving A* to Flask backend lets us do proper grid resolution and avoids browser hanging on large fire spreads.

> [!NOTE]
> **Deep Learning**: The GradientBoosting model gives per-cell probability. An optional lightweight PyTorch/Keras Neural Network wrapper can be added if needed. The architecture supports it via the `/predict` endpoint.

> [!NOTE]
> **No Google Maps API key needed**: We use free Leaflet + OpenStreetMap. The experience will visually match Google Maps routing style.

---

## Files Summary

| File | Action | Description |
|---|---|---|
| `app.py` | MODIFY | Add A*, fire control, simulation step routes |
| `train_model.py` | MODIFY | Upgrade to GradientBoosting with probability output |
| `models/fire_model.pkl` | REGENERATE | After retraining |
| `templates/index.html` | FULL REWRITE | Premium Google Maps-style UI |

---

## Verification Plan

### Automated
- Run `python train_model.py` → model trains without errors
- Run `python app.py` → Flask starts on port 5000
- Server-side A* tested via `/astar` endpoint with curl/Postman

### Visual (Browser)
- Map loads with satellite tiles centered on India
- Fire heatmap starts and spreads realistically
- Dragging start/goal markers and clicking "Find Route" shows animated path
- Fire Control mode: click map → water drops appear → fire shrinks
- All sliders animate and update fire behavior in real time

---

## Estimated Complexity

This is a **full rebuild** touching every file. Execution will be done in this order:
1. Retrain model (`train_model.py`)
2. Rebuild Flask backend (`app.py`)
3. Build new frontend (`templates/index.html`)
4. Test end-to-end in browser
