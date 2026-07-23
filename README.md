# 🌲 ForestGuard AI — Wildfire Control & Escape System

ForestGuard AI is an advanced, real-time tactical analysis and decision support system designed to manage and mitigate wildfire disasters. It combines a machine learning-driven cellular automaton simulation with advanced pathfinding algorithms to provide predictive spread modeling, safe escape routing, and strategic fire containment recommendations.

---

## 🚀 Core Features

1. **AI-Powered Wildfire Spread Simulation**
   * Uses an ensemble of **Gradient Boosting** and **Random Forest** classification models trained on the UCI Forest Fires dataset.
   * Dynamically predicts fire spread probability based on real-time weather conditions: temperature, relative humidity, wind speed, wind direction, and fuel indexes (FFMC, DMC, DC, ISI).
   * Models cellular automaton spread with realistic wind-vector biases.

2. **Safe Evacuation Path Planning**
   * Implements a custom **A\* Pathfinding Algorithm** designed for dynamic hazard avoidance.
   * Treats active fire zones and burned terrain as absolute hard obstacles.
   * Introduces a **safety buffer** clearance (up to ~1.65 km) around fire borders with proximity-based repulsion costs to keep escape routes well clear of fire lines.

3. **Alternative Road Routing**
   * Integrates the **Open Source Routing Machine (OSRM)** API to generate real road routes for comparison with AI-driven off-road paths.

4. **Strategic Fire Control & Tactical Analysis**
   * Dynamically calculates fire geometry and wind vectors to recommend strategic suppression zones.
   * Recommends positions for **primary firebreaks (Alpha)** ahead of the fire front, **flank control boundaries (Beta & Gamma)**, and **corridor shielding (Delta)** to protect evacuation routes.
   * Generates formatted situation reports detailing active cell counts, estimated diameter, spread risk level, and step-by-step action plans.

5. **Interactive Dark Mode Control Panel**
   * Responsive split-screen dashboard with full-screen Leaflet mapping.
   * Built-in database of Indian forest and hill regions (Western Ghats, Nilgiri Hills, Bandipur, Jim Corbett, etc.).
   * Interactive sliders for temperature, humidity, and wind speed, plus a custom compass selector for wind direction.
   * Real-time telemetry (burned area calculation in hectares, active fire cells, simulation steps, and dynamic progress bar).
   * Real-time AI Event Log showcasing backend predictions and pathfinding statistics.

---

## 🛠️ Tech Stack

* **Frontend**: HTML5, Vanilla CSS3 (Custom design tokens, glassmorphism, responsive flex layouts), Leaflet.js, Leaflet Heatmap.
* **Backend**: Flask (Python 3)
* **Machine Learning**: Scikit-Learn (`GradientBoostingClassifier`, `RandomForestClassifier`), Joblib, Pandas, NumPy.
* **Data Source**: UCI Machine Learning Repository (Forest Fires Dataset).

---

## 📦 Project Directory Structure

```text
├── data/
│   └── forestfires.csv           # Training dataset
├── models/
│   ├── fire_model.pkl            # Trained Gradient Boosting model
│   ├── fire_rf_model.pkl         # Trained Random Forest model
│   └── feature_names.json        # JSON mapping of feature names
├── templates/
│   └── index.html                # Frontend dashboard & Leaflet UI
├── venv/                         # Python Virtual Environment
├── .gitignore                    # Git ignore file (excludes venv/ & cache)
├── app.py                        # Flask server containing simulation & routing API
├── train_model.py                # Model training and feature engineering pipeline
└── Documentation.txt             # Project goals and requirements sheet
```

---

## 💻 Local Installation & Setup

### Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### Step 1: Clone or Download the Project
Download the repository files to your local machine.

### Step 2: Open Terminal / Shell
Open your terminal (PowerShell, Command Prompt, or Bash) and navigate to the project directory:
```bash
cd "e:\FAI&AN - SEM2 PROJECT"
```

### Step 3: Run the Application
The project includes a pre-configured Python virtual environment (`venv`) with all necessary dependencies installed. You can launch the server using it directly:

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\python.exe app.py
```

**On Windows (Command Prompt):**
```cmd
.\venv\Scripts\python app.py
```

**On macOS / Linux (if setting up fresh):**
```bash
source venv/bin/activate
python app.py
```

---

## 🖥️ How to Use the System

1. **Launch the Dashboard**: Open your browser and navigate to **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)**.
2. **Select a Forest Region**: Use the top-left map panel to examine different pre-set locations or scan coordinates.
3. **Trigger a Wildfire**: Click anywhere on the map to seed a fire cell.
4. **Adjust Weather Conditions**: Use the sidebar controls to set:
   * **Temperature**: Higher temps increase ignition rates.
   * **Humidity**: Dry air accelerates spread.
   * **Wind Speed & Direction**: Wind pushes the fire front along a directional vector.
5. **Run Simulation**: Click **Start Simulation**. Watch the cellular automaton propagate in real-time.
6. **Plan Evacuation**:
   * Click **🏃 Set Start** and place a marker on the map where a person is trapped.
   * Click **🏁 Set Goal** and place a marker in a safe location.
   * Click **🗺️ Find Escape Route**. The backend A* algorithm will calculate a safe path bypassing the fire and its safety buffer zone.
7. **Perform Tactical Analysis**: Click **🚒 Fire Control Analysis** to see where the AI recommends establishing firebreaks and dropping retardants.

---

## 🧠 Optional: Retraining the Machine Learning Models

The models are pre-trained and saved in `models/`. However, if you wish to retrain them on the UCI dataset:
```bash
.\venv\Scripts\python.exe train_model.py
```
This script will:
1. Load `data/forestfires.csv`.
2. Apply feature engineering (date mapping, binary labeling for area spread).
3. Train Gradient Boosting (primary) and Random Forest (secondary) classifiers.
4. Export updated `.pkl` files and `feature_names.json` to the `models/` directory.
