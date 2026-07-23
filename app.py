from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import json
import heapq
import math
import os

app = Flask(__name__)

# ── Load Models ────────────────────────────────────────────────────────────────
try:
    gb_model = joblib.load('models/fire_model.pkl')
    rf_model = joblib.load('models/fire_rf_model.pkl')
    with open('models/feature_names.json') as f:
        FEATURE_NAMES = json.load(f)
    print("[OK] Models loaded.")
except Exception as e:
    print(f"[WARN] Model load error: {e}")
    gb_model = rf_model = None
    FEATURE_NAMES = ['temp', 'RH', 'wind', 'FFMC', 'DMC', 'DC', 'ISI', 'month_num', 'day_num']

WIND_DIR_MAP = {
    'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
    'S': 180, 'SW': 225, 'W': 270, 'NW': 315
}

# ── Helper: Build features DataFrame ─────────────────────────────────────────
def build_features(temp, hum, wind, ffmc=85.0, dmc=26.0, dc=94.0, isi=5.0, month=6, day=3):
    return pd.DataFrame([[temp, hum, wind, ffmc, dmc, dc, isi, month, day]],
                        columns=FEATURE_NAMES)

# ── Helper: Wind bias vector ──────────────────────────────────────────────────
def wind_bias(wind_dir, wind_speed):
    """Returns (dlat, dlon) bias from wind direction."""
    angle = math.radians(WIND_DIR_MAP.get(wind_dir, 0))
    strength = min(wind_speed / 40.0, 1.0) * 0.0006
    # Wind blows fire DOWNWIND: N wind pushes fire South → negative lat
    dlat = -math.cos(angle) * strength
    dlon =  math.sin(angle) * strength
    return dlat, dlon

# ── Route: Home ───────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("index.html")

# ── Route: Predict spread probability for a single cell ──────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        temp     = float(data.get('temp', 30))
        hum      = float(data.get('hum', 40))
        wind     = float(data.get('wind', 15))
        wind_dir = data.get('wind_dir', 'N')
        ffmc     = float(data.get('ffmc', 85.0))
        dmc      = float(data.get('dmc', 26.0))
        dc       = float(data.get('dc', 94.0))
        isi      = float(data.get('isi', 5.0))
        month    = int(data.get('month', 6))
        day      = int(data.get('day', 3))

        features = build_features(temp, hum, wind, ffmc, dmc, dc, isi, month, day)

        if gb_model:
            proba = gb_model.predict_proba(features)[0]
            spread_prob = float(proba[1])
            # Ensemble: average with RF
            if rf_model:
                rf_proba = rf_model.predict_proba(features)[0]
                spread_prob = (spread_prob + float(rf_proba[1])) / 2.0
        else:
            # Fallback heuristic if model not loaded
            spread_prob = min(1.0, (temp / 50) * (1 - hum / 100) * (wind / 30))

        # Adjust probability based on live conditions
        humidity_factor = max(0.1, 1.0 - (hum / 100.0))
        temp_factor     = min(1.5, temp / 30.0)
        wind_factor     = min(1.4, wind / 20.0)
        adjusted_prob   = min(1.0, spread_prob * humidity_factor * temp_factor * wind_factor)

        dlat, dlon = wind_bias(wind_dir, wind)

        return jsonify({
            "spread": int(adjusted_prob > 0.45),
            "probability": round(adjusted_prob, 3),
            "wind_bias": {"dlat": dlat, "dlon": dlon}
        })
    except Exception as e:
        return jsonify({"error": str(e), "spread": 0, "probability": 0.0}), 500

# ── Route: Simulate one step of cellular automaton (batch) ───────────────────
@app.route("/simulate_step", methods=["POST"])
def simulate_step():
    """
    Input:  { fire_cells: [[lat,lon],...], temp, hum, wind, wind_dir, step }
    Output: { new_cells: [[lat,lon,prob],...], burned: [[lat,lon],...],
              wind_vector: {dlat, dlon, angle_deg, speed}, spread_direction: str }
    """
    try:
        data      = request.json
        fire_cells = data.get('fire_cells', [])
        temp      = float(data.get('temp', 30))
        hum       = float(data.get('hum', 40))
        wind      = float(data.get('wind', 15))
        wind_dir  = data.get('wind_dir', 'N')
        step_size = float(data.get('step_size', 0.0015))

        if gb_model:
            features    = build_features(temp, hum, wind)
            base_prob   = float(gb_model.predict_proba(features)[0][1])
            if rf_model:
                rf_p    = float(rf_model.predict_proba(features)[0][1])
                base_prob = (base_prob + rf_p) / 2.0
        else:
            base_prob = min(1.0, (temp / 50) * (1 - hum / 100) * (wind / 30))

        humidity_factor = max(0.1, 1.0 - (hum / 100.0))
        temp_factor     = min(1.3, temp / 35.0)
        wind_factor     = min(1.2, wind / 25.0)
        net_prob        = min(0.72, base_prob * humidity_factor * temp_factor * wind_factor)

        # Wind direction vector for directional fire spread
        wind_angle_deg = WIND_DIR_MAP.get(wind_dir, 0)
        wind_angle_rad = math.radians(wind_angle_deg)
        # Fire travels DOWNWIND: N wind → fire goes south (negative lat), etc.
        dlat_wind = -math.cos(wind_angle_rad)
        dlon_wind =  math.sin(wind_angle_rad)

        # Scale bias by wind speed (stronger wind = more directional bias)
        bias_strength = min(wind / 40.0, 1.0) * step_size * 0.55
        dlat_bias = dlat_wind * bias_strength
        dlon_bias = dlon_wind * bias_strength

        # Always include 8 directions; downwind ones get boosted probability
        directions = [
            (step_size, 0), (-step_size, 0),
            (0, step_size), (0, -step_size),
            (step_size, step_size), (-step_size, step_size),
            (step_size, -step_size), (-step_size, -step_size)
        ]

        existing = set(f"{round(c[0],4)},{round(c[1],4)}" for c in fire_cells)
        new_cells_map = {}
        burned = []

        MAX_NEW_PER_STEP = 30   # slightly larger cap for directional spread
        MAX_ACTIVE_CELLS = 200

        for cell in fire_cells:
            if len(new_cells_map) >= MAX_NEW_PER_STEP:
                break
            lat, lon = cell[0], cell[1]
            burned.append([lat, lon])
            for dlat, dlon in directions:
                # Alignment with downwind direction: +1 = fully downwind, -1 = upwind
                mag = math.sqrt(dlat**2 + dlon**2) + 1e-9
                alignment = (dlat * dlat_wind + dlon * dlon_wind) / mag
                # Downwind: boost probability. Upwind: suppress it.
                if alignment > 0.5:
                    dir_boost = 1.0 + alignment * 1.2   # strong downwind boost
                elif alignment < -0.3:
                    dir_boost = max(0.1, 1.0 + alignment * 0.8)  # upwind suppressed
                else:
                    dir_boost = 1.0

                cell_prob = min(0.88, net_prob * dir_boost)

                # Apply wind drift offset
                new_lat = round(lat + dlat + dlat_bias, 4)
                new_lon = round(lon + dlon + dlon_bias, 4)
                key     = f"{new_lat},{new_lon}"

                if key not in existing and cell_prob > 0.22:
                    if np.random.random() < cell_prob:
                        if key not in new_cells_map or new_cells_map[key] < cell_prob:
                            new_cells_map[key] = cell_prob
                            if len(new_cells_map) >= MAX_NEW_PER_STEP:
                                break

        new_cells = [[float(k.split(',')[0]), float(k.split(',')[1]), round(v, 3)]
                     for k, v in new_cells_map.items()]

        # Compute spread centroid direction for frontend arrow
        if new_cells and fire_cells:
            fire_lat = sum(c[0] for c in fire_cells) / len(fire_cells)
            fire_lon = sum(c[1] for c in fire_cells) / len(fire_cells)
            new_lat_avg = sum(c[0] for c in new_cells) / len(new_cells)
            new_lon_avg = sum(c[1] for c in new_cells) / len(new_cells)
            spread_dlat = new_lat_avg - fire_lat
            spread_dlon = new_lon_avg - fire_lon
            spread_angle = math.degrees(math.atan2(spread_dlon, spread_dlat))
        else:
            spread_angle = 0.0

        return jsonify({
            "new_cells":  new_cells,
            "burned":     burned,
            "probability": round(net_prob, 3),
            "wind_vector": {
                "dlat": round(dlat_wind, 4),
                "dlon": round(dlon_wind, 4),
                "angle_deg": wind_angle_deg,
                "speed": wind
            },
            "spread_angle_deg": round(spread_angle, 1),
            "spread_direction": wind_dir
        })
    except Exception as e:
        return jsonify({"error": str(e), "new_cells": [], "burned": []}), 500

# ── Route: A* pathfinding ─────────────────────────────────────────────────────
@app.route("/astar", methods=["POST"])
def astar():
    """
    Input:  { start:[lat,lon], goal:[lat,lon], fire_cells:[[lat,lon],...],
              burned_cells:[[lat,lon],...], step:0.001 }
    Output: { path:[[lat,lon],...], found:bool, distance_km:float,
              fire_zones_avoided:int }
    Fire cells AND their large ring buffer are ABSOLUTE hard obstacles.
    The algorithm guarantees the path never comes within BUFFER cells of any fire.
    """
    try:
        data         = request.json
        start        = tuple(data['start'])
        goal         = tuple(data['goal'])
        fire_cells   = data.get('fire_cells', [])
        burned_cells = data.get('burned_cells', [])
        step         = float(data.get('step', 0.001))

        # ── SAFETY BUFFER: 15 cells × 0.001 deg ≈ 1.65 km clearance around every fire cell ──
        BUFFER = 15
        fire_set = set()
        # Also build a quick list of (lat, lon) for distance checks
        fire_xy = []
        for fc in fire_cells:
            base_lat = round(fc[0] / step) * step
            base_lon = round(fc[1] / step) * step
            fire_xy.append((base_lat, base_lon))
            for di in range(-BUFFER, BUFFER + 1):
                for dj in range(-BUFFER, BUFFER + 1):
                    # Only add cells within circular buffer (not square)
                    if di * di + dj * dj <= BUFFER * BUFFER:
                        k = f"{round(base_lat + di*step, 4)},{round(base_lon + dj*step, 4)}"
                        fire_set.add(k)

        # ── Burned cells buffer: 8 cells ──
        BURNED_BUFFER = 8
        burned_set = set()
        for bc in burned_cells:
            b_lat = round(bc[0] / step) * step
            b_lon = round(bc[1] / step) * step
            for di in range(-BURNED_BUFFER, BURNED_BUFFER + 1):
                for dj in range(-BURNED_BUFFER, BURNED_BUFFER + 1):
                    if di * di + dj * dj <= BURNED_BUFFER * BURNED_BUFFER:
                        burned_set.add(f"{round(b_lat+di*step, 4)},{round(b_lon+dj*step, 4)}")

        def h(a, b):
            return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

        def key(p):
            return f"{round(p[0],4)},{round(p[1],4)}"

        def to_point(k):
            parts = k.split(',')
            return (float(parts[0]), float(parts[1]))

        def neighbors(p):
            lat, lon = p
            dirs = [(step,0),(-step,0),(0,step),(0,-step),
                    (step,step),(-step,step),(step,-step),(-step,-step)]
            return [(round(lat+d[0],4), round(lon+d[1],4)) for d in dirs]

        def danger_cost(p):
            pk = key(p)
            # ── ABSOLUTE HARD BLOCK: fire set (circular buffer) ──
            if pk in fire_set:
                return float('inf')
            # ── ABSOLUTE HARD BLOCK: burned set ──
            if pk in burned_set:
                return float('inf')
            # ── Extra proximity cost just outside the buffer to steer well clear ──
            cost = 0
            for fx, fy in fire_xy:
                dist = math.sqrt((p[0]-fx)**2 + (p[1]-fy)**2)
                if dist < step * BUFFER:
                    return float('inf')   # double-check: within buffer → block
                elif dist < step * (BUFFER + 6):
                    cost += 80            # strong repulsion just outside buffer
                elif dist < step * (BUFFER + 12):
                    cost += 25            # moderate repulsion
            return cost

        # ── A* search ──
        open_heap = []
        g_cost    = {key(start): 0.0}
        parent    = {}
        visited   = set()
        heapq.heappush(open_heap, (h(start, goal), 0.0, key(start)))
        found_key  = None
        iterations = 0
        MAX_ITER   = 40000   # higher limit to allow long detours

        while open_heap and iterations < MAX_ITER:
            iterations += 1
            f, g, ck = heapq.heappop(open_heap)
            if ck in visited:
                continue
            visited.add(ck)
            cp = to_point(ck)

            if h(cp, goal) < step * 2.0:
                found_key = ck
                break

            for nb in neighbors(cp):
                nk = key(nb)
                if nk in visited:
                    continue
                dc = danger_cost(nb)
                if dc == float('inf'):
                    continue  # hard block — never enter
                move_cost = step * math.sqrt(2) if (nb[0]!=cp[0] and nb[1]!=cp[1]) else step
                new_g = g + move_cost + dc * 0.00005
                if nk not in g_cost or new_g < g_cost[nk]:
                    g_cost[nk] = new_g
                    parent[nk] = ck
                    heapq.heappush(open_heap, (new_g + h(nb, goal), new_g, nk))

        # Only accept closest node if it's very near goal AND not in fire zone
        if not found_key and visited:
            safe_visited = [k for k in visited if k not in fire_set and k not in burned_set]
            if safe_visited:
                closest = min(safe_visited, key=lambda k: h(to_point(k), goal))
                if h(to_point(closest), goal) < step * 4:
                    found_key = closest

        if not found_key:
            return jsonify({
                "found": False, "path": [], "distance_km": 0,
                "fire_zones_avoided": len(fire_cells)
            })

        # ── Reconstruct path ──
        path = []
        cur  = found_key
        while cur:
            p = to_point(cur)
            path.append([p[0], p[1]])
            cur = parent.get(cur)
        path.reverse()

        # ── Final safety validation: reject any path point inside fire zone ──
        for pt in path:
            pk = key(pt)
            if pk in fire_set or pk in burned_set:
                # Path is unsafe — return not found
                return jsonify({
                    "found": False, "path": [], "distance_km": 0,
                    "fire_zones_avoided": len(fire_cells),
                    "error": "Path validation failed — fire zone detected"
                })

        dist_km = 0.0
        for i in range(1, len(path)):
            dlat = (path[i][0]-path[i-1][0]) * 111.0
            dlon = (path[i][1]-path[i-1][1]) * 111.0 * math.cos(math.radians(path[i][0]))
            dist_km += math.sqrt(dlat**2 + dlon**2)

        return jsonify({
            "found":            True,
            "path":             path,
            "distance_km":      round(dist_km, 2),
            "nodes_explored":   iterations,
            "fire_zones_avoided": len(fire_cells)
        })
    except Exception as e:
        return jsonify({"error": str(e), "found": False, "path": []}), 500

# ── Route: AI Fire Control Analysis (replaces water/sand suppression) ──────────
@app.route("/fire_control_analysis", methods=["POST"])
def fire_control_analysis():
    """
    Heuristic tactical analysis of fire spread. Identifies strategic suppression
    zones using geometry + wind direction — no external API required.

    Input:  { fire_cells, burned_cells, wind_dir, wind, hum,
              start_pos, goal_pos, region_name }
    Output: { zones:[{lat,lon,radius,priority,label,reason}],
              analysis:str, summary:str }
    """
    try:
        data         = request.json
        fire_cells   = data.get('fire_cells', [])
        burned_cells = data.get('burned_cells', [])
        wind_dir     = data.get('wind_dir', 'N')
        wind_speed   = float(data.get('wind', 15))
        hum          = float(data.get('hum', 40))
        start_pos    = data.get('start_pos')
        goal_pos     = data.get('goal_pos')
        region_name  = data.get('region_name', 'Forest Region')

        if not fire_cells:
            return jsonify({
                "zones": [],
                "analysis": "No active fire cells detected.\nStart the simulation first.",
                "summary": "No active fire"
            })

        # ── Fire geometry ──
        fire_lats = [c[0] for c in fire_cells]
        fire_lons = [c[1] for c in fire_cells]
        centroid_lat = sum(fire_lats) / len(fire_lats)
        centroid_lon = sum(fire_lons) / len(fire_lons)
        lat_min, lat_max = min(fire_lats), max(fire_lats)
        lon_min, lon_max = min(fire_lons), max(fire_lons)
        fire_spread_km = math.sqrt(
            ((lat_max - lat_min) * 111)**2 +
            ((lon_max - lon_min) * 111)**2
        )

        # ── Wind vector (fire travels downwind) ──
        wind_angle_rad = math.radians(WIND_DIR_MAP.get(wind_dir, 0))
        dlat_wind = -math.cos(wind_angle_rad)   # N wind → fire drifts south
        dlon_wind =  math.sin(wind_angle_rad)

        # ── Fire front: cells furthest downwind ──
        def wind_proj(cell):
            return ((cell[0] - centroid_lat) * dlat_wind +
                    (cell[1] - centroid_lon) * dlon_wind)

        front_cell = sorted(fire_cells, key=wind_proj, reverse=True)[0]

        # ── Adaptive offset based on fire size ──
        offset = max(0.0025, fire_spread_km / 111 * 0.45)

        dir_map = {
            'N':'northern','NE':'northeastern','E':'eastern','SE':'southeastern',
            'S':'southern','SW':'southwestern','W':'western','NW':'northwestern'
        }
        downwind_name = dir_map.get(wind_dir, 'downwind')

        # ── Zone 1: Primary firebreak ahead of fire front ──
        z1_lat = round(front_cell[0] + dlat_wind * offset, 4)
        z1_lon = round(front_cell[1] + dlon_wind * offset, 4)

        # ── Zone 2: Left flank ──
        z2_lat = round(centroid_lat + dlat_wind*offset*0.35 - dlon_wind*offset*0.70, 4)
        z2_lon = round(centroid_lon + dlon_wind*offset*0.35 + dlat_wind*offset*0.70, 4)

        # ── Zone 3: Right flank ──
        z3_lat = round(centroid_lat + dlat_wind*offset*0.35 + dlon_wind*offset*0.70, 4)
        z3_lon = round(centroid_lon + dlon_wind*offset*0.35 - dlat_wind*offset*0.70, 4)

        zones = [
            {
                "lat": z1_lat, "lon": z1_lon, "radius": 280,
                "priority": "CRITICAL", "label": "Firebreak Alpha",
                "reason": (
                    f"Primary fire front advancing {downwind_name} at {wind_speed:.0f} km/h. "
                    f"Establishing a firebreak here intercepts the main spread vector and can "
                    f"reduce fire progression by 60-70%. Deploy aerial water bombers + ground firebreak crews. "
                    f"Act within 15 minutes."
                )
            },
            {
                "lat": z2_lat, "lon": z2_lon, "radius": 200,
                "priority": "HIGH", "label": "Flank Control Beta",
                "reason": (
                    "Left flank is widening at ~40% of primary spread rate. "
                    "Firebreak trenching and retardant drop here prevents lateral perimeter expansion. "
                    "Coordinate with Zone Alpha for simultaneous suppression."
                )
            },
            {
                "lat": z3_lat, "lon": z3_lon, "radius": 200,
                "priority": "HIGH", "label": "Flank Control Gamma",
                "reason": (
                    "Right flank containment closes the suppression pincer around the fire head. "
                    "A controlled backburn from this position redirects fire energy inward, "
                    "accelerating exhaustion of available ground fuel."
                )
            }
        ]

        # ── Zone 4: Corridor protection (if goal set) ──
        if goal_pos and start_pos:
            mid_lat = round((centroid_lat + goal_pos[0]) / 2, 4)
            mid_lon = round((centroid_lon + goal_pos[1]) / 2, 4)
            zones.append({
                "lat": mid_lat, "lon": mid_lon, "radius": 160,
                "priority": "MODERATE", "label": "Corridor Shield Delta",
                "reason": (
                    "Preventive water barrier on the evacuation corridor between start and escape goal. "
                    "Keep this channel clear with hose lines and patrol units. "
                    "Reassess every 20 min as fire position shifts."
                )
            })

        # ── Generate tactical analysis text ──
        fire_count   = len(fire_cells)
        burned_count = len(burned_cells)
        wind_label   = 'light' if wind_speed < 15 else 'moderate' if wind_speed < 30 else 'strong'
        risk_label   = 'LOW'   if wind_speed < 15 else 'MODERATE' if wind_speed < 25 else 'HIGH'
        hum_note     = (
            f'High humidity ({hum:.0f}%) slowing spread — exploit this window.'
            if hum > 60 else
            f'Low humidity ({hum:.0f}%) accelerating combustion — immediate action critical.'
        )

        lines = [
            f"TACTICAL FIRE CONTROL — {region_name}",
            "=" * 46,
            "",
            "SITUATION REPORT:",
            f"  Active fire cells  : {fire_count}",
            f"  Burned area cells  : {burned_count}",
            f"  Fire spread (est.) : {fire_spread_km:.2f} km diameter",
            f"  Wind               : {wind_label} {wind_dir} @ {wind_speed:.0f} km/h",
            f"  Spread risk        : {risk_label}",
            f"  Fire centroid      : [{centroid_lat:.4f}, {centroid_lon:.4f}]",
            "",
            f"CONTROL ZONES IDENTIFIED: {len(zones)}",
            "",
        ]
        for i, z in enumerate(zones, 1):
            lines += [
                f"  [{i}] {z['label']}  [{z['priority']}]",
                f"      {z['reason'][:150]}",
                ""
            ]
        lines += [
            "RECOMMENDED ACTION SEQUENCE:",
            "  1. Establish Firebreak Alpha — blocks primary spread",
            "  2. Deploy Beta + Gamma flanks simultaneously",
            "  3. Maintain Delta corridor for safe evacuation",
            "  4. Re-assess every 20 min as fire position shifts",
            "",
            f"WEATHER NOTE: {hum_note}",
        ]

        analysis_text = "\n".join(lines)
        summary = (
            f"{len(zones)} control zones — {risk_label} spread risk — "
            f"{wind_label} {wind_dir} wind"
        )

        return jsonify({
            "zones":    zones,
            "analysis": analysis_text,
            "summary":  summary
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e), "zones": [], "analysis": "Analysis failed.", "summary": "Error"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)