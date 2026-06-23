import numpy as np
import json

# --- TIMING CONSTANTS ---
FPS = 59.96
DT = 1.0 / FPS  

# --- 3D CALIBRATION CONSTANTS ---
K_INV = np.array([
    [ 3.47952092e-04,  0.00000000e+00, -6.68068016e-01],
    [ 0.00000000e+00,  3.47952092e-04, -3.75788259e-01],
    [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
])
R1 = np.array([ 0.8874196 , -0.41902869,  0.19209741])
R2 = np.array([-0.46015496, -0.82993448,  0.31538256])
R3 = np.array([ 0.02727392, -0.36827124, -0.92931826])
SCALE_FACTOR_FT = 245.2971911698062
FT_TO_M = 0.3048

def pixel_to_ground_meters(p_px):
    """Casts a 2D pixel to a 3D ground plane and scales directly to meters."""
    ray = K_INV @ np.array([p_px[0], p_px[1], 1.0])
    denom = np.abs(np.dot(ray, R3))
    
    # Avoid division by zero if point is perfectly on the horizon
    if denom < 1e-6:
        return np.array([0.0, 0.0]) 
        
    p_ground_3d = ray / denom
    
    # Project to 2D flat ground and apply physical scaling
    X_ft = np.dot(p_ground_3d, R1) * SCALE_FACTOR_FT
    Y_ft = np.dot(p_ground_3d, R2) * SCALE_FACTOR_FT
    
    return np.array([X_ft * FT_TO_M, Y_ft * FT_TO_M])

def moving_average(arr, window=15):
    pad_size = window // 2
    padded = np.pad(arr, (pad_size, pad_size), mode='reflect', reflect_type='odd')
    return np.convolve(padded, np.ones(window) / window, mode='valid')

def enforce_monotonic(y, decreasing=True):
    """
    Direction-aware PAVA. 
    Can flatten speed (decreasing) or heading (increasing/decreasing).
    """
    if not decreasing:
        y = -np.array(y)
        
    n = len(y)
    blocks = []
    
    for i in range(n):
        val, w, count = y[i], 1.0, 1
        while len(blocks) > 0 and blocks[-1][0] < val:
            prev_val, prev_w, prev_count = blocks.pop()
            new_w = prev_w + w
            val = (prev_val * prev_w + val * w) / new_w
            w = new_w
            count += prev_count
        blocks.append([val, w, count])
        
    res = []
    for b in blocks:
        res.extend([b[0]] * b[2])
        
    result = np.array(res)
    return result if decreasing else -result

def post_process_buggy_data(raw_trajectory):
    n_frames = len(raw_trajectory)
    
    if n_frames < 40:
        return {
            "trajectory": raw_trajectory, # Fallback
            "speeds": [], "headings": [], "lateral_acceleration": [],
            "summary": { "total_line_length_px": 0.0, "avg_speed_px_frame": 0.0, "max_lateral_accel_px_f2": 0.0 }
        }

    # --- 0. PERSPECTIVE TRANSFORMATION ---
    # Convert every raw pixel coordinate into real-world meters immediately
    real_trajectory = np.array([pixel_to_ground_meters(pt) for pt in raw_trajectory])
    
    x_raw = real_trajectory[:, 0]
    y_raw = real_trajectory[:, 1]

    # --- 1. SPATIAL SMOOTHING ---
    x_smooth = moving_average(moving_average(x_raw, 15), 15)
    y_smooth = moving_average(moving_average(y_raw, 15), 15)
    filtered_coords_m = np.column_stack((x_smooth, y_smooth))

    # --- 2. VELOCITY CALCULATIONS (ALREADY IN METERS) ---
    dx_m = np.zeros(n_frames)
    dy_m = np.zeros(n_frames)
    
    dx_m[1:-1] = (x_smooth[2:] - x_smooth[:-2]) / 2.0
    dy_m[1:-1] = (y_smooth[2:] - y_smooth[:-2]) / 2.0
    dx_m[0], dy_m[0] = x_smooth[1] - x_smooth[0], y_smooth[1] - y_smooth[0]
    dx_m[-1], dy_m[-1] = x_smooth[-1] - x_smooth[-2], y_smooth[-1] - y_smooth[-2]

    # Cleaned up velocity calculations (no more scale multipliers)
    vx = dx_m / DT
    vy = dy_m / DT
    
    raw_speeds_ms = np.sqrt(vx**2 + vy**2)
    raw_headings_rad = np.unwrap(np.arctan2(vy, vx))

    # --- NEW: NORMALIZE HEADING BASELINES ---
    # Force the starting angle of all trajectories into the positive 0 to 2*pi domain.
    # This maps both ~180 and ~-180 starts to ~180 without breaking the unwrapped continuity.
    baseline_shift = (raw_headings_rad[0] % (2 * np.pi)) - raw_headings_rad[0]
    raw_headings_rad += baseline_shift

    # --- 3. MONOTONIC ENFORCEMENT VIA PAVA ---
    pre_smoothed_speed = moving_average(raw_speeds_ms, 7)
    cleaned_speeds_ms = enforce_monotonic(pre_smoothed_speed, decreasing=True)
    
    is_decreasing_turn = raw_headings_rad[-1] < raw_headings_rad[0]
    cleaned_headings_rad = enforce_monotonic(raw_headings_rad, decreasing=is_decreasing_turn)
    headings_deg = cleaned_headings_rad * 180.0 / np.pi

    # --- 4. KINEMATIC LATERAL ACCELERATION ---
    d_theta = np.zeros(n_frames)
    d_theta[1:-1] = (cleaned_headings_rad[2:] - cleaned_headings_rad[:-2]) / 2.0
    d_theta[0] = cleaned_headings_rad[1] - cleaned_headings_rad[0]
    d_theta[-1] = cleaned_headings_rad[-1] - cleaned_headings_rad[-2]

    omega = d_theta / DT
    lat_accel_ms2 = cleaned_speeds_ms * np.abs(omega)

    # --- 5. SUMMARY COMPUTATION ---
    slice_speeds = cleaned_speeds_ms[10:]
    slice_accel = lat_accel_ms2[10:]
    
    total_distance_meters = np.sum(slice_speeds * DT)
    average_speed_ms = np.mean(slice_speeds) if len(slice_speeds) > 0 else 0.0
    peak_lateral_accel_ms2 = np.max(slice_accel) if len(slice_accel) > 0 else 0.0

    summary_object = {
        "total_line_length_px": float(round(total_distance_meters, 3)),       
        "avg_speed_px_frame": float(round(average_speed_ms, 2)),              
        "max_lateral_accel_px_f2": float(round(peak_lateral_accel_ms2, 4))    
    }

    return filtered_coords_m.tolist(), cleaned_speeds_ms.tolist(), headings_deg.tolist(), lat_accel_ms2.tolist(), summary_object

# --- EXECUTION WRAPPER ---
with open("chute_analytics_results.json", "r") as f:
    dashboard_data = json.load(f)

for division in dashboard_data:
    for buggy_id in dashboard_data[division]:
        buggy = dashboard_data[division][buggy_id]
        
        clean_pos, clean_v, clean_h, clean_a, summary = post_process_buggy_data(buggy["trajectory"])
        
        buggy["trajectory"] = clean_pos
        buggy["speeds"] = clean_v
        buggy["headings"] = clean_h
        buggy["lateral_acceleration"] = clean_a
        buggy["summary"] = summary

with open("chute_analytics_results_cleaned.json", "w") as f:
    json.dump(dashboard_data, f, indent=4)
print("Data pipeline metric post-processing successfully saved!")