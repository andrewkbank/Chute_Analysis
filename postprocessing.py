import numpy as np
import json

# --- METRIC CONFIGURATION CONSTANTS ---
FPS = 59.96
DT = 1.0 / FPS  

SCALE_X_M = 1.379 * 0.0254 
SCALE_Y_M = 1.595 * 0.0254  
MAX_LAT_ACCEL_MS2 = 0.208 * 0.0254 * (FPS ** 2) 

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
    raw_trajectory = np.array(raw_trajectory)
    n_frames = len(raw_trajectory)
    
    if n_frames < 40:
        return raw_trajectory.tolist(), [], [], []

    x_raw = raw_trajectory[:, 0]
    y_raw = raw_trajectory[:, 1]

    # --- 1. SPATIAL SMOOTHING (Required to prevent speed staircases) ---
    x_smooth = moving_average(moving_average(x_raw, 15), 15)
    y_smooth = moving_average(moving_average(y_raw, 15), 15)
    filtered_coords = np.column_stack((x_smooth, y_smooth))

    # --- 2. RAW VELOCITY & HEADING ---
    dx_px = np.zeros(n_frames)
    dy_px = np.zeros(n_frames)
    
    dx_px[1:-1] = (x_smooth[2:] - x_smooth[:-2]) / 2.0
    dy_px[1:-1] = (y_smooth[2:] - y_smooth[:-2]) / 2.0
    dx_px[0], dy_px[0] = x_smooth[1] - x_smooth[0], y_smooth[1] - y_smooth[0]
    dx_px[-1], dy_px[-1] = x_smooth[-1] - x_smooth[-2], y_smooth[-1] - y_smooth[-2]

    vx = (dx_px * SCALE_X_M) / DT
    vy = (dy_px * SCALE_Y_M) / DT
    
    raw_speeds_ms = np.sqrt(vx**2 + vy**2)
    raw_headings_rad = np.unwrap(np.arctan2(vy, vx))

    # --- 3. PAVA ON SPEED AND HEADING ---
    # Speed is monotonically decreasing (coasting)
    pre_smoothed_speed = moving_average(raw_speeds_ms, 7)
    cleaned_speeds_ms = enforce_monotonic(pre_smoothed_speed, decreasing=True)
    
    # Heading is monotonic (only turns right)
    # Determine general turn direction by comparing start and end angles
    is_decreasing_turn = raw_headings_rad[-1] < raw_headings_rad[0]
    cleaned_headings_rad = enforce_monotonic(raw_headings_rad, decreasing=is_decreasing_turn)
    headings_deg = cleaned_headings_rad * 180.0 / np.pi

    # --- 4. KINEMATIC LATERAL ACCELERATION ---
    d_theta = np.zeros(n_frames)
    d_theta[1:-1] = (cleaned_headings_rad[2:] - cleaned_headings_rad[:-2]) / 2.0
    d_theta[0] = cleaned_headings_rad[1] - cleaned_headings_rad[0]
    d_theta[-1] = cleaned_headings_rad[-1] - cleaned_headings_rad[-2]

    omega = d_theta / DT
    
    # Because heading is now strictly monotonic via PAVA, omega will never cross 0
    # This guarantees a perfectly smooth, single-lobed cornering force curve.
    lat_accel_raw = cleaned_speeds_ms * np.abs(omega)
    lat_accel_final = np.clip(lat_accel_raw, 0.0, MAX_LAT_ACCEL_MS2)


# --- 5. FIXED UNITS SUMMARY COMPUTATION ---
    # Omit the first 10 initialization frames to match visual slice rules
    slice_speeds = cleaned_speeds_ms[10:]
    slice_accel = lat_accel_final[10:]
    
    # Track Length: Integrated distance traveled in meters
    # Speed (m/s) * Time Step (s) = Step Distance (m)
    total_distance_meters = np.sum(slice_speeds * DT)
    
    # Average Speed: Mean of the metric speed profile across the track run
    average_speed_ms = np.mean(slice_speeds) if len(slice_speeds) > 0 else 0.0
    
    # Maximum Lateral Acceleration: Peak G-Force handling threshold reached in m/s²
    peak_lateral_accel_ms2 = np.max(slice_accel) if len(slice_accel) > 0 else 0.0

    # Retain your original JSON key structural layout to prevent breaking D3 hooks
    summary_object = {
        "total_line_length_px": float(round(total_distance_meters, 3)),       # Now outputs in Meters
        "avg_speed_px_frame": float(round(average_speed_ms, 2)),              # Now outputs in m/s
        "max_lateral_accel_px_f2": float(round(peak_lateral_accel_ms2, 4))    # Now outputs in m/s²
    }

    return filtered_coords.tolist(), cleaned_speeds_ms.tolist(), headings_deg.tolist(), lat_accel_final.tolist(), summary_object
# --- EXECUTION WRAPPER ---
with open("Chute_Analysis/chute_analytics_results.json", "r") as f:
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