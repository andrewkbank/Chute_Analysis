import cv2
import numpy as np
import json
import os
from pathlib import Path
from scipy.signal import butter, filtfilt
import re

CONFIG_FILE = "chute_config.json"
DATA_LOG_FILE = "chute_analytics_results.json"

def parse_buggy_identities(video_path):
    """Splits the filename by 'vs' to return both potential buggies and the division."""
    path_obj = Path(video_path)
    division = path_obj.parent.name 
    clean_name = os.path.splitext(path_obj.name)[0]
    
    parts = re.split(r'\s+vs\s+', clean_name, flags=re.IGNORECASE)
    buggies = [p.strip() for p in parts if p.strip()]
    
    # If the file naming layout is single-team fallback
    if not buggies:
        buggies = [clean_name]
    return buggies, division

# --- Boundary Lines Setup ---
points = []
def mark_line(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point recorded at: ({x}, {y})")

def get_boundary_lines(video_path):
    global points
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
        return config["entrance"], config["exit"]
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None, None

    print("\n--- NO CONFIG FOUND: MARK BOUNDARIES ---")
    cv2.namedWindow("Mark Boundaries")
    cv2.setMouseCallback("Mark Boundaries", mark_line)

    while True:
        temp_frame = frame.copy()
        if len(points) >= 2: cv2.line(temp_frame, tuple(points[0]), tuple(points[1]), (0, 255, 0), 2)
        if len(points) >= 4: cv2.line(temp_frame, tuple(points[2]), tuple(points[3]), (0, 0, 255), 2)
        cv2.imshow("Mark Boundaries", temp_frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or len(points) >= 4: break
            
    cv2.destroyAllWindows()
    entrance, exit_p = [points[0], points[1]], [points[2], points[3]]
    with open(CONFIG_FILE, "w") as f:
        json.dump({"entrance": entrance, "exit": exit_p}, f, indent=4)
    return entrance, exit_p

def select_buggy_roi_at_frame(video_path, entrance_line, exit_line):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    max_display_w = 1280
    if orig_w > max_display_w:
        scale_factor = max_display_w / orig_w
        display_w, display_h = max_display_w, int(orig_h * scale_factor)
    else:
        display_w, display_h = orig_w, orig_h

    cv2.destroyAllWindows() 
    cv2.namedWindow("Scrub to Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scrub to Frame", display_w, display_h)
    
    def on_trackbar(val): pass
    cv2.createTrackbar("Frame", "Scrub to Frame", 0, total_frames - 1, on_trackbar)
    cv2.waitKey(100) 
    
    chosen_frame = None
    while True:
        frame_idx = cv2.getTrackbarPos("Frame", "Scrub to Frame")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: break
            
        # Draw boundaries directly onto the scrub screen for setup visibility
        cv2.line(frame, tuple(entrance_line[0]), tuple(entrance_line[1]), (0, 255, 0), 2)
        cv2.line(frame, tuple(exit_line[0]), tuple(exit_line[1]), (0, 0, 255), 2)
        
        cv2.imshow("Scrub to Frame", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == ord(' '): 
            chosen_frame = frame_idx
            break
        elif key == ord('q'): break
            
    cv2.destroyWindow("Scrub to Frame")
    cv2.waitKey(1) 
    
    if chosen_frame is not None:
        cv2.namedWindow("Select Buggy", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Select Buggy", display_w, display_h)
        # Redisplay clean frame slice without drawn vectors for actual tracker calibration
        cap.set(cv2.CAP_PROP_POS_FRAMES, chosen_frame)
        _, clean_roi_frame = cap.read()
        roi = cv2.selectROI("Select Buggy", clean_roi_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select Buggy")
        cv2.waitKey(1)
        cap.release()
        return chosen_frame, roi
    
    cap.release()
    return None, None

# --- Core Math Helpers ---
def is_past_line(p, line_p1, line_p2):
    return (line_p2[0] - line_p1[0]) * (p[1] - line_p1[1]) - (line_p2[1] - line_p1[1]) * (p[0] - line_p1[0])

def process_buggy_kinematics(raw_trajectory, cutoff=0.08):
    """Applies a Butterworth filter and computes analytical speed and lateral acceleration."""
    if len(raw_trajectory) < 15:
        return None, None, 0.0

    traj_np = np.array(raw_trajectory)
    x, y = traj_np[:, 0], traj_np[:, 1]
    
    # Zero-phase Butterworth low-pass filter
    b, a = butter(3, cutoff, btype='low', analog=False)
    padlen = min(15, len(x) - 1)
    smoothed_x = filtfilt(b, a, x, padlen=padlen)
    smoothed_y = filtfilt(b, a, y, padlen=padlen)

    # Derivatives via global central differences
    vx = np.gradient(smoothed_x)
    vy = np.gradient(smoothed_y)
    ax = np.gradient(vx)
    ay = np.gradient(vy)

    # Compute speeds (pixels/frame)
    speeds = np.sqrt(vx**2 + vy**2)
    
    # Total Line Length calculation
    line_length = float(np.sum(speeds[1:]))

    # Analytical Lateral Acceleration: ac = |vx*ay - vy*ax| / sqrt(vx^2 + vy^2)
    numerator = np.abs(vx * ay - vy * ax)
    denominator = speeds.copy()
    denominator[denominator < 1e-5] = 1e-5  # Safe guard division
    
    lat_accel = numerator / denominator
    
    # Clean up trajectory output data pairs
    final_trajectory = [[round(float(smoothed_x[i]), 1), round(float(smoothed_y[i]), 1)] for i in range(len(x))]
    final_speeds = [round(float(s), 2) for s in speeds]
    final_lat_accel = [round(float(a), 3) for a in lat_accel]

    return final_trajectory, final_speeds, final_lat_accel, line_length


# --- Main Dual-Tracking Pipeline Wrapper ---
def analyze_video_multi_buggy(video_path):
    video_name = os.path.basename(video_path)
    entrance_line, exit_line = get_boundary_lines(video_path) # Assumed configuration helper exists
    
    buggies_found, race_division = parse_buggy_identities(video_path)
    
    # We will initialize tracking states dynamically based on user targets
    print(f"\nProcessing Video: {video_name} [{race_division}]")
    print(f"Identified tracks: {buggies_found}")
    track_targets = []
    
    for buggy in buggies_found:
        choice = input(f"Do you want to track '{buggy}'? (y/n): ").strip().lower()
        if choice == 'y':
            # Initialize unique tracking boxes/gates, start frames, and storage logs
            start_frame, roi = select_buggy_roi_at_frame(video_path, entrance_line, exit_line)
            if roi and roi != (0,0,0,0):
                rx, ry, rw, rh = map(int, roi)
                gate_padding = 50
                track_targets.append({
                    "name": buggy,
                    "start_frame": start_frame,
                    "gate": [rx - gate_padding, ry - gate_padding, rw + (gate_padding * 2), rh + (gate_padding * 2)],
                    "raw_trajectory": [],
                    "frame_bounds": {"start": None, "end": None},
                    "inside_chute": False,
                    "active": True
                })

    if not track_targets:
        print("No buggies selected for tracking. Exiting.")
        return

    # Instantiate single background model shared across all buggies
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=True)
    cap = cv2.VideoCapture(video_path)
    
    # Global earliest start frame shortcut seeding
    global_start_frame = min(t["start_frame"] for t in track_targets)
    
    # Warm up background model
    for _ in range(max(0, global_start_frame - 10)):
        ret, frame = cap.read()
        if not ret: break
        bg_subtractor.apply(frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, global_start_frame)
    gate_padding = 20

    while cap.isOpened():
        current_frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = cap.read()
        if not ret: break

        fg_mask = bg_subtractor.apply(frame)
        _, thresholded_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)

        active_or_waiting_count = 0
        just_deactivated = False # Flags if a buggy exited on this specific frame
        
        for t in track_targets:
            if not t["active"]: continue
            active_or_waiting_count += 1
            
            # If a buggy hasn't reached its start_frame yet, skip its tracking block
            if current_frame_id < t["start_frame"]: continue

            # --- [Keep standard Gated Tracking & Contour Detection code here] ---
            gx, gy, gw, gh = t["gate"]
            gx1, gy1 = max(0, gx), max(0, gy)
            gx2, gy2 = min(frame.shape[1], gx + gw), min(frame.shape[0], gy + gh)
            
            local_mask = thresholded_mask[gy1:gy2, gx1:gx2]
            contours, _ = cv2.findContours(local_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(largest_contour) > 20:
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"]) + gx1
                        cy = int(M["m01"] / M["m00"]) + gy1
                        center = np.array([cx, cy])
                        
                        bx, by, bw, bh = cv2.boundingRect(largest_contour)
                        t["gate"] = [bx + gx1 - gate_padding, by + gy1 - gate_padding, bw + (gate_padding * 2), bh + (gate_padding * 2)]

                        side_entrance = is_past_line(center, entrance_line[0], entrance_line[1])
                        side_exit = is_past_line(center, exit_line[0], exit_line[1])
                        
                        if side_entrance < 0 and not t["inside_chute"]:
                            t["inside_chute"] = True
                            t["frame_bounds"]["start"] = current_frame_id
                            print(f"--> [{t['name']}] Entered Chute at frame {current_frame_id}")

                        if t["inside_chute"]:
                            t["raw_trajectory"].append([float(cx), float(cy)])

                        # When the buggy crosses the exit line:
                        if side_exit < 0 and t["inside_chute"]:
                            t["inside_chute"] = False
                            t["frame_bounds"]["end"] = current_frame_id
                            t["active"] = False
                            just_deactivated = True  # Trigger timeline check
                            print(f"--> [{t['name']}] Exited Chute at frame {current_frame_id}")

                        # Visual overlays labeled by target name
                        cv2.rectangle(frame, (bx + gx1, by + gy1), (bx + gx1 + bw, by + gy1 + bh), (0, 255, 0), 2)
                        cv2.putText(frame, t["name"], (bx + gx1, by + gy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                continue
            
            # --- [Keep standard visual drawing box/text overlays] ---

        # Visual bounds lines rendering
        cv2.line(frame, tuple(entrance_line[0]), tuple(entrance_line[1]), (0, 255, 0), 2)
        cv2.line(frame, tuple(exit_line[0]), tuple(exit_line[1]), (0, 0, 255), 2)
        cv2.imshow("Multi-Buggy Synchronized Tracker", frame)
        
        # --- TIMELINE FAST-FORWARD ENGINE ---
        if just_deactivated:
            # Gather all buggies that are still waiting for their start frame
            waiting_buggies = [t for t in track_targets if t["active"] and current_frame_id < t["start_frame"]]
            
            if waiting_buggies:
                # Find the earliest start frame among the remaining waiting buggies
                next_jump_frame = min(t["start_frame"] for t in waiting_buggies)
                
                # Only jump if the gap is worth skipping (e.g., more than 5 frames)
                if next_jump_frame > (current_frame_id + 5):
                    print(f"--> [Timeline Skip] Fast-forwarding empty gap: Frame {current_frame_id} -> {next_jump_frame}")
                    
                    # Hard-set the OpenCV video capture read pointer
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_jump_frame)
                    
                    # Optional: Re-verify active count immediately to prevent dropouts
                    active_or_waiting_count = len([t for t in track_targets if t["active"]])

        # If no buggies are left active or waiting, or 'q' is pressed, kill the video loop
        if active_or_waiting_count == 0 or (cv2.waitKey(1) & 0xFF == ord('q')): 
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- SAVE PROCESSOR AND NEW JSON MATRIX STRUCTURING ---
    # Read existing database entries
    if os.path.exists(DATA_LOG_FILE):
        with open(DATA_LOG_FILE, "r") as f:
            try: database = json.load(f)
            except json.JSONDecodeError: database = {}
    else:
        database = {}

    # Enforce safe nested division grouping to prevent naming overlaps
    if race_division not in database:
        database[race_division] = {}

    for t in track_targets:
        if len(t["raw_trajectory"]) < 15:
            print(f"Skipping telemetry compile for {t['name']}: Insufficient path data collected.")
            continue

        traj, speeds, lat_accel, total_len = process_buggy_kinematics(t["raw_trajectory"])
        
        if traj is None: continue

        # Lightweight arrays (Flat sequential mapping with frame limits defined in summary)
        run_data = {
            "summary": {
                "source_video": video_name,
                "start_frame": t["frame_bounds"]["start"],
                "end_frame": t["frame_bounds"]["end"],
                "total_line_length_px": round(total_len, 2),
                "avg_speed_px_frame": round(float(np.mean([s for s in speeds if s > 0])), 2) if speeds else 0.0,
                "max_lateral_accel_px_f2": round(float(np.max(lat_accel)), 4) if lat_accel else 0.0
            },
            "speeds": speeds,
            "lateral_acceleration": lat_accel,
            "trajectory": traj
        }

        # Save to specific division sub-object namespace securely
        database[race_division][t["name"]] = run_data
        print(f"--> Saved tracking compilation for '{t['name']}' inside category '{race_division}' successfully.")

    with open(DATA_LOG_FILE, "w") as f:
        json.dump(database, f, indent=2)

analyze_video_multi_buggy("Lines/aligned_videos/2026 Womens Prelims/Spirit D (Kingpin) vs DG B (Insite).mp4")

#The drone moves in "2026 Mens Finals/SDC A (Paranoia).mp4"