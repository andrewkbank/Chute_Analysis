import cv2
import numpy as np

# Real-world F-150 dimensions (inches)
TRUCK_LENGTH_IN = 232.0
TRUCK_WIDTH_IN = 79.9
PHYSICAL_RATIO = TRUCK_WIDTH_IN / TRUCK_LENGTH_IN  # ~0.3444

video_path = "Chute_Analysis/aligned_videos/2026 Mens Prelims/CIA A (Kingfisher) vs Spirit C (Kingpin).mp4"  # <-- Change to your video filename
current_frame_idx = 0

current_line_start = None
active_line_type = "length"  # Toggles between 'length' and 'width'

# Storage schema: { frame_idx: { 'length': ((x1,y1), (x2,y2)), 'width': ((x1,y1), (x2,y2)) } }
frame_data = {}

def mouse_callback(event, x, y, flags, param):
    global current_line_start, active_line_type, current_frame_idx, frame_data

    if event == cv2.EVENT_LBUTTONDOWN:
        current_line_start = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        if current_line_start:
            end_point = (x, y)
            if current_frame_idx not in frame_data:
                frame_data[current_frame_idx] = {}
            
            frame_data[current_frame_idx][active_line_type] = (current_line_start, end_point)
            print(f"[Frame {current_frame_idx}] Set {active_line_type} vector.")
            current_line_start = None

# Initialize Capture Engine
cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cv2.namedWindow("Turn Warp Analyzer")
cv2.setMouseCallback("Turn Warp Analyzer", mouse_callback)

print("""
=== TURN & WARP ANALYZER INSTRUCTIONS ===
1. 'A' / 'D' keys -> Move backward / forward through frames.
2. Click-and-drag to draw the bounding line vectors.
3. Press 'TAB' to toggle between drawing LENGTH and WIDTH.
4. Press 'C' to clear markings on the current frame.
5. Press 'S' to solve the 2D directional warp breakdown.
6. Press 'Q' to quit.
""")

while True:
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    metrics = frame_data.get(current_frame_idx, {})

    # Draw Length line (Neon Cyan)
    if "length" in metrics:
        pts = metrics["length"]
        cv2.line(display_frame, pts[0], pts[1], (255, 255, 0), 2)
        cv2.putText(display_frame, "Length Vector", (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    # Draw Width line (Neon Orange)
    if "width" in metrics:
        pts = metrics["width"]
        cv2.line(display_frame, pts[0], pts[1], (0, 165, 255), 2)
        cv2.putText(display_frame, "Width Vector", (pts[0][0], pts[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    # HUD Overlay text
    cv2.putText(display_frame, f"Frame: {current_frame_idx}/{total_frames}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display_frame, f"ACTIVE MODE: {active_line_type.upper()} (TAB to swap)", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow("Turn Warp Analyzer", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("d"):
        current_frame_idx = min(total_frames - 1, current_frame_idx + 1)
    elif key == ord("a"):
        current_frame_idx = max(0, current_frame_idx - 1)
    elif key == 9:  # TAB Key
        active_line_type = "width" if active_line_type == "length" else "length"
    elif key == ord("c"):
        if current_frame_idx in frame_data:
            del frame_data[current_frame_idx]
            print(f"Cleared frame {current_frame_idx}")
    elif key == ord("s"):
        print("\n=== AXIAL DISTORTION SOLVER (X vs Y WARP) ===")
        
        for f_idx, markers in sorted(frame_data.items()):
            if "length" in markers and "width" in markers:
                l_pts, w_pts = markers["length"], markers["width"]
                
                # Compute vector component deltas
                dx_l, dy_l = l_pts[1][0] - l_pts[0][0], l_pts[1][1] - l_pts[0][1]
                dx_w, dy_w = w_pts[1][0] - w_pts[0][0], w_pts[1][1] - w_pts[0][1]
                
                # Absolute pixel lengths
                l_px = np.sqrt(dx_l**2 + dy_l**2)
                w_px = np.sqrt(dx_w**2 + dy_w**2)
                
                # Calculate orientation angles relative to screen horizontal (X-axis)
                angle_l = np.abs(np.arctan2(dy_l, dx_l) * 180 / np.pi)
                angle_w = np.abs(np.arctan2(dy_w, dx_w) * 180 / np.pi)
                
                # Structural Scaling Ratios
                measured_ratio = w_px / l_px
                total_warp_deviation = measured_ratio / PHYSICAL_RATIO
                
                print(f"Frame {f_idx:04d}:")
                print(f"  -> Length Orientation Angle: {angle_l:.1f}° | Width Angle: {angle_w:.1f}°")
                print(f"  -> Global Warp Factor (Width/Length vs Real): {total_warp_deviation:.4f}")
                
                # Directional Analysis Interpretation
                # If length is aligned mostly with screen X (angle near 0/180) and width with screen Y (angle near 90)
                if angle_l < 25 or angle_l > 155:
                    scale_x = TRUCK_LENGTH_IN / l_px
                    scale_y = TRUCK_WIDTH_IN / w_px
                    print(f"  [Calculated Scale Mapping] X-Axis: {scale_x:.3f} in/px | Y-Axis: {scale_y:.3f} in/px")
                # If the truck has rotated 90 degrees in the turn (length is now vertical, width horizontal)
                elif 65 < angle_l < 115:
                    scale_y = TRUCK_LENGTH_IN / l_px
                    scale_x = TRUCK_WIDTH_IN / w_px
                    print(f"  [Calculated Scale Mapping] X-Axis: {scale_x:.3f} in/px | Y-Axis: {scale_y:.3f} in/px")
                else:
                    print("  [Calculated Scale Mapping] Diagonal orientation; use data points to interpolate intermediate shear boundaries.")
        print("=============================================\n")

cap.release()
cv2.destroyAllWindows()