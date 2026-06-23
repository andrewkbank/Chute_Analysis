import cv2
import numpy as np
import math

# --- CONFIGURATION ---
IMAGE_PATH = "Chute_Analysis/chute.png" # Replace with your image path
SPOT_WIDTH_FT = 9.0

# --- GLOBALS ---
state = 0  # 0: Verticals, 1: Street Lines, 2: 9ft Scale Points
lines_vert = []
lines_street = []
points_scale = []
temp_point = None

def compute_vanishing_point(lines):
    """Calculates the vanishing point given a list of line segments."""
    A = []
    for p1, p2 in lines:
        # Cross product of two points gives the homogeneous line equation (ax + by + c = 0)
        l = np.cross([p1[0], p1[1], 1], [p2[0], p2[1], 1])
        A.append(l)
    A = np.array(A)
    # SVD solves for the intersection point that minimizes the least squares error
    _, _, V = np.linalg.svd(A)
    vp = V[-1]
    return vp / vp[2] if vp[2] != 0 else vp

def pixel_to_ground(p_px, K_inv, r1, r2, r3):
    """Casts a ray from the camera through the pixel onto the ground plane."""
    ray = K_inv @ np.array([p_px[0], p_px[1], 1.0])
    
    # Calculate intersection with the ground plane (where depth = 1 camera height)
    denom = np.abs(np.dot(ray, r3))
    if denom < 1e-6:
        return None # Point is exactly on the horizon
        
    p_ground_3d = ray / denom
    
    # Project the 3D point onto the 2D ground axes
    X = np.dot(p_ground_3d, r1)
    Y = np.dot(p_ground_3d, r2)
    return np.array([X, Y])

def mouse_callback(event, x, y, flags, param):
    global state, temp_point, lines_vert, lines_street, points_scale, img_display
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if state == 0 or state == 1:
            if temp_point is None:
                temp_point = (x, y)
                cv2.circle(img_display, temp_point, 4, (0, 0, 255) if state == 0 else (255, 0, 0), -1)
            else:
                cv2.line(img_display, temp_point, (x, y), (0, 0, 255) if state == 0 else (255, 0, 0), 2)
                cv2.circle(img_display, (x, y), 4, (0, 0, 255) if state == 0 else (255, 0, 0), -1)
                if state == 0:
                    lines_vert.append((temp_point, (x, y)))
                else:
                    lines_street.append((temp_point, (x, y)))
                temp_point = None
                
        elif state == 2:
            points_scale.append((x, y))
            cv2.circle(img_display, (x, y), 4, (0, 255, 0), -1)
            if len(points_scale) > 1:
                cv2.line(img_display, points_scale[-2], points_scale[-1], (0, 255, 0), 2)

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        print(f"Error: Could not load {IMAGE_PATH}")
        exit()

    h, w = img.shape[:2]
    img_display = img.copy()

    cv2.namedWindow("Calibration UI")
    cv2.setMouseCallback("Calibration UI", mouse_callback)

    instructions = [
        "STATE 0: Draw VERTICAL lines (lamp posts). Click start & end. Press 'n' when done.",
        "STATE 1: Draw STREET lines (yellow lines). Click start & end. Press 'n' when done.",
        "STATE 2: Click sequential parking dividers (9ft gaps). Press 'n' to calculate!"
    ]

    print("--- CAMERA CALIBRATION STARTED ---")
    
    while True:
        temp_img = img_display.copy()
        
        # Display instructions on screen
        cv2.rectangle(temp_img, (0, 0), (w, 40), (0,0,0), -1)
        cv2.putText(temp_img, instructions[state], (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if temp_point is not None:
            # Live line drawing preview
            m_x, m_y = pygame_mouse = cv2.getWindowImageRect("Calibration UI")[0:2] # Fallback for preview
        
        cv2.imshow("Calibration UI", temp_img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('n'):
            if state == 0 and len(lines_vert) < 2:
                print("Please draw at least 2 vertical lines.")
                continue
            if state == 1 and len(lines_street) < 2:
                print("Please draw at least 2 street lines.")
                continue
            if state == 2 and len(points_scale) < 2:
                print("Please click at least 2 points to establish scale.")
                continue
            
            state += 1
            if state > 2:
                break
        elif key == ord('q'):
            print("Aborted.")
            exit()

    cv2.destroyAllWindows()

    # --- MATH PIPELINE ---
    print("\n--- CALCULATING 3D PROJECTION ---")
    vp_z = compute_vanishing_point(lines_vert)
    vp_y = compute_vanishing_point(lines_street)

    # 1. Calculate Focal Length (f) using orthogonality of Vanishing Points
    cx, cy = w / 2, h / 2
    vzx, vzy = vp_z[0] - cx, vp_z[1] - cy
    vyx, vyy = vp_y[0] - cx, vp_y[1] - cy
    
    dot_product = vzx * vyx + vzy * vyy
    if dot_product < 0:
        f = math.sqrt(-dot_product)
        print(f"Calculated Camera Focal Length: {f:.2f} pixels")
    else:
        print("Warning: Orthogonality check failed. Defaulting to estimated 60-deg FOV.")
        f = w / (2 * math.tan(math.radians(30)))

    # 2. Construct Intrinsic Camera Matrix (K)
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0, 1]
    ])
    K_inv = np.linalg.inv(K)

    # 3. Compute Rotation Matrix relative to the ground plane
    # R3 (Up Vector)
    r3 = K_inv @ vp_z
    if r3[1] > 0: r3 = -r3  # Ensure it points UP (negative Y in OpenCV)
    r3 = r3 / np.linalg.norm(r3)

    # R2 (Street Direction Vector)
    r2_raw = K_inv @ vp_y
    r2 = r2_raw - np.dot(r2_raw, r3) * r3  # Gram-Schmidt orthogonalization
    r2 = r2 / np.linalg.norm(r2)

    # R1 (Cross-Street Direction Vector)
    r1 = np.cross(r2, r3)
    r1 = r1 / np.linalg.norm(r1)

    # 4. Calculate Final Real-World Scale
    print("Averaging parking spot widths...")
    unscaled_pts = [pixel_to_ground(pt, K_inv, r1, r2, r3) for pt in points_scale]
    
    distances = []
    for i in range(1, len(unscaled_pts)):
        d = np.linalg.norm(unscaled_pts[i] - unscaled_pts[i-1])
        distances.append(d)
        
    avg_unscaled_dist = np.mean(distances)
    scale_factor = SPOT_WIDTH_FT / avg_unscaled_dist

    print(f"Calculated Scale Factor: {scale_factor:.4f} ft per unit")
    
    # --- TEST FUNCTION ---
    def get_real_distance(px1, px2):
        g1 = pixel_to_ground(px1, K_inv, r1, r2, r3)
        g2 = pixel_to_ground(px2, K_inv, r1, r2, r3)
        unscaled_dist = np.linalg.norm(g2 - g1)
        return unscaled_dist * scale_factor

    test_px1 = points_scale[0]
    test_px2 = points_scale[1]
    test_dist = get_real_distance(test_px1, test_px2)
    print(f"\nTest Measurement between first two clicks: {test_dist:.2f} ft")
    
    print("\nSUCCESS! To use this in your main data pipeline, copy these values:")
    print("K_inv Matrix:\n", repr(K_inv))
    print("R1 Vector:", repr(r1))
    print("R2 Vector:", repr(r2))
    print("R3 Vector:", repr(r3))
    print("Scale Factor:", scale_factor)