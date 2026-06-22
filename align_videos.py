import cv2
import numpy as np
import os
import glob
import json
import sys
import argparse

def get_video_files(base_dir):
    """Recursively search for mp4, avi, and mov video files in base_dir,
    excluding files inside 'aligned_videos' directory.
    """
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.MP4', '*.AVI', '*.MOV']
    videos = []
    
    for root, dirs, files in os.walk(base_dir):
        if 'aligned_videos' in root:
            continue
        for ext in video_extensions:
            for filepath in glob.glob(os.path.join(root, ext)):
                filepath = os.path.abspath(filepath)
                if filepath not in videos:
                    videos.append(filepath)
    return sorted(videos)

def select_reference_frame(video_path, headless=False, default_frame=0):
    """Interactively scrub through a video to select a frame as the reference."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open reference video: {video_path}")
        return None, None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if headless:
        print(f"Headless mode: Auto-selecting reference frame index {default_frame} from {os.path.basename(video_path)}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, default_frame)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return frame, default_frame
        else:
            print("Error: Could not read the specified default frame.")
            return None, None

    print("\n--- Interactive Reference Frame Selection ---")
    print(f"Loading video: {os.path.basename(video_path)}")
    print(f"Total frames: {total_frames} | Resolution: {width}x{height} | FPS: {fps:.2f}")
    print("\nControls:")
    print("  [Space]          - Play / Pause")
    print("  [Right Arrow]/[D] - Step forward 10 frames")
    print("  [Left Arrow]/[A]  - Step backward 10 frames")
    print("  [Up Arrow]/[W]    - Step forward 100 frames")
    print("  [Down Arrow]/[S]  - Step backward 100 frames")
    print("  [Enter] or [Y]    - Accept current frame as Reference")
    print("  [Esc] or [Q]      - Abort script")
    
    frame_idx = default_frame
    playing = False
    ref_frame = None
    
    cv2.namedWindow("Select Reference Frame", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Reference Frame", 1280, 720)
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            if frame_idx >= total_frames:
                frame_idx = total_frames - 1
            elif frame_idx < 0:
                frame_idx = 0
            continue
            
        ref_frame = frame.copy()
        
        display_img = frame.copy()
        h, w, _ = display_img.shape
        overlay_text = f"Frame: {frame_idx}/{total_frames - 1} | Status: {'PLAYING' if playing else 'PAUSED'}"
        cv2.putText(display_img, overlay_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(display_img, "Press [Enter] to choose this frame", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Select Reference Frame", display_img)
        
        wait_time = int(1000 / fps) if playing else 0
        key = cv2.waitKey(wait_time) & 0xFF
        
        if playing:
            frame_idx = (frame_idx + 1) % total_frames
            
        if key == ord(' '):
            playing = not playing
        elif key in [83, ord('d'), ord('D')]: # Right / D
            frame_idx = min(frame_idx + 10, total_frames - 1)
            playing = False
        elif key in [81, ord('a'), ord('A')]: # Left / A
            frame_idx = max(frame_idx - 10, 0)
            playing = False
        elif key in [82, ord('w'), ord('W')]: # Up / W
            frame_idx = min(frame_idx + 100, total_frames - 1)
            playing = False
        elif key in [84, ord('s'), ord('S')]: # Down / S
            frame_idx = max(frame_idx - 100, 0)
            playing = False
        elif key in [13, ord('y'), ord('Y')]: # Enter or Y
            break
        elif key in [27, ord('q'), ord('Q')]: # Esc or Q
            print("Selection aborted by user.")
            cv2.destroyAllWindows()
            cap.release()
            return None, None
            
    cv2.destroyWindow("Select Reference Frame")
    cap.release()
    return ref_frame, frame_idx

def draw_static_mask(ref_img, skip_masking=False):
    """Draw an interactive polygon mask on the reference image to specify
    the region of static background elements.
    """
    img_h, img_w, _ = ref_img.shape
    
    if skip_masking:
        print("Masking: Skipped (Using full frame)")
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask.fill(255)
        return mask

    pts = []
    
    print("\n--- Draw Static Background Mask ---")
    print("Masking instructions:")
    print("  We want to match features ONLY in static areas (e.g. sidewalks, buildings, curbs).")
    print("  Exclude moving parts (the buggy track area, swaying trees, active spectators).")
    print("\nControls:")
    print("  [Left Click]     - Add vertex for polygon")
    print("  [Right Click]/[C]- Close current polygon")
    print("  [R]              - Reset and clear mask")
    print("  [D]              - Skip and use full image (Default)")
    print("  [Enter]          - Confirm mask and continue")
    print("  [Esc] or [Q]     - Abort script")

    window_name = "Draw Static Mask - Select Background Areas"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    polygon_closed = False
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal pts, polygon_closed
        if polygon_closed:
            return
            
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(pts) >= 3:
                polygon_closed = True

    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display_img = ref_img.copy()
        
        if len(pts) > 0:
            for i in range(len(pts)):
                cv2.circle(display_img, pts[i], 5, (0, 0, 255), -1)
                if i > 0:
                    cv2.line(display_img, pts[i-1], pts[i], (0, 255, 0), 2)
            
            if polygon_closed:
                cv2.line(display_img, pts[-1], pts[0], (0, 255, 0), 2)
                overlay = display_img.copy()
                cv2.fillPoly(overlay, [np.array(pts, dtype=np.int32)], (0, 255, 0))
                cv2.addWeighted(overlay, 0.4, display_img, 0.6, 0, display_img)
        
        cv2.putText(display_img, "Draw polygon over STATIC areas. [Enter] to confirm.", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        if polygon_closed:
            cv2.putText(display_img, "Polygon closed! Press [Enter] to approve, or [R] to reset.", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(display_img, "Click to add points. Right Click to close.", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow(window_name, display_img)
        key = cv2.waitKey(50) & 0xFF
        
        if key == ord('c') or key == ord('C'):
            if len(pts) >= 3:
                polygon_closed = True
        elif key == ord('r') or key == ord('R'):
            pts = []
            polygon_closed = False
            print("Mask reset.")
        elif key == ord('d') or key == ord('D'):
            pts = []
            polygon_closed = False
            print("Skipping mask - using full image.")
            break
        elif key in [13, ord('\r'), ord('\n')]: # Enter
            if not polygon_closed and len(pts) >= 3:
                polygon_closed = True
            else:
                break
        elif key in [27, ord('q'), ord('Q')]: # Esc or Q
            print("Masking aborted by user.")
            cv2.destroyAllWindows()
            return None

    cv2.destroyWindow(window_name)
    
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 255)
    else:
        mask.fill(255)
        
    return mask

def save_metadata(output_dir, ref_video_path, ref_frame_idx, ref_frame, mask):
    """Saves reference frame, mask, and configuration metadata."""
    meta_dir = os.path.join(output_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(meta_dir, "reference_frame.png"), ref_frame)
    cv2.imwrite(os.path.join(meta_dir, "reference_mask.png"), mask)
    
    config = {
        "reference_video": os.path.abspath(ref_video_path),
        "reference_frame_index": ref_frame_idx,
        "resolution": [ref_frame.shape[1], ref_frame.shape[0]]
    }
    
    with open(os.path.join(meta_dir, "reference_config.json"), "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved reference metadata to: {meta_dir}")

def load_metadata(output_dir):
    """Attempts to load reference frame, mask, and configuration metadata."""
    meta_dir = os.path.join(output_dir, "metadata")
    config_path = os.path.join(meta_dir, "reference_config.json")
    frame_path = os.path.join(meta_dir, "reference_frame.png")
    mask_path = os.path.join(meta_dir, "reference_mask.png")
    
    if os.path.exists(config_path) and os.path.exists(frame_path) and os.path.exists(mask_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            ref_frame = cv2.imread(frame_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if ref_frame is not None and mask is not None:
                return ref_frame, mask, config
        except Exception as e:
            print(f"Error loading metadata: {e}")
    return None, None, None

def align_video(video_path, output_path, ref_img, ref_kp, ref_des, detector, 
                use_orb=False, proc_w=1280, min_matches=15, min_inliers=10, 
                headless=False, limit_frames=None, num_samples=20):
    """Calculates a single robust median Homography matrix across a set of sampled frames,
    then warps all frames uniformly using this matrix. This eliminates frame-to-frame
    alignment jittering and accelerates the processing loop.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return False
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if limit_frames is not None:
        total_frames = min(total_frames, limit_frames)
        
    ref_h_orig, ref_w_orig = ref_img.shape[:2]
    
    # 1. Setup VideoWriter (always writes at reference frame original size)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (ref_w_orig, ref_h_orig))
    
    if not out.isOpened():
        print(f"Error: Could not open VideoWriter for {output_path}")
        cap.release()
        return False
        
    print(f"\nAligning: {os.path.basename(video_path)}")
    print(f"Output: {output_path}")
    print(f"Original Res: {width}x{height} -> Target Res: {ref_w_orig}x{ref_h_orig}")
    
    # Calculate scale factors for target processing
    if proc_w is None or proc_w <= 0 or proc_w >= width:
        resize_target = False
        proc_w_target = width
        proc_h_target = height
        S_target = np.eye(3, dtype=np.float32)
    else:
        resize_target = True
        s_target = proc_w / width
        proc_w_target = int(proc_w)
        proc_h_target = int(height * s_target)
        S_target = np.diag([s_target, s_target, 1.0])
        
    # Scale matrix for reference mapping
    if proc_w is None or proc_w <= 0 or proc_w >= ref_w_orig:
        S_ref_inv = np.eye(3, dtype=np.float32)
    else:
        s_ref = proc_w / ref_w_orig
        S_ref_inv = np.diag([1.0/s_ref, 1.0/s_ref, 1.0])
        
    # Setup matcher
    if use_orb:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        
    # --- PHASE 1: Homography Estimation via Sampling ---
    print(f"Estimating framing using {num_samples} sample frames...")
    
    # Sample uniformly from 5% to 95% of video to avoid transition/fade-in frames
    start_frame = int(total_frames * 0.05)
    end_frame = int(total_frames * 0.95)
    
    if end_frame <= start_frame:
        sample_indices = [0]
    else:
        sample_indices = np.linspace(start_frame, end_frame, num_samples, dtype=int).tolist()
        
    H_list = []
    
    for s_idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, s_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        if resize_target:
            frame_proc = cv2.resize(frame, (proc_w_target, proc_h_target))
        else:
            frame_proc = frame
            
        gray = cv2.cvtColor(frame_proc, cv2.COLOR_BGR2GRAY)
        kp, des = detector.detectAndCompute(gray, None)
        
        if des is not None and len(des) >= 2:
            try:
                matches = matcher.knnMatch(des, ref_des, k=2)
                
                # Apply Lowe's ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)
                
                if len(good_matches) >= min_matches:
                    src_pts = np.float32([kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    
                    H_proc, mask_ransac = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                    if H_proc is not None:
                        inliers = int(np.sum(mask_ransac))
                        if inliers >= min_inliers:
                            # Scale the Homography matrix back to full resolution
                            H_large = S_ref_inv @ H_proc @ S_target
                            # Normalize Homography scale factor so bottom-right element is 1.0
                            if abs(H_large[2, 2]) > 1e-8:
                                H_large = H_large / H_large[2, 2]
                            H_list.append(H_large)
            except Exception as e:
                pass
                
    # Compute median Homography
    if len(H_list) > 0:
        H_final = np.median(np.array(H_list), axis=0)
        print(f"  Successfully estimated alignment from {len(H_list)}/{len(sample_indices)} sample frames.")
    else:
        # Fallback to scale matrix
        sx = ref_w_orig / width
        sy = ref_h_orig / height
        H_final = np.diag([sx, sy, 1.0])
        print("  Warning: No features matched on sample frames. Falling back to simple centering/scaling.")
        
    # --- PHASE 2: High-Speed Warping ---
    print("Warping all frames uniformly...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    frame_count = 0
    
    if not headless:
        progress_window = "Alignment Process Preview"
        cv2.namedWindow(progress_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(progress_window, 960, 540)
        
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if limit_frames is not None and frame_count > limit_frames:
            break
            
        # Warp with static median Homography
        warped = cv2.warpPerspective(frame, H_final, (ref_w_orig, ref_h_orig))
        out.write(warped)
        
        # Display visual check occasionally (every 10 frames to optimize speed)
        if not headless and frame_count % 10 == 0:
            ref_vis = cv2.resize(ref_img, (960, 540))
            warped_vis = cv2.resize(warped, (960, 540))
            blend = cv2.addWeighted(ref_vis, 0.4, warped_vis, 0.6, 0)
            
            info_str = f"Frame: {frame_count}/{total_frames} | Warping | Jitter-Free Static Homography"
            cv2.putText(blend, info_str, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(blend, "Press 'Q' to abort / skip", (15, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            cv2.imshow(progress_window, blend)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nAlignment process interrupted by user.")
                confirm = input("Skip current video [s] or Abort entire process [a]? ").strip().lower()
                if confirm == 'a':
                    cv2.destroyAllWindows()
                    cap.release()
                    out.release()
                    return False
                else:
                    print("Skipping current video.")
                    break
                    
        if frame_count % 100 == 0 or frame_count == total_frames:
            sys.stdout.write(f"\rProgress: {frame_count}/{total_frames} frames warped.")
            sys.stdout.flush()
            
    print(f"\nDone. Saved aligned video to {output_path}")
    cap.release()
    out.release()
    return True

def main():
    parser = argparse.ArgumentParser(description="CMU Buggy Chute Drone Video Alignment Tool")
    parser.add_argument("--ref-idx", type=int, default=None, help="Index of reference video")
    parser.add_argument("--ref-frame", type=int, default=0, help="Reference frame index (default: 0)")
    parser.add_argument("--no-mask", action="store_true", help="Skip mask creation and use entire frame")
    parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI displays)")
    parser.add_argument("--limit-frames", type=int, default=None, help="Limit frames to process per video (useful for testing)")
    parser.add_argument("--detector", choices=["sift", "orb"], default="sift", help="Feature detector (default: sift)")
    parser.add_argument("--proc-width", type=int, default=1280, help="Resized frame width for feature matching (default: 1280, 0 to disable)")
    parser.add_argument("--num-samples", type=int, default=200, help="Number of frames to sample for static homography calculation (default: 200)")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(base_dir, "aligned_videos")
    
    print("=" * 60)
    print(" CMU Buggy Chute Drone Video Alignment Tool ")
    print("=" * 60)
    
    videos = get_video_files(base_dir)
    if not videos:
        print("No videos (.mp4, .avi, .mov) found in the Lines directory structure.")
        print(f"Please place your videos in folders under: {base_dir}")
        return
        
    print(f"Found {len(videos)} video file(s):")
    for idx, video_path in enumerate(videos):
        rel_path = os.path.relpath(video_path, base_dir)
        print(f"  [{idx}] {rel_path}")
        
    ref_frame = None
    mask = None
    config = None
    
    if args.ref_idx is None:
        existing_ref = load_metadata(output_base_dir)
        if existing_ref[0] is not None:
            ref_frame, mask, config = existing_ref
            print("\nFound existing alignment configuration from a previous run:")
            print(f"  Source Video: {os.path.basename(config.get('reference_video', 'Unknown'))}")
            print(f"  Frame Index: {config.get('reference_frame_index', 'Unknown')}")
            
            if args.headless:
                print("Headless mode: Reusing existing reference configuration.")
                reuse = 'y'
            else:
                reuse = input("\nDo you want to reuse this reference configuration? (y/n): ").strip().lower()
                
            if reuse != 'y':
                ref_frame = None
                mask = None
                config = None
            
    if ref_frame is None:
        if args.ref_idx is not None:
            if 0 <= args.ref_idx < len(videos):
                ref_video_path = videos[args.ref_idx]
                ref_frame_idx = args.ref_frame
            else:
                print(f"Error: Reference index {args.ref_idx} out of bounds.")
                return
        else:
            while True:
                try:
                    ref_choice = input(f"\nChoose reference video index [0-{len(videos)-1}]: ").strip()
                    ref_idx = int(ref_choice)
                    if 0 <= ref_idx < len(videos):
                        ref_video_path = videos[ref_idx]
                        ref_frame_idx = 0
                        break
                    print("Index out of bounds.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
                
        ref_frame, ref_frame_idx = select_reference_frame(
            ref_video_path, 
            headless=args.headless, 
            default_frame=args.ref_frame if args.ref_idx is not None else 0
        )
        if ref_frame is None:
            print("Reference selection failed. Exiting.")
            return
            
        mask = draw_static_mask(ref_frame, skip_masking=args.no_mask or args.headless)
        if mask is None:
            print("Mask drawing cancelled. Exiting.")
            return
            
        save_metadata(output_base_dir, ref_video_path, ref_frame_idx, ref_frame, mask)
        
    ref_h_orig, ref_w_orig = ref_frame.shape[:2]
    
    # 4. Prepare reference variables for matching resolution
    if args.proc_width is None or args.proc_width <= 0 or args.proc_width >= ref_w_orig:
        proc_w_ref = ref_w_orig
        proc_h_ref = ref_h_orig
        ref_img_proc = ref_frame
        mask_proc = mask
    else:
        s_ref = args.proc_width / ref_w_orig
        proc_w_ref = int(args.proc_width)
        proc_h_ref = int(ref_h_orig * s_ref)
        ref_img_proc = cv2.resize(ref_frame, (proc_w_ref, proc_h_ref))
        mask_proc = cv2.resize(mask, (proc_w_ref, proc_h_ref), interpolation=cv2.INTER_NEAREST)
        
    ref_gray = cv2.cvtColor(ref_img_proc, cv2.COLOR_BGR2GRAY)
    
    print(f"\nInitializing {args.detector.upper()} detector...")
    if args.detector == "orb":
        detector = cv2.ORB_create(nfeatures=2000)
    else:
        detector = cv2.SIFT_create(nfeatures=2000)
    
    ref_kp, ref_des = detector.detectAndCompute(ref_gray, mask=mask_proc)
    print(f"Found {len(ref_kp)} keypoints in masked reference frame (at {proc_w_ref}x{proc_h_ref}).")
    if len(ref_kp) < 10:
        print("Warning: Very few keypoints found in masked region. Feature matching may fail.")
        print("Consider selecting a different mask or reference frame.")
        
    print(f"\nPreparing to align {len(videos)} video files...")
    
    for idx, video_path in enumerate(videos):
        rel_path = os.path.relpath(video_path, base_dir)
        output_path = os.path.join(output_base_dir, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = align_video(
            video_path, 
            output_path, 
            ref_frame, 
            ref_kp, 
            ref_des, 
            detector,
            use_orb=(args.detector == "orb"),
            proc_w=args.proc_width,
            headless=args.headless,
            limit_frames=args.limit_frames,
            num_samples=args.num_samples
        )
        if not success:
            print("\nBatch alignment halted or aborted.")
            break
            
    cv2.destroyAllWindows()
    print("\nProcessing complete!")
    print(f"All aligned videos are stored in: {output_base_dir}")

if __name__ == "__main__":
    main()
