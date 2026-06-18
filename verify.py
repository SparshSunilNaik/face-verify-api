import argparse, cv2, numpy as np
from deepface import DeepFace

def read_bgr(path: str):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img

def capture_one_frame(cam_index=0, window="Webcam (press c to capture, q to quit)"):
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_index}")
    frame_to_return = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Failed to read from camera.")
            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                frame_to_return = frame.copy()
                break
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    if frame_to_return is None:
        raise SystemExit("No frame captured. Run again and press 'c' to capture.")
    return frame_to_return

def main():
    parser = argparse.ArgumentParser(description="Face verification: reference image vs webcam snapshot")
    parser.add_argument("--ref", required=True, help="Path to reference image (jpg/png)")
    parser.add_argument("--cam", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--model", default="Facenet512", help="DeepFace model (default Facenet512)")
    parser.add_argument("--detector", default="mtcnn", help="Detector backend (default mtcnn)")
    parser.add_argument("--metric", default="cosine", help="Distance metric (cosine/euclidean/euclidean_l2)")
    parser.add_argument("--custom_threshold", type=float, default=None, help="Override model threshold")
    args = parser.parse_args()

    print("Loading reference image…")
    ref_bgr = read_bgr(args.ref)

    print("Warming up model (first run downloads weights)…")
    _ = DeepFace.build_model(args.model)
    print("Ready.\n")

    print("Open camera and press 'c' to capture a photo (or 'q' to quit).")
    live_bgr = capture_one_frame(cam_index=args.cam)

    # DeepFace expects RGB arrays
    ref_rgb  = cv2.cvtColor(ref_bgr,  cv2.COLOR_BGR2RGB)
    live_rgb = cv2.cvtColor(live_bgr, cv2.COLOR_BGR2RGB)

    print("Verifying…")
    res = DeepFace.verify(
        ref_rgb, live_rgb,
        model_name=args.model,
        detector_backend=args.detector,
        distance_metric=args.metric,
        enforce_detection=True
    )

    # Optional threshold override
    threshold = res.get("threshold")
    verified  = bool(res["verified"])
    if args.custom_threshold is not None and threshold is not None:
        verified = (res["distance"] <= args.custom_threshold)
        threshold = args.custom_threshold

    verdict = "MATCH" if verified else "NO MATCH"
    print("\n=== RESULT ===")
    print(f"{verdict}")
    print(f"Distance: {res['distance']:.4f}  (threshold: {threshold if threshold is not None else 'n/a'})")
    print(f"Model: {args.model} | Detector: {args.detector} | Metric: {args.metric}")

if __name__ == "__main__":
    main()
