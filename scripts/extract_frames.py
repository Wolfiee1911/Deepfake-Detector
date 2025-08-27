import os, json, cv2, mediapipe as mp
from tqdm import tqdm

# Output folders
os.makedirs('data/frames/real', exist_ok=True)
os.makedirs('data/frames/fake', exist_ok=True)

mp_face = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

def save_faces(video_path, out_dir, every_n=6, max_frames=120):
    cap = cv2.VideoCapture(video_path)
    i = saved = 0
    while saved < max_frames:
        ok, frame = cap.read()
        if not ok: break
        if i % every_n == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)
            if res.detections:
                # Pick the highest confidence face
                d = max(res.detections, key=lambda d: d.score[0])
                bb = d.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x1 = max(0, int(bb.xmin * w) - 10)
                y1 = max(0, int(bb.ymin * h) - 10)
                x2 = min(w, int((bb.xmin + bb.width) * w) + 10)
                y2 = min(h, int((bb.ymin + bb.height) * h) + 10)
                face = frame[y1:y2, x1:x2]
                if face.size:
                    cv2.imwrite(os.path.join(out_dir, f"{os.path.basename(video_path)}_{i}.jpg"), face)
                    saved += 1
        i += 1
    cap.release()

# Run for both splits
for split, out in [("configs/train_split.json", "data/frames"), ("configs/val_split.json", "data/frames")]:
    d = json.load(open(split))
    for p in tqdm(d['real'], desc=f"Extract real from {split}"):
        save_faces(p, os.path.join(out, "real"))
    for p in tqdm(d['fake'], desc=f"Extract fake from {split}"):
        save_faces(p, os.path.join(out, "fake"))

print("✅ Face frames extracted to data/frames/{real,fake}")
