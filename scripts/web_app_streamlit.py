# scripts/web_app_streamlit.py
# Deepfake Detector UI (spatial-only) — streamlined & aesthetic
# Compatible with older Streamlit widgets; includes big verdict badges.

import os, io, json, math, tempfile
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image

import torch
import timm
import streamlit as st
import matplotlib.pyplot as plt
from torchvision import transforms

# -----------------------------
# Paths & defaults
# -----------------------------
CKPT_PATH = "checkpoints/spatial_convnext_tiny_video.pt"
TITLE = "🛡️ Deepfake Detector — Spatial (Beta)"
SAMPLE_FPS_DEFAULT = 4
MAX_FRAMES_DEFAULT = 512
THRESHOLD_DEFAULT = 0.50
SMOOTH_K_DEFAULT = 9       # moving average (frames)
MIN_EVENT_SEC = 1.5        # min duration above threshold to flag

# -----------------------------
# CSS (badges, cards, tweaks)
# -----------------------------
BADGE_CSS = """
<style>
.badge {
  display:inline-block; padding:10px 16px; border-radius:999px;
  font-weight:700; letter-spacing:.3px; font-size:1.05rem;
  border:1px solid rgba(255,255,255,0.08);
  box-shadow:0 1px 2px rgba(0,0,0,.2) inset;
}
.badge-real {
  background:linear-gradient(135deg, #0E6E3A, #16A34A);
  color:#ECFDF5;
}
.badge-fake {
  background:linear-gradient(135deg, #7F1D1D, #DC2626);
  color:#FEF2F2;
}
.metric-card {
  border:1px solid rgba(255,255,255,0.08);
  border-radius:12px; padding:12px 14px; margin-top:8px;
  background:rgba(255,255,255,0.03);
}
.small { font-size:0.9rem; opacity:0.85; }
.kpill {
  display:inline-block; padding:4px 10px; border-radius:999px;
  background:rgba(255,255,255,0.06); border:1px solid rgba(255,255,255,0.08);
  margin-right:6px; font-size:0.85rem;
}
.seg-head {
  font-weight:700; margin-bottom:6px;
}
</style>
"""

# -----------------------------
# Transforms (match training)
# -----------------------------
TFM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

# -----------------------------
# Model loading (cached)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_model_and_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model("convnext_tiny", pretrained=False, num_classes=1)
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        if isinstance(ckpt, dict) and "model" in ckpt:
            ckpt = ckpt["model"]
        model.load_state_dict(ckpt, strict=False)
        loaded = True
    else:
        loaded = False
    model.eval().to(device).float()
    return model, device, loaded

# -----------------------------
# Optional: face crop (MediaPipe)
# -----------------------------
class FaceCropper:
    def __init__(self, conf=0.5):
        import mediapipe as mp
        self.det = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=conf
        )

    def crop(self, rgb: np.ndarray, pad:int=12) -> np.ndarray:
        h, w = rgb.shape[:2]
        res = self.det.process(rgb)
        if not res or not res.detections:
            return rgb
        d = max(res.detections, key=lambda d: d.score[0])
        bb = d.location_data.relative_bounding_box
        x1 = max(0, int(bb.xmin * w) - pad)
        y1 = max(0, int(bb.ymin * h) - pad)
        x2 = min(w, int((bb.xmin + bb.width) * w) + pad)
        y2 = min(h, int((bb.ymin + bb.height) * h) + pad)
        face = rgb[y1:y2, x1:x2]
        return face if face.size else rgb

# -----------------------------
# Video sampling
# -----------------------------
def sample_frames(path:str, target_fps:int=4, max_n:int=512) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return [], float(target_fps)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 25.0
    step = max(1, int(round(fps / max(1, target_fps))))
    frames = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i % step == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            frames.append(rgb)
            if len(frames) >= max_n:
                break
        i += 1
    cap.release()
    eff_fps = fps / step if step > 0 else target_fps
    return frames, float(eff_fps)

# -----------------------------
# Inference helpers
# -----------------------------
def p_fake_series(model, device, frames: List[np.ndarray], invert: bool, cropper: Optional[FaceCropper]) -> List[float]:
    out = []
    with torch.inference_mode():
        for rgb in frames:
            if cropper is not None:
                rgb = cropper.crop(rgb)
            x = TFM(Image.fromarray(rgb)).unsqueeze(0).to(device, dtype=torch.float32)
            p_real = torch.sigmoid(model(x)).item()
            p_fake = 1.0 - p_real if invert else p_real
            out.append(float(p_fake))
    return out

def smooth(sig: List[float], k:int=9) -> List[float]:
    k = int(max(1, k))
    if k == 1 or len(sig) == 0:
        return sig
    w = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(np.array(sig, dtype=np.float32), w, mode="same").tolist()

def find_events(pfake: List[float], thr: float, fps: float, min_sec: float) -> List[Tuple[int,int,float]]:
    if not pfake: return []
    arr = np.array(pfake, dtype=np.float32)
    above = arr >= thr
    events = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            end = i - 1
            dur = (end - start + 1) / max(1e-6, fps)
            if dur >= min_sec:
                events.append((start, end, float(arr[start:end+1].mean())))
            start = None
    if start is not None:
        end = len(arr) - 1
        dur = (end - start + 1) / max(1e-6, fps)
        if dur >= min_sec:
            events.append((start, end, float(arr[start:end+1].mean())))
    return events

# -----------------------------
# Plot
# -----------------------------
def plot_series(series: List[float], thr: float):
    fig, ax = plt.subplots(figsize=(8, 2.6), dpi=150)
    if len(series):
        ax.plot(series, linewidth=1.8)
    ax.axhline(thr, linestyle="--", linewidth=1.2, color="red")
    ax.set_ylim(0, 1)
    ax.set_ylabel("p(fake)")
    ax.set_xlabel("frame index")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Deepfake Detector", page_icon="🛡️", layout="wide")
st.markdown(BADGE_CSS, unsafe_allow_html=True)
st.title(TITLE)
st.caption("Spatial-only demo • Toggle **Invert labels** if your model outputs p(REAL).")

with st.sidebar:
    st.markdown("### Settings")
    sample_fps = st.slider("Sampling FPS", 1, 12, SAMPLE_FPS_DEFAULT, 1)
    max_frames  = st.slider("Max frames", 64, 1024, MAX_FRAMES_DEFAULT, 64)
    threshold   = st.slider("Decision threshold (p(fake) ≥ thr → FAKE)", 0.05, 0.95, THRESHOLD_DEFAULT, 0.01)
    smooth_k    = st.slider("Smoothing window (frames)", 1, 31, SMOOTH_K_DEFAULT, 2)
    invert_out  = st.toggle("Invert labels (treat model output as p(REAL))", value=True)
    use_face    = st.toggle("Face crop (MediaPipe)", value=True)

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
if uploaded is not None:
    # Temp file for OpenCV/Streamlit
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    c1, c2 = st.columns([4,1])
    with c1:
        st.video(tmp_path)
    with c2:
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        run = st.button("Analyze", type="primary")

    if run:
        model, device, loaded = load_model_and_device()
        if not loaded:
            st.warning(f"Checkpoint not found at `{CKPT_PATH}` — using random weights.")
        cropper = FaceCropper(0.5) if use_face else None

        with st.spinner("⏳ Sampling frames..."):
            frames, eff_fps = sample_frames(tmp_path, sample_fps, max_frames)
        if not frames:
            st.error("Could not read frames from the video.")
            st.stop()

        with st.spinner("🧠 Running model..."):
            pfake = p_fake_series(model, device, frames, invert_out, cropper)
            pfake_s = smooth(pfake, k=smooth_k)
            mean_p = float(np.mean(pfake_s))

        verdict_fake = (mean_p >= threshold)

        # Badge verdict
        if verdict_fake:
            st.markdown(f"<span class='badge badge-fake'>🔴 VERDICT: FAKE</span>", unsafe_allow_html=True)
        else:
            st.markdown(f"<span class='badge badge-real'>🟢 VERDICT: REAL</span>", unsafe_allow_html=True)

        # Key metrics
        mcol1, mcol2, mcol3 = st.columns(3)
        with mcol1:
            st.markdown("<div class='metric-card'>"
                        f"<div class='small'>Mean p(fake)</div>"
                        f"<div style='font-size:1.4rem;font-weight:700'>{mean_p:.3f}</div>"
                        "</div>", unsafe_allow_html=True)
        with mcol2:
            st.markdown("<div class='metric-card'>"
                        f"<div class='small'>Threshold</div>"
                        f"<div style='font-size:1.4rem;font-weight:700'>{threshold:.2f}</div>"
                        "</div>", unsafe_allow_html=True)
        with mcol3:
            st.markdown("<div class='metric-card'>"
                        f"<div class='small'>Frames • FPS(sampled)</div>"
                        f"<div style='font-size:1.4rem;font-weight:700'>{len(pfake_s)} • {eff_fps:.2f}</div>"
                        "</div>", unsafe_allow_html=True)

        st.subheader("Timeline")
        fig = plot_series(pfake_s, threshold)
        st.pyplot(fig, clear_figure=True)

        st.subheader("Evidence: segments above threshold")
        events = find_events(pfake_s, threshold, eff_fps, MIN_EVENT_SEC)
        if not events:
            st.info("No segments stayed above the threshold long enough.")
        else:
            for i, (s, e, mpf) in enumerate(events, 1):
                mid = (s + e) // 2
                frame = frames[min(max(0, mid), len(frames)-1)]
                a, b = st.columns([1,3])
                with a:
                    st.image(frame, caption=f"Segment {i}: frames {s}–{e} (mean p(fake)={mpf:.2f})")
                with b:
                    hints = []
                    seg = np.array(pfake_s[s:e+1], dtype=np.float32)
                    if len(seg) >= 3 and (seg[-1] - seg[0]) > 0.15:
                        hints.append("Rising probability within this segment.")
                    if seg.mean() - mean_p > 0.10:
                        hints.append("Segment’s mean p(fake) ≫ overall mean.")
                    if use_face:
                        hints.append("Face ROI used to reduce background bias.")
                    if not hints:
                        hints.append("Above-threshold region—inspect lips/skin consistency.")
                    st.markdown("<div class='seg-head'>Why flagged</div>" +
                                "<div>• " + "<br>• ".join(hints) + "</div>", unsafe_allow_html=True)

        # Downloadable JSON report
        report = {
            "video": uploaded.name,
            "path": tmp_path,
            "mapping": "p(fake) = 1 - p(real)" if invert_out else "p(fake) = model_output",
            "threshold": threshold,
            "mean_p_fake": mean_p,
            "frames_analyzed": len(pfake_s),
            "fps_sampled": eff_fps,
            "face_crop": bool(use_face),
            "events": [{"start_frame": int(s), "end_frame": int(e), "mean_p": float(m)} for (s,e,m) in events],
            "series": [float(x) for x in pfake_s]
        }
        st.download_button(
            "⬇️ Download analysis (JSON)",
            data=json.dumps(report, indent=2),
            file_name=f"{os.path.splitext(uploaded.name)[0]}_report.json",
            mime="application/json"
        )

else:
    st.info("Upload an MP4/MOV/AVI/MKV, then click **Analyze**. "
            "If your model shows p(REAL), keep **Invert labels** ON (default).")
