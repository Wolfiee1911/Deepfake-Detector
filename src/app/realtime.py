# src/app/realtime.py
import cv2, torch, numpy as np
from src.vision.spatial_model import load_spatial_model

cap = cv2.VideoCapture(0)  # try 1 if 0 doesn't work
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = load_spatial_model(device)
alpha, p_smooth = 0.8, 0.5

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # BGR -> RGB, resize, make contiguous to avoid negative-stride issues
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224)).copy()

    # To tensor (FP32)
    x = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()
    x = x.to(device, dtype=torch.float32) / 255.0   # <-- FP32 input

    # (debug once) assert dtypes
    # print("x dtype:", x.dtype, "model dtype:",
    #       next(model.parameters()).dtype, flush=True)

    with torch.inference_mode():
        logits = model(x)                 # model is FP32, input FP32
        prob = torch.sigmoid(logits).item()

    p_smooth = alpha * p_smooth + (1 - alpha) * prob
    txt = f"Fake prob: {p_smooth:.2f}"
    cv2.putText(frame, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Deepfake Monitor', frame)

    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
