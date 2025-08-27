import gradio as gr
import numpy as np

def demo_fn(n):
    n = int(n)
    xs = np.arange(n, dtype=np.int32)
    ys = (np.sin(xs/5.0)+1.0)/2.0
    # Plot as image (HxWx3 uint8)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    import cv2

    fig, ax = plt.subplots(figsize=(5,3))
    ax.plot(xs, ys)
    ax.set_title("Sanity Plot")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    import numpy as np
    img = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mean_y = float(ys.mean())
    payload = {"values": [float(v) for v in ys.tolist()]}
    return img, mean_y, payload

with gr.Blocks() as demo:
    n = gr.Slider(10, 200, value=50, step=10, label="N")
    btn = gr.Button("Render")
    out_img = gr.Image(label="Plot image", type="numpy")
    out_num = gr.Number(label="Mean", precision=4)
    out_json = gr.JSON(label="Payload")
    btn.click(demo_fn, [n], [out_img, out_num, out_json])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
