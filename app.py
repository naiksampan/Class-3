import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import os
import tempfile
import zipfile
from pathlib import Path

st.set_page_config(page_title="YOLOv8 Landmark Inference", layout="wide")

st.title("YOLOv8 — Image Inference with Landmarks")
st.write("Upload images or provide a local folder path. The app will run YOLOv8 inference and display annotated results (including landmarks/keypoints if present).")

# --- Sidebar: model path and options ---
st.sidebar.header("Model & Inference Options")
model_path = st.sidebar.text_input("Path to YOLOv8 weights (best.pt)", value="/content/drive/MyDrive/Sanket/Recent_data/runs2/weights/best.pt")
conf_thres = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.01)
iou_thres = st.sidebar.slider("IOU threshold (NMS)", 0.0, 1.0, 0.45, 0.01)
max_det = st.sidebar.number_input("Max detections per image", min_value=1, max_value=1000, value=100)
save_output = st.sidebar.checkbox("Save annotated outputs to folder", value=True)
output_folder = st.sidebar.text_input("Output folder (if saving)", value="./yolo_outputs")
run_button = st.sidebar.button("Load model & run inference")

# --- Helper: cached model loader ---
@st.cache_resource(show_spinner=False)
def load_model(path):
    try:
        model = YOLO(path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- Input: upload files OR local folder ---
st.subheader("Input images")
col1, col2 = st.columns([2,1])
with col1:
    uploaded_files = st.file_uploader("Upload image files (PNG/JPG). You can select multiple.", type=["png","jpg","jpeg"], accept_multiple_files=True)
    local_folder = st.text_input("Or enter a local folder path containing images (leave blank if using upload)")
with col2:
    st.markdown("### Quick tips")
    st.markdown("- If using Google Drive/Colab paths, ensure runtime has access.")
    st.markdown("- Model must be compatible with your keypoint/head outputs if you want landmarks to appear.")
    st.markdown("- You can adjust thresholds in the sidebar.")

# --- Prepare list of image sources ---
image_entries = []  # list of tuples: (source_name, numpy_image, source_type)
# Uploaded
if uploaded_files:
    for f in uploaded_files:
        file_bytes = f.read()
        img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        np_img = np.array(img)[:, :, ::-1]  # PIL->BGR for cv2/Ultralytics plotting compatibility
        image_entries.append((f.name, np_img, "upload"))

# Local folder
if local_folder:
    p = Path(local_folder)
    if p.exists() and p.is_dir():
        for img_path in sorted(p.glob("*")):
            if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                np_img = cv2.imread(str(img_path))
                if np_img is None:
                    continue
                image_entries.append((img_path.name, np_img, "local"))
    else:
        st.warning("Provided local folder path does not exist or is not a folder.")

if len(image_entries) == 0:
    st.info("No images provided yet. Upload files or enter a folder path and click 'Load model & run inference' in the sidebar.")
    
# --- Run inference when button pressed ---
if run_button and len(image_entries) > 0:
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    if model is None:
        st.stop()

    # ensure output folder exists if saving
    if save_output:
        os.makedirs(output_folder, exist_ok=True)

    results_for_display = []  # list of (name, annotated_image_bytes)

    # Run predictions image-by-image to keep memory manageable and to show progress
    progress_bar = st.progress(0)
    total = len(image_entries)
    for idx, (name, np_img, stype) in enumerate(image_entries):
        progress_bar.progress(int((idx/total)*100))
        # YOLO predict accepts path, array, or list. We'll pass the numpy image (BGR).
        try:
            # The predict call - set conf and other args
            results = model.predict(
                source=np_img,
                conf=conf_thres,
                iou=iou_thres,
                max_det=int(max_det),
                save=False,
                verbose=False,
                show=False
            )
        except Exception as e:
            st.error(f"Inference failed on {name}: {e}")
            continue

        # results is a list (one element per image)
        if len(results) == 0:
            st.warning(f"No results for {name}")
            annotated_bgr = np_img
        else:
            r = results[0]
            # r.plot returns numpy array in RGB by default in some ulralytics versions; user used kpt_line=False -> do the same
            try:
                annotated = r.plot(kpt_line=False)  # returns numpy array (usually RGB)
            except Exception:
                # fallback: convert r.orig_img and draw via r.boxes / r.keypoints if needed
                # but try simple approach: use original image
                annotated = r.orig_img if hasattr(r, "orig_img") else np_img

            # r.plot likely returns RGB; convert to BGR for cv2.imwrite if needed
            if annotated is None:
                annotated_bgr = np_img
            else:
                # If shape matches and dtype is uint8, check channel order by sampling (we'll assume it's RGB from ultralytics)
                if annotated.shape[2] == 3:
                    # convert RGB -> BGR for cv2 saving consistency
                    annotated_bgr = annotated[:, :, ::-1]
                else:
                    annotated_bgr = annotated

        # Convert annotated image to displayable bytes (PNG)
        annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB for PIL
        pil_img = Image.fromarray(annotated_rgb)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf_bytes = buf.getvalue()

        results_for_display.append((name, buf_bytes))

        # Save to disk if requested
        if save_output:
            out_path = os.path.join(output_folder, f"annotated_{name}")
            # ensure extension
            if not Path(out_path).suffix:
                out_path = out_path + ".png"
            try:
                with open(out_path, "wb") as out_f:
                    out_f.write(buf_bytes)
            except Exception as e:
                st.error(f"Failed to save {out_path}: {e}")

    progress_bar.progress(100)
    st.success("Inference complete.")

    # --- Display results in grid ---
    st.subheader("Results")
    cols = st.columns(3)
    for i, (name, img_bytes) in enumerate(results_for_display):
        col = cols[i % 3]
        with col:
            st.image(img_bytes, caption=name, use_column_width=True)
            # Download button for individual image
            st.download_button(label="Download", data=img_bytes, file_name=f"annotated_{name}", mime="image/png")

    # Option to download all results as zip
    if save_output:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w") as zf:
            for name, img_bytes in results_for_display:
                arc_name = f"annotated_{name}"
                if not Path(arc_name).suffix:
                    arc_name = arc_name + ".png"
                zf.writestr(arc_name, img_bytes)
        zip_buffer.seek(0)
        st.download_button("Download all results (ZIP)", data=zip_buffer, file_name="yolo_annotated_results.zip", mime="application/zip")

else:
    # If run button not pressed but images present, show sample previews
    if len(image_entries) > 0:
        st.subheader("Preview of provided images")
        preview_cols = st.columns(3)
        for i, (name, np_img, _) in enumerate(image_entries):
            img_rgb = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            col = preview_cols[i % 3]
            with col:
                st.image(pil_img, caption=name, use_column_width=True)

st.markdown("---")
st.caption("Built with YOLO (ultralytics) and Streamlit. For large batches consider running inference on GPU-enabled environment for speed.")
