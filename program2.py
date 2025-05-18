import streamlit as st
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw
import io

st.set_page_config(page_title="YOLO AutoClicker Simulator", layout="centered")
st.title("YOLO AutoClicker Simulator")

# Load YOLO model once
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Ensure best.pt is in the repo or path

model = load_model()

# Session state initialization
if 'running' not in st.session_state:
    st.session_state.running = False
if 'log' not in st.session_state:
    st.session_state.log = []
if 'last_click_time' not in st.session_state:
    st.session_state.last_click_time = 0
if 'detections' not in st.session_state:
    st.session_state.detections = []

# Play/Pause buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Play"):
        st.session_state.running = True
with col2:
    if st.button("Pause"):
        st.session_state.running = False

# Image uploader
uploaded_file = st.file_uploader("Upload an image for detection", type=["jpg", "jpeg", "png"])
image = None

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLO detection
    results = model(image)
    boxes = results[0].boxes
    detected = []

    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        detected.append({
            'class': cls_name,
            'box': (x1, y1, x2, y2),
            'center': (cx, cy)
        })

    st.session_state.detections = detected

    # Draw detections
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    for d in detected:
        draw.rectangle(d['box'], outline="red", width=3)
        draw.text((d['box'][0], d['box'][1] - 10), d['class'], fill="red")
    st.image(draw_image, caption="Detections", use_column_width=True)

    # Clickable buttons for each detection
    st.subheader("Click Detected Items")
    for idx, d in enumerate(detected):
        if st.button(f"Click: {d['class']} ({int(d['center'][0])}, {int(d['center'][1])})", key=f"manual_{idx}"):
            st.session_state.log.append(f"Manual click on {d['class']} at {time.strftime('%X')}")

# Auto-clicking logic
AUTO_CLICK_INTERVAL = 3  # seconds
current_time = time.time()

if st.session_state.running and st.session_state.detections:
    if current_time - st.session_state.last_click_time >= AUTO_CLICK_INTERVAL:
        auto_target = st.session_state.detections[0]
        st.session_state.log.append(f"Auto-clicked {auto_target['class']} at {time.strftime('%X')}")
        st.session_state.last_click_time = current_time
        st.experimental_rerun()

# Log
st.subheader("Click Log")
if st.session_state.log:
    for entry in reversed(st.session_state.log[-20:]):
        st.text(entry)
else:
    st.write("No clicks recorded yet.")
