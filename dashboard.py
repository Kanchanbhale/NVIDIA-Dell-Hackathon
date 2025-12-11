import streamlit as st
import cv2
import time
from ultralytics import YOLO
import numpy as np
import requests
import json

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "runs/detect/train/weights/best.pt"

st.set_page_config(
    page_title="Fall-Risk AI System",
    layout="wide",
)

# Load YOLO model
model = YOLO(MODEL_PATH)


# -----------------------------
# NVIDIA NIM LLM SETTINGS
# -----------------------------
NVIDIA_API_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_MODEL = "meta/llama-3.1-8b-instruct"

# Store API key in Streamlit secrets (recommended)
NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]

SYSTEM_PROMPT = """
You are a real-time medical fall-risk agent.
You receive structured events from a computer vision system.

Your goals:
1. Determine if this is a real fall.
2. Classify severity: none, low, medium, high, emergency.
3. Decide an action: monitor, call_family, call_911.
4. Output ONLY the short UI message the dashboard should display.
"""


def call_nvidia_agent(event_dict):
    """
    Sends structured CV events to NVIDIA LLM agent for reasoning.
    Returns a short UI summary string.
    """

    user_message = f"""
CV Event:
{json.dumps(event_dict, indent=2)}

Based on timeline, severity, and fall behavior,
return ONE short sentence suitable for a safety dashboard.
"""

    headers = {
        "Authorization": f"Bearer {NVIDIA_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": NVIDIA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]
    }

    try:
        response = requests.post(NVIDIA_API_URL, headers=headers, json=payload)
        data = response.json()

        return data["choices"][0]["message"]["content"]

    except Exception as e:
        return f"⚠ LLM error: {str(e)}"


# -----------------------------
# Session State
# -----------------------------
if "running" not in st.session_state:
    st.session_state.running = False

if "alert_message" not in st.session_state:
    st.session_state.alert_message = "No events yet."

if "fall_start_time" not in st.session_state:
    st.session_state.fall_start_time = None


# -----------------------------
# Sidebar Controls
# -----------------------------
st.sidebar.header("Control Panel")

video_path = st.sidebar.text_input(
    "Video file path",
    value="data/raw/le2i/Home_01/Home_01/Videos/video (8).avi"
)

start_btn = st.sidebar.button("▶ Start Video")
stop_btn = st.sidebar.button("⏹ Stop Video")

if start_btn:
    st.session_state.running = True
if stop_btn:
    st.session_state.running = False


# -----------------------------
# Main Layout
# -----------------------------
st.title("Fall-Risk AI System")
st.caption("Real-time fall detection powered by YOLO + NVIDIA LLM agent reasoning.")

left, right = st.columns([3, 1])

with left:
    st.subheader("📺 Live Detection Feed")
    video_placeholder = st.empty()

with right:
    st.subheader("⚡ Real-Time Agent Summary")
    agent_box = st.empty()


# -----------------------------
# NEW AGENTIC AI REASONING (LLM)
# -----------------------------
def agent_reasoning_llm(fall_detected: bool, confidence: float):
    """
    Replaces the old rule-based system with NVIDIA LLM agent.
    """

    # Reset timer if no fall detected
    if not fall_detected:
        st.session_state.fall_start_time = None
        elapsed = 0
    else:
        # Start timer on first fall
        if st.session_state.fall_start_time is None:
            st.session_state.fall_start_time = time.time()
        elapsed = time.time() - st.session_state.fall_start_time

    # Build structured event for the LLM agent
    event = {
        "fall_detected": fall_detected,
        "confidence": round(confidence, 3),
        "elapsed_time_sec": round(elapsed, 2)
    }

    # Send to NVIDIA LLM agent
    llm_result = call_nvidia_agent(event)

    return llm_result


# -----------------------------
# VIDEO PROCESSING LOOP
# -----------------------------
def run_video():
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        agent_box.error("❌ Cannot open video file.")
        st.session_state.running = False
        return

    while st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            agent_box.warning("Video ended.")
            break

        # YOLO inference
        results = model(frame, verbose=False)[0]

        annotated = results.plot()

        # Detect fall class
        fall_detected = False
        fall_conf = 0.0

        if results.boxes is not None and len(results.boxes.cls) > 0:
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs = results.boxes.conf.cpu().numpy()

            if 0 in classes:
                fall_detected = True
                # take highest confidence
                fall_conf = float(np.max(confs))

        # LLM agent reasoning
        summary = agent_reasoning_llm(fall_detected, fall_conf)
        st.session_state.alert_message = summary

        # Display video feed
        video_placeholder.image(annotated, channels="BGR")

        # Display LLM message with severity coloring
        msg_lower = summary.lower()

        if "911" in msg_lower or "emergency" in msg_lower:
            agent_box.error(summary)
        elif "high" in msg_lower or "risk" in msg_lower:
            agent_box.warning(summary)
        else:
            agent_box.info(summary)

        time.sleep(0.03)

    cap.release()


# -----------------------------
# RUN LOOP
# -----------------------------
if st.session_state.running:
    run_video()
else:
    agent_box.info(st.session_state.alert_message)
    video_placeholder.info("Click ▶ Start Video to begin.")
