import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import time
from datetime import datetime
from collections import deque
import csv
import threading

# Dash相关
import dash
from dash import dcc
from dash import html
import plotly.express as px

# Initialize YOLOv8 model
model = YOLO('yolov8n.pt')

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Open video file
video_path = "football.mp4"
cap = cv2.VideoCapture(video_path)

# Get video FPS and total frames
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Calculate frame numbers for minutes 35-36
start_frame = int(35 * 60 * fps)  # Starting frame for minute 35
end_frame = int(36 * 60 * fps)  # Ending frame for minute 36

# Set video starting position
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Ball possession parameters
POSSESSION_THRESHOLD = 0.5  # Time threshold for ball possession (seconds)
BALL_DETECTION_THRESHOLD = 0.3  # Lower threshold for ball detection
IOU_THRESHOLD = 0.15  # IOU threshold for possession

# Performance optimization parameters
FRAME_SKIP = 2  # Process every nth frame
RESIZE_FACTOR = 0.5  # Resize factor for input frames

# Ball tracking parameters
MAX_BALL_HISTORY = 10  # Number of frames to keep ball position history
ball_positions = deque(maxlen=MAX_BALL_HISTORY)  # Store recent ball positions
last_valid_ball = None  # Store last valid ball detection
ball_missing_frames = 0  # Counter for consecutive frames with missing ball
MAX_MISSING_FRAMES = 5  # Maximum frames to predict ball position

# Initialize CSV file for results
csv_filename = f"football_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
csv_headers = ['Time', 'Frame', 'Player_ID', 'Has_Ball', 'Ball_Position', 'Player_Position', 'Ball_Confidence']
with open(csv_filename, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

# ========== Global Variables ==========
player_data = pd.DataFrame(columns=["player_id", "x_position", "y_position", "possession_count", "sprint_speed"])
possession_counter = {}  # Count the number of possessions for each player

# ========== Dash App ==========
app = dash.Dash(__name__)
app.layout = html.Div([
    html.H1("GAA Player Real-Time Possession Analysis"),
    dcc.Graph(id="live-graph"),
    dcc.Interval(id="interval-component", interval=2000, n_intervals=0)  # Refresh every 2 seconds
])

@app.callback(
    dash.dependencies.Output("live-graph", "figure"),
    [dash.dependencies.Input("interval-component", "n_intervals")]
)
def update_graph_live(n):
    if player_data.empty:
        return px.scatter().update_layout(template="plotly_dark")
    fig = px.scatter(
        data_frame=player_data,
        x="x_position",
        y="y_position",
        color="possession_count",
        size="sprint_speed",
        hover_data=["player_id"]
    ).update_layout(template="plotly_dark")
    return fig

def run_dash():
    app.run(debug=False, use_reloader=False)

def enhance_ball_features(frame):
    """Enhance image features to improve ball detection"""
    # Convert to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge channels
    enhanced = cv2.merge([cl, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    return enhanced

def predict_ball_position():
    """Predict ball position based on recent history"""
    if len(ball_positions) < 2:
        return last_valid_ball

    # Calculate average velocity from recent positions
    velocities = []
    for i in range(1, len(ball_positions)):
        prev = ball_positions[i - 1]
        curr = ball_positions[i]
        if prev is not None and curr is not None:
            vx = (curr[2] + curr[0]) / 2 - (prev[2] + prev[0]) / 2
            vy = (curr[3] + curr[1]) / 2 - (prev[3] + prev[1]) / 2
            velocities.append((vx, vy))

    if not velocities:
        return last_valid_ball

    # Use average velocity to predict next position
    avg_vx = sum(v[0] for v in velocities) / len(velocities)
    avg_vy = sum(v[1] for v in velocities) / len(velocities)

    last_pos = ball_positions[-1]
    if last_pos is None:
        return last_valid_ball

    predicted = [
        last_pos[0] + avg_vx,
        last_pos[1] + avg_vy,
        last_pos[2] + avg_vx,
        last_pos[3] + avg_vy
    ]

    return predicted

def calculate_iou(box1, box2):
    """Calculate IoU (Intersection over Union) of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection)

def video_analysis():
    global player_data, possession_counter

    # Initialize model and video
    model = YOLO('yolov8n.pt')
    tracker = DeepSort(max_age=30)
    video_path = "football.mp4"
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(35 * 60 * fps)
    end_frame = int(36 * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    POSSESSION_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.15
    FRAME_SKIP = 2
    RESIZE_FACTOR = 0.5
    possession_start = {}

    current_frame = start_frame

    while cap.isOpened() and current_frame < end_frame:
        success, frame = cap.read()
        if not success:
            break

        if current_frame % FRAME_SKIP != 0:
            current_frame += 1
            continue

        height, width = frame.shape[:2]
        resized_frame = cv2.resize(frame, (int(width * RESIZE_FACTOR), int(height * RESIZE_FACTOR)))

        results = model(resized_frame)
        ball_box = None
        player_detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() / RESIZE_FACTOR
                conf = box.conf[0].cpu().numpy()
                cls = box.cls[0].cpu().numpy()
                if cls == 0:  # Person
                    player_detections.append(([x1, y1, x2, y2], conf, cls))
                elif cls == 32:  # Ball
                    ball_box = [x1, y1, x2, y2]

        frame_player_data = []
        if player_detections:
            tracks = tracker.update_tracks(player_detections, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                ltrb = track.to_ltrb()
                x_center = (ltrb[0] + ltrb[2]) / 2
                y_center = (ltrb[1] + ltrb[3]) / 2

                # Possession detection
                has_ball = False
                if ball_box is not None:
                    iou = calculate_iou(ball_box, ltrb)
                    current_time = time.time()
                    if iou > IOU_THRESHOLD:
                        if track_id not in possession_start:
                            possession_start[track_id] = current_time
                        elif current_time - possession_start[track_id] >= POSSESSION_THRESHOLD:
                            has_ball = True
                            possession_counter[track_id] = possession_counter.get(track_id, 0) + 1
                    else:
                        possession_start.pop(track_id, None)

                frame_player_data.append({
                    "player_id": track_id,
                    "x_position": x_center,
                    "y_position": y_center,
                    "possession_count": possession_counter.get(track_id, 0),
                    "sprint_speed": 1  # Replace with actual speed if available
                })

        # Update global DataFrame
        if frame_player_data:
            player_data = pd.DataFrame(frame_player_data)

        current_frame += 1

    cap.release()

# ========== Start Analysis and Dash ==========
if __name__ == "__main__":
    dash_thread = threading.Thread(target=run_dash)
    dash_thread.daemon = True
    dash_thread.start()
    video_analysis()