import cv2
import pickle
import numpy as np
import insightface
import datetime
import os
import time
from pathlib import Path

from modules.esp32_cam_stream import ESP32Cam


# CONFIG
DB_PATH = "./data/embeddings/embeddings.pkl"

BASE_DIR = Path(__file__).resolve().parent.parent

ATTENDANCE_DIR = BASE_DIR / "monitor"
ATTENDANCE_DIR.mkdir(exist_ok=True)

today = datetime.date.today()

CSV_PATH = ATTENDANCE_DIR / f"attendance_{today}.csv"

THRESHOLD = 0.55

FRAME_SKIP = 8


with open(DB_PATH, "rb") as f:
    db = pickle.load(f)

attendance = set()

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as f:
        f.write("timestamp,name\n")


app = insightface.app.FaceAnalysis()

app.prepare(
    ctx_id=-1,          # CPU
    det_size=(320, 320)
)



# ESP32 CAMERA INIT
cam = ESP32Cam()



# HELPER FUNCTIONS
def recognize_face(new_emb):

    best_name = "Unknown"
    best_score = -1

    for name, emb in db.items():

        sim = np.dot(new_emb, emb)

        if sim > best_score:
            best_score = sim
            best_name = name

    if best_score < THRESHOLD:
        best_name = "Unknown"

    return best_name, best_score


def mark_attendance(name):

    if name != "Unknown" and name not in attendance:

        attendance.add(name)

        with open(CSV_PATH, "a") as f:
            f.write(f"{datetime.datetime.now()},{name}\n")

        print(f"[ATTENDANCE] {name}")





# MAIN LOOP
frame_count = 0

while True:

    ret, frame = cam.read()

    if not ret:
        time.sleep(0.001)
        continue

    frame = cv2.resize(frame, (700, 500))

    frame_count += 1

    if frame_count % FRAME_SKIP == 0:

        # smaller frame for faster inference
        small_frame = cv2.resize(frame, (320, 240))

        faces = app.get(small_frame)

        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240

        for face in faces:

            x1, y1, x2, y2 = face.bbox

            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)

            emb = face.embedding
            emb = emb / np.linalg.norm(emb)

            name, score = recognize_face(emb)

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            cv2.putText(
                frame,
                f"{name} {score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            mark_attendance(name)

    cv2.putText(
        frame,
        f"Present Count: {len(attendance)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )


    y_offset = 70

    for person in attendance:

        cv2.putText(
            frame,
            person,
            (10, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        y_offset += 30

    cv2.imshow("AI Attendance Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break



cam.release()

cv2.destroyAllWindows()

print(f"Saved -> {CSV_PATH}")