import cv2
import pickle
import numpy as np
import insightface
import datetime
import os
import threading
import time
from pathlib import Path

# ==========================================
# CONFIG
# ==========================================
DB_PATH = "../data/embeddings/embeddings.pkl"



BASE_DIR = Path(__file__).resolve().parent.parent  
ATTENDANCE_DIR = BASE_DIR / "monitor"
ATTENDANCE_DIR.mkdir(exist_ok=True)
today = datetime.date.today()
CSV_PATH = ATTENDANCE_DIR / f"attendance_{today}.csv"






url = '192.168.134.129'

STREAM_URL = f"http://{url}:81/stream"
THRESHOLD = 0.55

latest_frame = None
running = True


# ==========================================
# LOAD DB
# ==========================================
with open(DB_PATH, "rb") as f:
    db = pickle.load(f)

attendance = set()

if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w") as f:
        f.write("timestamp,name\n")


# ==========================================
# INSIGHTFACE INIT
# ==========================================
app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1, det_size=(320,320))   # much faster


# ==========================================
# FUNCTIONS
# ==========================================
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



def camera_reader():
    global latest_frame, running

    cap = cv2.VideoCapture(0)

    while running:
        ret, frame = cap.read()

        if ret:
            latest_frame = frame
        else:
            print("Reconnecting stream...")
            cap.release()
            time.sleep(1)
            cap = cv2.VideoCapture(0)

    cap.release()


threading.Thread(target=camera_reader, daemon=True).start()


frame_count = 0

while True:
    if latest_frame is None:
        continue

    frame = latest_frame.copy()
    frame = cv2.resize(frame, (700, 500))

    frame_count += 1

    # only process every 8th frame
    if frame_count % 8 == 0:
        faces = app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)

            emb = face.embedding / np.linalg.norm(face.embedding)
            name, score = recognize_face(emb)

            color = (0,255,0) if name != "Unknown" else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f"{name} {score:.2f}",
                        (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, color, 2)

            mark_attendance(name)


    cv2.putText(frame,
                f"Present Count: {len(attendance)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    # display names
    y_offset = 70

    for person in attendance:
        cv2.putText(frame,
                    person,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2)

        y_offset += 30

    cv2.imshow("AI Attendance Monitor", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        running = False
        break


cv2.destroyAllWindows()
print(f"Saved -> {CSV_PATH}")