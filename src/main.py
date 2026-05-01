import cv2
import pickle
import numpy as np
import insightface
import datetime
import os


DB_PATH = "../data/embeddings/embeddings.pkl"
ATTENDANCE_DIR = "attendance"
os.makedirs(ATTENDANCE_DIR, exist_ok=True)
today = datetime.date.today()
CSV_PATH = os.path.join(ATTENDANCE_DIR, f"attendance_{today}.csv")


with open(DB_PATH, "rb") as f:
    db = pickle.load(f)


app = insightface.app.FaceAnalysis()
app.prepare(ctx_id=-1)  # CPU


THRESHOLD = 0.65

def recognize_face(new_emb, db, threshold=THRESHOLD):
    best_name = "Unknown"
    best_score = -1

    for name, emb in db.items():
        sim = np.dot(new_emb, emb)  # cosine similarity
        if sim > best_score:
            best_score = sim
            best_name = name

    if best_score < threshold:
        best_name = "Unknown"

    return best_name, best_score


attendance = set()  # to avoid double-counting

def mark_attendance(name):
    if name != "Unknown" and name not in attendance:
        attendance.add(name)
        # Append to CSV
        with open(CSV_PATH, "a") as f:
            f.write(f"{datetime.datetime.now()},{name}\n")
        print(f"[ATTENDANCE] {name} marked present!")


stream_url = "http://192.168.76.129:81/stream"

cap = cv2.VideoCapture(stream_url)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        embedding = face.embedding / np.linalg.norm(face.embedding)
        name, score = recognize_face(embedding, db)


        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)


        mark_attendance(name)

    # Show total students detected
    cv2.putText(frame, f"Total Present: {len(attendance)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Classroom Attendance", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break












cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Attendance saved to {CSV_PATH}")