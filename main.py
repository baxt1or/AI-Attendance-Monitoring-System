import cv2

stream_url = "http://192.168.76.129:81/stream"

cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("Cannot open stream")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("ESP32 Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()