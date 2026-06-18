import cv2

# On-device lightweight ML model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start laptop webcam (this is your edge device)
cap = cv2.VideoCapture(0)

print("Running ON-EDGE AI... Press 'q' to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # On-device face detection
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw boxes on faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("ON-DEVICE EDGE AI (Laptop Webcam)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
