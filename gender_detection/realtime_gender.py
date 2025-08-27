import cv2
import numpy as np

# Load gender model
model = 'gender_net.caffemodel'
proto = 'gender_deploy.prototxt'
gender_net = cv2.dnn.readNetFromCaffe(proto, model)

# Gender labels
genders = ['Male', 'Female']

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)  # Use 0 for default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w].copy()

        # Preprocess face for gender model
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
                                     (78.426, 87.768, 114.895), swapRB=False)
        gender_net.setInput(blob)
        preds = gender_net.forward()
        gender = genders[preds[0].argmax()]

        # Draw results
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 0, 0), 2)

    # Display the output
    cv2.imshow("Real-time Gender Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
