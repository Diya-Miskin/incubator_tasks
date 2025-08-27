import cv2
import numpy as np

# Load Haar Cascade for face detection (comes with OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_average_color(region):
    if region.size == 0:
        return (0, 0, 0)
    avg_color = cv2.mean(region)[:3]
    return tuple(map(int, avg_color))

def classify_skin_tone(bgr):
    hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]
    v = hsv[2]
    if v < 60:
        return "Dark"
    elif v < 160:
        return "Medium"
    else:
        return "Fair"

def classify_hair_color(bgr):
    b, g, r = bgr
    if r > 100 and g < 80 and b < 80:
        return "Red"
    elif b < 60 and g < 60 and r < 60:
        return "Black"
    elif r > 150 and g > 120 and b > 100:
        return "Blonde"
    elif r > 90 and g > 60 and b > 40:
        return "Brown"
    else:
        return "Dark"

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Face region
        face = frame[y:y+h, x:x+w]

        # Skin tone from center of face
        skin_region = face[h//3:h//2, w//3:2*w//3]
        skin_color = get_average_color(skin_region)
        skin_tone = classify_skin_tone(np.array(skin_color, dtype=np.uint8))

        # Hair color from region above forehead
        top_y = max(y - h//4, 0)
        hair_region = frame[top_y:y, x:x+w]
        hair_color = get_average_color(hair_region)
        hair_label = classify_hair_color(np.array(hair_color, dtype=np.uint8))

        # Draw rectangle and labels
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Skin: {skin_tone}", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Hair: {hair_label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Show frame
    cv2.imshow("Hair and Skin Tone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
