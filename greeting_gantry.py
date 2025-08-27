#  greeting_gantry.py — Main program
# python
# Copy code
import cv2
import time
import mediapipe as mp
from utils import get_random_animation

# Init pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
draw = mp.solutions.drawing_utils

# Check for person in frame
def detect_person(frame, pose):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)
    return results.pose_landmarks is not None

# Play looping animation until person leaves
def play_animation_while_person_detected(video_path, cam_check, pose):
    print(f"🎞️ Playing: {video_path}")
    anim = cv2.VideoCapture(video_path)

    while True:
        ret, frame = anim.read()
        if not ret:
            anim.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = cv2.resize(frame, (800, 600))
        cv2.imshow("Interactive Character", frame)

        # Check if person still in frame
        ret2, cam_frame = cam_check.read()
        if not ret2:
            break

        person_still_here = detect_person(cam_frame, pose)
        if not person_still_here:
            break

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    anim.release()
    print("👋 Person left — back to idle mode.")

# Main loop
cam_check = cv2.VideoCapture(0)
person_present = False

while True:
    ret, frame = cam_check.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    person_detected = detect_person(frame, pose)

    # If new person enters
    if person_detected and not person_present:
        person_present = True
        animation = get_random_animation()
        if animation:
            cam_check.release()
            cv2.destroyAllWindows()
            cam_check = cv2.VideoCapture(0)
            play_animation_while_person_detected(animation, cam_check, pose)

    # Idle mode display
    if not person_detected:
        person_present = False

    cv2.putText(frame, "👋 Waiting for visitors...", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow("Idle Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam_check.release()
cv2.destroyAllWindows()