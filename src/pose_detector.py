# pose_detector.py

import cv2
import mediapipe as mp
from angle_utils import calculate_angle

# 1. init mediapipe pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. open video
cap = cv2.VideoCapture("data/test_video1.mp4")
# before while loop — add this after cap = cv2.VideoCapture(...)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("data/output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# 3. loop frames
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 4. convert BGR → RGB (mediapipe needs RGB)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 5. process frame
    results = pose.process(rgb)
    
    # 6. extract landmarks if detected
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # get hip, knee, ankle coords as [x, y]
        hip   = [landmarks[23].x, landmarks[23].y]
        knee  = [landmarks[25].x, landmarks[25].y]
        ankle = [landmarks[27].x, landmarks[27].y]
        
        # calculate angle
        angle = calculate_angle(hip, knee, ankle)
        
        # print angle
        print(angle)
        
        # draw landmarks on frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()
cap.release()
cv2.destroyAllWindows()