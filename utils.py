import cv2

def draw_custom_muscle_lines(frame, points, POSE_PAIRS):
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (0, 255, 0), 2)