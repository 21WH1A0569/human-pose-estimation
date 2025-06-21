import cv2
import numpy as np
import os
from tkinter import Tk, filedialog, messagebox
from utils import draw_custom_muscle_lines

# Pose model configuration
protoFile = "models/pose_deploy_linevec.prototxt"
weightsFile = "models/pose_iter_440000.caffemodel"

nPoints = 18
POSE_PAIRS = [
    [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
    [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
    [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
]



def select_image_and_run():
    root = Tk()
    root.withdraw()
    image_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )

    if not image_path:
        messagebox.showerror("Error", "No image selected!")
        return

    frame = cv2.imread(image_path)
    if frame is None:
        messagebox.showerror("Error", "Failed to load image!")
        return

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368),
                                    (0, 0, 0), swapRB=False, crop=False)
    net.setInput(inpBlob)
    output = net.forward()

    H = output.shape[2]
    W = output.shape[3]

    points = []
    for i in range(nPoints):
        probMap = output[0, i, :, :]
        _, prob, _, point = cv2.minMaxLoc(probMap)

        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(frame, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1, cv2.LINE_AA)
            points.append((int(x), int(y)))
        else:
            points.append(None)

    draw_custom_muscle_lines(frame, points, POSE_PAIRS)

    # Determine if all keypoints are detected
    if all(p is not None for p in points):
        message = "All keypoints were detected "
        color = (0, 128, 0)  # Green
    else:
        message = "All keypoints were NOT detected "
        color = (0, 0, 255)  # Red

    # Prepare background canvas
    canvas_height = frameHeight + 200
    canvas_width = frameWidth + 400
    background = np.full((canvas_height, canvas_width, 3), (230, 255, 240), dtype=np.uint8)

    # Center the frame
    y_offset = (canvas_height - frameHeight - 50) // 2
    x_offset = (canvas_width - frameWidth) // 2
    background[y_offset:y_offset + frameHeight, x_offset:x_offset + frameWidth] = frame

    # Put message below the image (centered horizontally)
    text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    text_x = (canvas_width - text_size[0]) // 2
    text_y = y_offset + frameHeight + 30
    cv2.putText(background, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, color, 2, cv2.LINE_AA)


    # Save and show
    os.makedirs("output", exist_ok=True)
    output_path = os.path.join("output", "skeleton.jpg")
    cv2.imwrite(output_path, background)

    screen_width = 10000  # or get actual screen resolution dynamically
    scale = screen_width / background.shape[1]
    display_height = int(background.shape[0] * scale)
    resized_background = cv2.resize(background, (screen_width, display_height))

    cv2.namedWindow("Centered Pose Estimation Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Centered Pose Estimation Output", 1000, 800)  # Width x Height in pixels
    cv2.imshow("Centered Pose Estimation Output", background)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run
select_image_and_run()