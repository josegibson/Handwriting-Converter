import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2

global_arr = np.array([])

def is_new_paragraph(prev_last_point, curr_first_point, threshold=200):
    if prev_last_point is None or curr_first_point is None:
        return False
    if curr_first_point[0] - prev_last_point[0] < -threshold:
        return True
    return False


def strokes_to_image(strokes):

    strokes = np.array(strokes)

    # Calculate the bounding box of the strokes
    x_min, y_min = np.min(strokes, axis=(0, 1))
    x_max, y_max = np.max(strokes, axis=(0, 1))
    width = x_max - x_min
    height = y_max - y_min

    # Calculate the center of the bounding box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Create an empty image
    img = np.zeros((height, width), dtype=np.uint8)

    # Shift the strokes to the center of the image and draw them
    shifted_strokes = np.array(strokes) - [center_x, center_y]
    for stroke in shifted_strokes:
        cv2.polylines(img, [stroke.astype(np.int32)], False, 255, thickness=2)

    return img


for root, dirs, files in os.walk("original"):
    for file in files:
        if file.endswith(".xml"):
            xml_path = os.path.join(root, file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            strokesets = root.find("StrokeSet")
            strokes = []
            for stroke in strokesets.findall("./Stroke"):
                points = []
                for point in stroke.findall("./Point"):
                    x = float(point.get("x"))
                    y = float(point.get("y"))
                    points.append([x, y])
                strokes.append(points)

            prev_last_point = None
            strokes_of_line = []
            for stroke in strokes:
                curr_first_point = stroke[0]
                if is_new_paragraph(prev_last_point, curr_first_point):
                    
                    cv2.imshow(strokes_to_image(strokes_of_line))

                    strokes_of_line = []

                else:
                    strokes_of_line.append(stroke)
                prev_last_point = stroke[-1]
