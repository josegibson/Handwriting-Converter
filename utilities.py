import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
from itertools import chain

class Stroke:

    def __init__(self, points = [], start_time = None, end_time = None):
        self.points = points
        self.start_time = start_time
        self.end_time = end_time

    def add_point(self, x, y, t):
        self.points.append((x,y,t))

    def get_points(self):
        return self.points
    

class StrokeList:

    def __init__(self, strokes = []):
        self.strokes = strokes

    def add_stroke(self, stroke):
        self.strokes.append(stroke)

    def get_strokes(self):
        return self.strokes

    def from_folder(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
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
                            t = float(point.get("t"))
                            points.append((x, y, t))
                        strokes.append(Stroke(points))
                    self.add_stroke(strokes)

    def belongs_to_new_line(self, prev_last_point, curr_first_point, threshold=350):
        if prev_last_point is None or curr_first_point is None:
            return False
        if curr_first_point[0] - prev_last_point[0] < -threshold:
            return True
        return False
    

class StrokeProcessor:

    def __init__(self, num_points = None):
        self.num_points = num_points

    def strokes_to_sequence(self, strokes):
        ''' Convert strokes to the displacement representation of [dx, dy, t, p] 
            dx and dy are the displacements in x and y directions, 
            t is the time interval between the current and previous point
            p is the pen state (0 for pen up and 1 for pen down)'''
        
        # Initialize the displacement representation
        sequence = []

        # Iterate over the strokes
        for stroke in strokes:

            end_point = None

            # Iterate over the points in the stroke
            for i in range(len(stroke)):

                # set the penstate default to 1
                p = 1

                # Calculate the displacement in x and y directions and time interval
                if i == 0:
                    if end_point is None:
                        dx = stroke[i][0]
                        dy = stroke[i][1]
                        dt = 0
                    else:
                        # the displacement from the last point of the previous stroke
                        dx = stroke[i][0] - end_point[0]
                        dy = stroke[i][1] - end_point[1]
                        dt = stroke[i][2] - end_point[2]

                        # the displacement from the last point of the previous stroke 
                        # to the first point of the new stroke is with penstate 0
                        p = 0
                else:
                    # the displacement from the previous point
                    dx = stroke[i][0] - stroke[i-1][0]
                    dy = stroke[i][1] - stroke[i-1][1]
                    dt = stroke[i][2] - stroke[i-1][2]

                # Append the displacement to the displacement data
                sequence.append(dx)
                sequence.append(dy)

            # Update the end point of the stroke
            end_point = stroke[-1]



    def sequence_to_strokes(self, sequence):
        strokes = []
        points = []
        for i in range(0, len(sequence), 2):
            points.append((sequence[i], sequence[i+1]))
        strokes.append(Stroke(points))
        return StrokeList(strokes)

    def normalize_sequence(self, sequence):
        sequence = np.array(sequence)
        sequence = sequence / 1000
        return sequence

    def denormalize_sequence(self, sequence):
        sequence = np.array(sequence)
        sequence = sequence * 1000
        return sequence

    def _reduce_points(self, points):
        if len(points) <= self.num_points:
            return points
        new_points = []
        for i in range(self.num_points):
            index = int(i * len(points) / self.num_points)
            new_points.append(points[index])
        return new_points
    

class ImageConverter:

    def __init__(self, height, width, min_percentile=0, max_percentile=100):
        self.height = height
        self.width = width

    def strokes_to_image(self, strokes):
        point_pool = list(chain.from_iterable(strokes))
        point_pool = np.array(point_pool)

        x_perc = [np.min(point_pool[:, 0]), np.max(point_pool[:, 0])]
        y_perc = [np.min(point_pool[:, 1]), np.max(point_pool[:, 1])]

        width = abs(x_perc[1] - x_perc[0])
        height = abs(y_perc[1] - y_perc[0])

        scale_factor = min(self.width / width, self.height / height)

        img = np.zeros((self.height, self.width), dtype=np.uint8)

        for stroke in strokes:
            stroke = np.array(stroke)
            stroke[:, 0] = (stroke[:, 0] - x_perc[0]) * scale_factor
            stroke[:, 1] = (stroke[:, 1] - y_perc[0]) * scale_factor
            stroke = stroke.astype(np.int32)
            cv2.polylines(img, [stroke], False, 255, thickness=2)

        return img


    
