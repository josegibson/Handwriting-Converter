import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


class Point:
    def __init__(self, x, y, t):
        self.x = x
        self.y = y
        self.t = t

    def to_3d(self):
        return self.x, self.y, self.t
    
    def to_2d(self):
        return self.x, self.y

    def __str__(self) -> str:
        return f"({self.x}, {self.y}, {self.t})"

    def __sub__(self, other):
        return self.x - other.x, self.y - other.y, self.t - other.t


class Stroke:
    def __init__(self, points=[], start_time=None, end_time=None):
        self.points = points
        self.start_time = start_time
        self.end_time = end_time

    def add_point(self, x, y, t):
        self.points.append(Point(x, y, t))

    def get_points(self, as_tuple=False, for_image=False):
        if as_tuple:
            if for_image:
                return [point.get_tuple()[:2] for point in self.points]
            else:
                return [point.get_tuple() for point in self.points]
        return self.points

    def get_first_point(self):
        return (
            self.points[0] if type(self.points[0]) == Point else Point(self.points[0])
        )

    def get_last_point(self):
        return (
            self.points[-1]
            if type(self.points[-1]) == Point
            else Point(self.points[-1])
        )

    def __sub__(self, other):
        last_point = self.get_first_point()
        first_point = other.get_last_point()
        return first_point.x - last_point.x

    def __len__(self):
        return len(self.points)

    def __str__(self) -> str:
        return f"\nStroke with {len(self.points)} points."

    def __getitem__(self, index):
        return self.points[index]


class StrokeSet:
    def __init__(self, strokes=[]):
        self.strokes = strokes

    def to_sequence(self):
        displacement_data = []
        for stroke in self.strokes:
            end_point = None
            for i in range(len(stroke)):
                p = 1
                if i == 0:
                    if end_point is None:
                        dx, dy, dt = stroke[i].x, stroke[i].y, 0

                    else:
                        dx, dy, dt = stroke[i] - end_point
                        p = 0
                else:
                    dx, dy, dt = stroke[i] - stroke[i - 1]
                displacement_data.append([dx, dy, dt, p])
            end_point = stroke[-1]

        # replace the first point's displacement with average x, y of all displacements for x, y and 0 for t
        avg_x = sum([x for x, y, t, p in displacement_data]) / len(displacement_data)
        avg_y = sum([y for x, y, t, p in displacement_data]) / len(displacement_data)
        displacement_data[0] = [avg_x, avg_y, 0, 0]

        return displacement_data

    def sequence_to_strokes(self, sequence):
        strokes = []
        stroke = []
        for x, y, t, p in sequence:
            if p == 0 and len(stroke) > 0:
                strokes.append(stroke)
                stroke = []
            stroke.append([x, y, t])
        strokes.append(stroke)
        return strokes

    def normalize_sequence(self, sequence):
        max_x = max([abs(x) for x, y, t, p in sequence])
        max_y = max([abs(y) for x, y, t, p in sequence])
        max_t = max([abs(t) for x, y, t, p in sequence])

        normalized_sequences = []
        for x, y, t, p in sequence:
            normalized_sequences.append([x / max_x, y / max_y, t / max_t, p])

        return normalized_sequences

    def format_strokeset(self):
        points_array = [stroke.get_points(as_tuple=True) for stroke in self.strokes]
        points_array = np.concatenate(points_array)

        x_min, y_min = np.min(points_array, axis=0)[:2]
        x_max, y_max = np.max(points_array, axis=0)[:2]

        self.width = abs(x_max - x_min)
        self.height = abs(y_max - y_min)

        self.bbox_center = (x_min + x_max // 2, y_min + y_max // 2)

        self.strokes = [
            Stroke(
                [
                    Point(x - self.bbox_center[0], y - self.bbox_center[1], t)
                    for x, y, t in stroke.get_points(as_tuple=True)
                ]
            )
            for stroke in self.strokes
        ]

        # return the self.strokes with converting all points in the strokes to tuples and avoiding the time values of each point
        return [
            stroke.get_points(as_tuple=True, for_image=True) for stroke in self.strokes
        ]

    def strokeset_to_image(self, image_size=(1000, 200)):
        strokeset = self.format_strokeset()

        img = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        self.scale_factor = min(image_size[0] / self.width, image_size[1] / self.height)

        target_center = (image_size[0] // 2, image_size[1] // 2, 0)

        strokeset = [
            (np.array(stroke)) * self.scale_factor + target_center
            for stroke in strokeset
        ]
        strokeset = [stroke.astype(np.int32) for stroke in strokeset]

        cv2.fillPoly(img, strokeset, color=255)

        return img

    def split_by_lines(self):
        stroke_sets = []
        line_strokes = []
        prev_stroke = None
        for curr_stroke in self.strokes:
            if prev_stroke:
                if curr_stroke - prev_stroke < -350:
                    stroke_sets.append(StrokeSet(line_strokes))
                    line_strokes = []
            line_strokes.append(curr_stroke)
            prev_stroke = curr_stroke
        stroke_sets.append(StrokeSet(line_strokes))
        return stroke_sets

    def split_by_lift_threshold(self, threshold=10):
        stroke_sets = []
        line_strokes = []
        prev_stroke = None
        for curr_stroke in self.strokes:
            if curr_stroke - prev_stroke > threshold:
                stroke_sets.append(line_strokes)
                line_strokes = []
            line_strokes.append(curr_stroke)
            prev_stroke = curr_stroke
        stroke_sets.append(line_strokes)
        return stroke_sets

    def __str__(self) -> str:
        return f"\nStrokeSet with {len(self.strokes)} strokes"


class DataProcessor:
    def __init__(self, strokeSets=[]):
        self.strokeSets = strokeSets

    def add_strokeSet(self, strokeSet):
        self.strokeSets.append(StrokeSet(strokeSet))

    def get_strokeSets(self):
        return self.strokeSets

    def from_folder(self, folder_path):
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".xml"):
                    xml_path = os.path.join(root, file)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    _strokesets = root.find("StrokeSet")
                    strokeSet = []
                    for stroke in _strokesets.findall("./Stroke"):
                        points = []
                        for point in stroke.findall("./Point"):
                            x = float(point.get("x"))
                            y = float(point.get("y"))
                            t = float(point.get("time"))
                            points.append(Point(x, y, t))
                        strokeSet.append(Stroke(points))
                    self.add_strokeSet(strokeSet)

    def spread_lines(self):
        spread_strokeSets = []
        for strokeSet in self.strokeSets:
            spread_strokeSets.extend(strokeSet.split_by_lines())
        return spread_strokeSets

    def spread_lift_threshold(self, threshold=10):
        spread_strokeSets = []
        for strokeSet in self.strokeSets:
            spread_strokeSets.extend(strokeSet.split_by_lift_threshold(threshold))
        return spread_strokeSets


# show an image of a stroke


"""DatasetCreator class:
This class will use the StrokeProcessor class to create a dataset from a folder of .xml files containing handwriting data.
It will use the stroke data to create a set of strokes, sequences, and images for each data point, and store these in a dictionary with keys for the stroke, sequence, and image data, as well as any metadata associated with the data point.
To make it future-proof, we can add functions to handle the OCR scanned text associated with the handwriting data."""


"""HandwritingDataset class:
This class will define the structure of a handwriting dataset, containing a list of data points and their corresponding strokes, sequences, images, and metadata.
It will have functions to load and save the dataset to disk, as well as preprocess the data for training and testing the machine learning model."""
