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

    def __init__(self, points=None, start_time=None, end_time=None):
        self.points = []
        self.start_time = start_time
        self.end_time = end_time
        if points is not None:
            for p in points:
                if isinstance(p, Point):
                    self.points.append(p)
                elif isinstance(p, (list, tuple)) and len(p) == 3:
                    self.points.append(Point(*p))
                else:
                    raise ValueError("Invalid input to Stroke constructor")

    def add_point(self, point):
        if not isinstance(point, Point):
            raise ValueError("Input must be a Point object")
        self.points.append(point)
    
    def to_nD_tuples(self, dim=2):
        return [point.to_3d()[:dim] for point in self.points]

    def __sub__(self, other):
        last_point = self.points[0]
        first_point = other.points[-1]
        return ((first_point.x - last_point.x)**2 + (first_point.y - last_point.y)**2)**0.5

    def __str__(self) -> str:
        return f"\nStroke with {len(self.points)} points."
    
    def __getitem__(self, index):
        return self.points[index]
    
    def __iter__(self):
        return iter(self.points)
    
    def __len__(self):
        return len(self.points)

class StrokeSet:

    def __init__(self, strokes=None):
        self.strokes = []

        if strokes is not None:
            for s in strokes:
                if isinstance(s, Stroke):
                    self.strokes.append(s)
                elif isinstance(s, (list, tuple)):
                    self.strokes.append(Stroke([Point(*p) for p in s]))
                else:
                    raise ValueError("Input must be a Stroke or a list/tuple of points")
        
        self.format_strokeset()

    @classmethod
    def from_sequence_data(self, sequence_data):
        strokes = []
        current_stroke = []

        for i, data in enumerate(sequence_data):
            x = data[0] + self.bbox.x_min
            y = data[1] + self.bbox.y_min
            t = data[2]
            is_start = data[3]
            point = Point(x, y, t)
            if is_start:
                if current_stroke:
                    strokes.append(Stroke(current_stroke))
                current_stroke = [point]
            else:
                current_stroke.append(point)
            if i == len(sequence_data) - 1 and current_stroke:
                strokes.append(Stroke(current_stroke))
        return StrokeSet(strokes)

    @classmethod
    def from_sequence(self, sequence_data):
        strokes = []
        current_stroke = []

        curr_x = 0
        curr_y = 0
        curr_t = 0

        for i, data in enumerate(sequence_data):
            curr_x += data[0]
            curr_y += data[1]
            curr_t += data[2]
            p = data[3]
        
            point = Point(curr_x, curr_y, curr_t)

            if p:
                current_stroke.append(point)
            else:
                if current_stroke:
                    strokes.append(Stroke(current_stroke))
                current_stroke = [point]

        if current_stroke:
            strokes.append(Stroke(current_stroke))

        return StrokeSet(strokes)

    def add_stroke(self, stroke):
        if not isinstance(stroke, Stroke):
            raise ValueError("Input must be a Stroke object")
        self.strokes.append(stroke)
        self.bbox = self._calculate_bbox()
        
    def _calculate_bbox(self):
        """Calculate the bounding box of the strokeset."""
        all_points = np.concatenate([stroke.points for stroke in self.strokes])
        x_min, y_min = np.min([(point.x, point.y) for point in all_points], axis=0)
        x_max, y_max = np.max([(point.x, point.y) for point in all_points], axis=0)

        return {"x_min": x_min, "y_min": y_min, 
                "x_max": x_max, "y_max": y_max, 
                "width": x_max - x_min, "height": y_max - y_min,
                'center': ((x_max + x_min) / 2, (y_max + y_min) / 2),
                'aspect_ratio': (x_max - x_min) / (y_max - y_min) if y_max - y_min != 0 else 'inf'}

    def format_strokeset(self):
        # substract the bbox centre from all the points in the strokeset

        self.bbox = self._calculate_bbox()

        for stroke in self.strokes:
            for point in stroke:
                point.x = point.x - self.bbox['x_min']
                point.y = point.y - self.bbox['y_min']
                
        self.bbox = self._calculate_bbox()

    def to_sequence(self):
        displacement_data = []
        prev_point = None
        for stroke in self.strokes:
            for i, point in enumerate(stroke):
                dx, dy, dt = point - prev_point if prev_point is not None else (point.x, point.y, 0)
                displacement_data.append([dx, dy, dt, int(i != 0)])
                prev_point = point

        return np.array(displacement_data)

    def to_numpy_image(self, image_size = (1000, 200)):

        for stroke in self.strokes:
            for point in stroke:
                point.x = point.x - self.bbox["center"][0]
                point.y = point.y - self.bbox["center"][1]
        
        max_dim = max(self.bbox["width"], self.bbox["height"])
        scale = image_size[0] / max_dim
        image = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)

        for stroke in self.strokes:
            for point in stroke:
                point.x = int(point.x * scale + image_size[0] / 2)
                point.y = int(point.y * scale + image_size[1] / 2)

            
            if not stroke:
                continue
            pts = np.array(stroke.to_nD_tuples())
            
            cv2.polylines(image, [pts], False, 255, thickness=2)

        return image

    def split_by_space(self, space_threshold=350):
        stroke_sets = []
        current_stroke_set = []
        prev_stroke = None
        for curr_stroke in self.strokes:
            if prev_stroke and curr_stroke - prev_stroke > space_threshold:
                stroke_sets.append(StrokeSet(current_stroke_set))
                current_stroke_set = []
            current_stroke_set.append(curr_stroke)
            prev_stroke = curr_stroke
        stroke_sets.append(StrokeSet(current_stroke_set))
        return stroke_sets

    def __str__(self) -> str:
        return f"\nStrokeSet with {len(self.strokes)} strokes"

    def __getitem__(self, index):
        return self.strokes[index]
    
    def __iter__(self):
        return iter(self.strokes)
    
    def __len__(self):
        return len(self.strokes)


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

    def split_strokesets(self, threshold=10):
        split_stroke_sets = []
        for stroke_set in self.strokeSets:
            split_stroke_sets.extend(stroke_set.split_by_space(threshold))
        return split_stroke_sets


#take a strokeset and convert to image, display, convert it to sequence data, convert the sequence data back to strokeset and to image and display

'''data_processor = DataProcessor()
data_processor.from_folder("original\\a01")
stroke_sets = data_processor.split_strokesets(3000)
stroke_set = stroke_sets[0]

#display the original strokeset
image = stroke_set.to_numpy_image()
cv2.imshow("original", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(stroke_set)
sequence_data = stroke_set.to_sequence()
stroke_set = StrokeSet.from_sequence(sequence_data)
print(stroke_set)

#display the strokeset after converting to sequence and back
image = stroke_set.to_numpy_image()
cv2.imshow("after converting to sequence and back", image)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
