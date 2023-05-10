import xml.etree.ElementTree as ET
import numpy as np
import cv2
from itertools import chain

class HandwritingScaler:
    """Scale the handwriting to a target size while preserving the aspect ratio"""

    def __init__(self, data, target_bbox=(1000, 300)):
        self.target_bbox = target_bbox
        self.data = data

    def generate_image(self, data_point, min_percentile=0, max_percentile=100):

        list_of_points = list(chain.from_iterable(self.data[data_point]))
        points_array = np.array(list_of_points)

        x_perc = np.percentile(points_array[:, 0], [min_percentile, max_percentile])
        y_perc = np.percentile(points_array[:, 1], [min_percentile, max_percentile])

        x_perc = [np.min(points_array[:, 0]), np.max(points_array[:, 0])]
        y_perc = [np.min(points_array[:, 1]), np.max(points_array[:, 1])]

        width = abs(x_perc[1] - x_perc[0])
        height = abs(y_perc[1] - y_perc[0])

        scale_factor = min(self.target_bbox[0] / width, self.target_bbox[1] / height)

        img = np.zeros((self.target_bbox[1], self.target_bbox[0]), dtype=np.uint8)

        target_center = (self.target_bbox[0] // 2, self.target_bbox[1] // 2)
        bbox_center = ((x_perc[1] + x_perc[0]) // 2, (y_perc[1] + y_perc[0]) // 2)

        for i, strokes in enumerate(self.data[data_point]):
            strokes = np.array(strokes)
            strokes = strokes - bbox_center
            strokes = strokes * scale_factor + target_center
            strokes = strokes.astype(np.int32)
            cv2.polylines(img, [strokes], False, 255, thickness=2)

        return img

    def display(self, **kwargs):
        img = self.generate_image(**kwargs)
        cv2.imshow('test', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
