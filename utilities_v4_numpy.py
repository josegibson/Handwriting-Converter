import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import glob


def from_folder(folder_path):
    strokesets = []
    xml_paths = glob.glob(os.path.join(folder_path, "**/*.xml"), recursive=True)
    for xml_path in xml_paths:
        tree = ET.parse(xml_path)
        _strokesets = tree.find(".//StrokeSet")
        strokeSet = [np.array([(float(point.get("x")), float(point.get("y")), float(point.get("time")))
                               for point in stroke.findall("./Point")]) for stroke in _strokesets.findall("./Stroke")]
        strokesets.append(format_strokeset(strokeSet))
    return strokesets

def split_strokesets(strokesets, threshold=3500):
    new_strokesets = []
    for strokeset in strokesets:
        strokes = [stroke for stroke in strokeset]  # convert to list
        starts = [stroke[0] for stroke in strokes]
        ends = [stroke[-1] for stroke in strokes]
        dists = [np.sqrt((start[0]-end[0])**2 + (start[1]-end[1])**2) for start, end in zip(starts[1:], ends[:-1])]
        split_indices = np.where(np.array(dists) > threshold)[0] + 1
        split_indices = np.concatenate(([0], split_indices, [len(strokes)]))
        split_strokesets = [strokes[start:end] for start, end in zip(split_indices[:-1], split_indices[1:])]
        split_strokesets = list(map(format_strokeset, split_strokesets))
        new_strokesets.extend(split_strokesets)
    return new_strokesets

def format_strokeset(strokeset):
    # compute bbox properties of strokeset
    pts = np.concatenate(strokeset, axis=0)  # extract the x,y coordinates
    pts = pts.reshape((-1, 3))[:, :2]
    x_min, y_min = np.min(pts, axis=0)
    x_max, y_max = np.max(pts, axis=0)
    w, h = x_max - x_min, y_max - y_min
    if w == 0 or h == 0:
        return None
    
    # transform the points in the strokeset so that the top-left of the bbox is their origin
    # starting time is set to 0 for that substract the first point time from all points time
    for stroke in strokeset:
        stroke -= [x_min, y_min, stroke[0, 2]]

    return strokeset

def display_strokeset(strokeset, scale_factor=0.1):
    pts = np.concatenate(strokeset, axis=0)
    x_min, y_min = np.min(pts[:, :2], axis=0)
    x_max, y_max = np.max(pts[:, :2], axis=0)
    w, h = int(scale_factor * (x_max - x_min)), int(scale_factor * (y_max - y_min))
    if w == 0 or h == 0:
        return
    image = np.zeros((h, w), dtype=np.uint8)
    pts -= [x_min, y_min, 0]
    pts *= scale_factor
    pts = pts[:, :2]  # remove the z-dimension
    pts = pts.astype(np.int32)
    strokes = np.split(pts, np.cumsum([len(s) for s in strokeset[:-1]]))
    for stroke in strokes:
        cv2.polylines(image, [stroke], isClosed=False, color=255, thickness=2)
    cv2.imshow("Strokeset", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_formatted_strokeset(strokeset, image_size=(1000, 200)):

    pts = np.concatenate(strokeset, axis=0)
    x_min, y_min = np.min(pts[:, :2], axis=0)
    x_max, y_max = np.max(pts[:, :2], axis=0)

    w, h = image_size
    scale_factor = min(w / (x_max - x_min), h / (y_max - y_min))

    img = np.zeros((h, w), dtype=np.uint8)

    for stroke in strokeset:
        stroke = stroke[:, :2] * scale_factor
        cv2.polylines(img, [stroke.astype(np.int32)], isClosed=False, color=255, thickness=2)

    cv2.imshow("formatted Strokeset", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def strokeset_to_seq(strokeset):
    displacement_data = []
    prev_point = None
    for stroke in strokeset:
        for i, point in enumerate(stroke):
            dx, dy, dt = point - prev_point if prev_point is not None else (point[0], point[1], 0)
            displacement_data.append([dx, dy, dt, int(i != 0)])
            prev_point = point
    return np.array(displacement_data)

def seq_to_strokeset(seq):
    # convert sequence data to a numpy array for better performance
    seq_arr = np.array(seq)

    # calculate cumulative sum of x, y, and t values
    cum_sum = np.cumsum(seq_arr, axis=0)

    # create boolean mask to identify end of strokes
    mask = seq_arr[:, 3] == 0

    # split the sequence into strokes based on the mask
    split_indices = np.nonzero(mask)[0] + 1
    strokeset = np.split(cum_sum, split_indices)

    return strokeset



import random

def test_seq_to_strokeset(filename, threshold):
    # Load strokesets from file
    strokesets = from_folder(filename)
    
    # Split strokesets
    strokesets = split_strokesets(strokesets, threshold)
    
    # Select a random strokeset
    test_strokeset = random.choice(strokesets)
    
    # Print shapes of strokes before conversion
    print("Shapes of strokes before conversion:")
    for i, stroke in enumerate(test_strokeset):
        print(f"Stroke {i+1}: {stroke.shape}")
    
    # Convert strokeset to sequence
    test_seq = strokeset_to_seq(test_strokeset)
    
    # Convert sequence back to strokeset
    strokeset_after_conversion = seq_to_strokeset(test_seq)
    
    # Print shapes of strokes after conversion
    print("\nShapes of strokes after conversion:")
    for i, stroke in enumerate(strokeset_after_conversion):
        print(f"Stroke {i+1}: {stroke.shape}", end="\t")
    print()
    
    # Display original strokeset and converted strokeset
    display_formatted_strokeset(test_strokeset)
    display_formatted_strokeset(strokeset_after_conversion)


test_seq_to_strokeset('original\\a01', 400)


