import xml.etree.ElementTree as ET
import numpy as np
import cv2


class HandwrittenLine:
    def __init__(self, line_strokes, text):
        self.text = text
        self.line_strokes = line_strokes
        self.min_x = None
        self.min_y = None
        self.max_x = None
        self.max_y = None

        self.bound_line()

    def bound_line(self):
        # find min and max x,y coordinates for the line
        self.min_x = min(point[0] for stroke in self.line_strokes for point in stroke) - 20
        self.min_y = min(point[1] for stroke in self.line_strokes for point in stroke) - 20
        self.max_x = max(point[0] for stroke in self.line_strokes for point in stroke) + 20
        self.max_y = max(point[1] for stroke in self.line_strokes for point in stroke) + 20

        # adjust the coordinates of the points in the line so that they are relative to the bounding box
        for stroke in self.line_strokes:
            for point in stroke:
                point[0] -= self.min_x
                point[1] -= self.min_y

        # return the adjusted strokes and bounding box details
        return self.min_x, self.min_y, self.max_x, self.max_y

    def get_line_image(self):
        # initialize a blank image
        img = np.zeros((self.max_y - self.min_y, self.max_x - self.min_x, 3), np.uint8)

        # draw each stroke onto the image
        for stroke in self.line_strokes:
            cv2.polylines(img, [stroke], False, (0, 255, 0), 2)

        return img
    
    def show_line_image(self):
        cv2.imshow(self.text, self.get_line_image())
        cv2.waitKey(0)
        cv2.destroyAllWindows()



def create_dataset(file_path ):

    # load eBeam XML file
    tree = ET.parse(file_path)
    root = tree.getroot()
    line_objects = []

    # get the strokeset from the XML file
    strokeset = root.find('StrokeSet')

    # get the transcription from the XML file
    transcription = root.find('Transcription')
    text = transcription.find('Text')
    linesprint = text.text.strip().splitlines()

    # set threshold for x-axis difference between paragraphs
    threshold = 200

    # initialize variables for tracking line status
    prev_last_point = None
    curr_first_point = None
    line_no = 0

    # initialize variables for tracking line strokes
    line_strokes = []

    for stroke in strokeset:

        # get points from stroke
        _points = stroke.findall('Point')

        # convert points to numpy array
        _points = np.array([[float(point.get('x')), float(point.get('y'))] for point in _points]) / 10
        _points = _points.astype(int)

        curr_first_point = _points[0]

        # check if this stroke is a new paragraph
        if prev_last_point is not None and curr_first_point is not None and curr_first_point[0] - prev_last_point[0] < -threshold:
            # this stroke is a new paragraph

            # add current line strokes to the line_objects list
            if line_strokes:
                line_objects.append(HandwrittenLine(line_strokes, linesprint[line_no]))
                line_no += 1
            line_strokes = []

        # add current stroke to current line strokes
        line_strokes.append(_points)

        # update variables for tracking paragraph boundaries
        prev_last_point = _points[-1]

    # add last line strokes to the line data dictionary
    if line_strokes:
        line_objects.append(HandwrittenLine(line_strokes, linesprint[line_no]))

    return line_objects


if __name__ == '__main__':
    line_objects = create_dataset('original/a01/a01-013/strokesz.xml')
    for line in line_objects:
        line.show_line_image()