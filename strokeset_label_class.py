import os
import numpy as np
import xml.etree.ElementTree as ET
import cv2


class StrokeSet:
    def __init__(self, strokes):
        self.strokes = strokes
        
    def to_displacement(self):
        ''' Convert strokes to the displacement representation of [dx, dy, t, p] 
            dx and dy are the displacements in x and y directions, 
            t is the time interval between the current and previous point
            p is the pen state (0 for pen up and 1 for pen down)'''

        # Initialize the displacement representation
        displacement_data = []

        # Iterate over the strokes
        for stroke in self.strokes:

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
                displacement_data.append([dx, dy, dt, p])

            # Update the end point of the stroke
            end_point = stroke[-1]
    
        return displacement_data
                
    def from_displacement(self, displacement_data, source = [0, 0, 0]):
        # Convert displacement representation to strokes

        # Initialize the strokes
        strokes = []

        # Initialize the current stroke
        stroke = []

        # Iterate over the displacement data
        for pt in displacement_data:
                
            # Calculate the current point
            source[0] += pt[0]
            source[1] += pt[1]
            source[2] += pt[2]

            # Append the current point to the current stroke
            stroke.append(source)

            # If the pen state is 0, append the current stroke to the strokes and reset the current stroke
            if pt[3] == 0:
                strokes.append(stroke)
                stroke = []
      
    def normalize_diplacement(self, displacement_data):
        # Normalize the strokes which is in displacement representation using the longest displacement as the denominator

        # convert displacement data to numpy array
        displacement_data = np.array(displacement_data)

        # Calculate the longest displacement
        longest_displacement = np.max(np.abs(displacement_data), axis=0)

        # Normalize the displacement data
        displacement_data = displacement_data / longest_displacement

        return displacement_data  
    
    def denormalize(self, displacement_data, normalisation_factor):
        # Convert normalized numpy array displacement_data back to original scale

        denormalized_data = []

        # Iterate over the displacement data
        for pt in displacement_data:
            denormalized_data.append(pt * normalisation_factor)
        
        return denormalized_data

