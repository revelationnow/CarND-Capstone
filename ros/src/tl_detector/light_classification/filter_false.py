import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
import os
from scipy.ndimage.measurements import label
global cache,cache_length
cache=[]
cache_length=0
import time
from PIL import Image

def add_heat(heatmap, bbox_list, found):

    # Iterate through list of bboxes
    i=0
    for box in bbox_list:
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        if i < len(found):
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] = found[i]
        else:
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        i=i+1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def filter_false(image,  box_list, found):
    # Read in a pickle file with bboxes saved
    # Each item in the "all_bboxes" list will contain a
    # list of boxes for one of the frames processed before

    global cache,cache_length

    cache_length+=1

    if cache_length  > 25 :
        cache_length=0
        cache = box_list
    else:
        cache = box_list + cache

    # Read in image similar to one shown above
    heat = np.zeros_like(image[:,:,0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat,box_list + cache,found)

    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heat,1)

    best_box = ((0,0),(0,0))
    best_heat = 0

    for box in box_list:
        heat_sum = np.sum(map(sum, heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]]))
        if heat_sum  > best_heat:
            best_heat = heat_sum
            best_box = box

    return best_box
