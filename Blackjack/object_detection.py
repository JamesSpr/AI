# https://www.youtube.com/watch?v=bYBFF-N1Z98&ab_channel=PyOhio - Card Detection
# https://www.youtube.com/watch?v=Tm_7fGolVGE&ab_channel=Murtaza%27sWorkshop-RoboticsandAI - Warp Perspective

# Improvements Example
# https://github.com/geaxgx/playing-card-detection/blob/master/creating_playing_cards_dataset.ipynb

import cv2
import numpy as np
import sys

class Card:
    def __init__(self, x, y, w, h, contour, warp):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.contour = contour
        self.warp = warp
        
        self.value = ""
        self.suit = ""

def get_edges(img):
    # Get Edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 10, 15, 15)
    edges = cv2.Canny(blur, 50, 150, False)

    return edges

def morphological_transform(img):
    # # Dilate Edges to close shape
    kernel = np.ones((2,2), np.uint8)
    dilate = cv2.dilate(img.copy(), kernel, iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,2))
    # morphology_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel,iterations=1)

    return dilate

def get_contours(img):
        # Get external contours from edges
    contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sorted(contours, key=cv2.contourArea, reverse=True)

    return contours

def extract_cards(img, min_area_threshold, max_area_threshold):
    """ Extract the cards within a provided image
    
    Parameters
    ----------
    img: Image
        The image for card detection
    min_area_threshold: float 
        The minimum threshold value for a cards area. Default value based on experimental settings
    max_area_threshold: float 
        The maximum threshold value for a cards area. Default value based on experimental settings
    
    returns cards: List of Card
        All of the detected cards in the frame with the positional data
    """

    edges = get_edges(img)
    mt = morphological_transform(edges)
    contours = get_contours(mt)    
    
    # Card orientation arrays for transformation to width and height
    width, height = 200, 250
    normalOrientation = np.float32([[0,0], [width, 0], [width, height], [0, height]])
    rotatedOrientation = np.float32([[width, 0], [width, height], [0, height], [0,0]])

    cards = []
    for contour in contours:
        # 
        area = cv2.contourArea(contour)
        if area > min_area_threshold and area < max_area_threshold:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Get dimensions and coordinates of box
            ((x,y), (w,h), _) = rect

            if w > h:
                M = cv2.getPerspectiveTransform(np.float32(box), normalOrientation)
            else:
                M = cv2.getPerspectiveTransform(np.float32(box), rotatedOrientation)
                
            warp = cv2.warpPerspective(img, M, (width, height), flags=cv2.INTER_NEAREST)
            cards.append(Card(x, y, w, h, contour, warp))

    return cards

def calibrate_card_area(img, min=0, max=0):
    """ Get the min and max area threshold values for card extraction.

    return
        img: Image
            The captured image from a camera
        min: float
            min_area_threshold
        max: float
            max_area_threshold
    """
    edges = get_edges(img)
    mt = morphological_transform(edges)
    contours = get_contours(mt)    

    min_area = sys.maxsize
    max_area = 0
    average_area = 0
    counter = 0

    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if area > 150: # Remove noisy contours
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)

            # Get dimensions and coordinates of box
            ((x,y), (w,h), _) = rect

            # Get center position for text
            textsize = cv2.getTextSize(str(area), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            textX: int = round(x - textsize[0] // 2)
            textY: int = round(y + textsize[1] // 2)

            # Draw contour and corresponding area
            img = cv2.drawContours(img, contours, i, (0, 255,0), 2)
            img = cv2.putText(img, (str(area)),(textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)  

            # Min, max and average areas
            if area < min_area:
                min_area = area
            if area > max_area:
                max_area = area
    
            average_area += area
            counter += 1
    
    # Area Calculations
    if counter > 0:
        average_area = average_area // counter
        area_diff = max_area - min_area
        min = average_area - area_diff
        max = average_area + area_diff

    return img, min, max