# Backstory
Originally completed in 2020 as a project for a university course.
 Submitted code is shown in the [original_university_project_files](https://github.com/JamesSpr/AI/tree/main/Blackjack/original_university_project_files)

This was conducted as a group assignment. My assigned role was to complete the computer vision component,
with another member assigned to the reinforcement learning.

After review, the Reinforcement Learning actions were not updating. Due to this I have reworked in this repo.
Additionally, the code has been refactored for reusability.

# Limitiations
The cards can not be overlapping or partially in the captured area.

# Local Development Setup
## Requirements
- Camera or Webcam  
    **Experimental Setup Specification**
    - Logitech C505 Webcam
    - Positioned ~620mm above playing space
- Python 3.8.10 

## Run with Calibration
1. Setup your physical environment to have a camera pointing at a playing area large enough to have ~ 2 x 6 cards with space between all of the cards.
2. Clone this repo
3. Open a terminal and naviagate to the repo directory
4. Run the command: `python.exe run.py`
5. Place cards within the captured playing area. The more cards, the better the calibration
5. Press C to start calibration mode. Take note of the min and max values displayed
6. Press ESC to exit.
7. Run the command: `python.exe run.py -camin MIN_VAL -camax MAX_VAL` where MIN_VAL and MAX_VAL are the measured values in step 5.
8. Press P to start the game.
9. Deal cards (You are the dealer) into the marked areas and follow the onscreen actions from the AI.  

**All Commands (Key Press)**

    P: Play  
    C: Calibrate  
    M: Menu  
    ESC: Quit  

# Project Overview
**Aim**: To create a blackjack agent that can play against a human dealer.  

**Objectives**
1. Computer Vision
    - Playing Card detection and classification
2. Reinforcement Learning
    - Algorithm to provide an action based on the agent and dealer hand values. 

## Computer Vision
Playing card detection and classification

### Dataset
- Manually captured images of individual cards and labelled them as "'Value''Suit'" e.g. AS, 2C, QD, KH
- Use [dataset.py](https://github.com/JamesSpr/AI/tree/main/Blackjack/dataset.py) to augment and split the dataset into train, validation and test folders.

To build the dataset run:  
`python.exe dataset.py`

Run the following command to view options such as paths, split ratios and augmentation  
`python.exe dataset.py --help`  

### Object Detetion
- Use [object_detection.py](https://github.com/JamesSpr/AI/tree/main/Blackjack/object_detection.py) to extract card instances from a provided frame.

The feature extraction utilises OpenCV-Python for the following:
- Edge Detection
- Dilation
- Contours
- Perspective Transformation

### Image Classification


## Reinforcement Learning
Reinforcement learning algorithm to output an action to the agent.

## Project Improvements
- Card Detection for Overlapping Cards