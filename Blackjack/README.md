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
    - Positioned ~570mm above playing space
    - Sheet layed down to reduce background noise
- Python 3.8.10 
- Tensorflow 2.10
- OpenCV-Python

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


#### Requirements
##### GPU Usage  
https://www.tensorflow.org/install/pip  
https://www.tensorflow.org/install/source#gpu

The following NVIDIA® software are only required for GPU support.
>- NVIDIA® GPU drivers version 450.80.02 or higher.
>- [CUDA® Toolkit 11.2](https://developer.nvidia.com/cuda-11.2.0-download-archive)  
>   - [Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html)
>- [cuDNN SDK 8.2.1](https://developer.nvidia.com/rdp/cudnn-archive)  
>   - Nvidia Developer Account Required
>   - [Installation Guide](https://docs.nvidia.com/deeplearning/cudnn/installation/windows.html)
>- (Optional) TensorRT to improve latency and throughput for inference.



## Reinforcement Learning
Reinforcement learning algorithm to output an action to the agent.

## Project Improvements
- Card Detection for Overlapping Cards