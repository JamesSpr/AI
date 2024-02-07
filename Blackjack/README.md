# Backstory

Originally completed in 2020 as a project for a university course.
 Submitted code is shown in the [original_university_project_files](https://github.com/JamesSpr/AI/tree/main/Blackjack/original_university_project_files)

This was conducted as a group assignment. My assigned role was to complete the computer vision component,
with another member assigned to the reinforcement learning.

After review, the Reinforcement Learning actions were not updating. Due to this I have reworked in this repo.
Additionally, the code has been refactored for reusability.

# Local Development Setup
## Requirements
- Camera or Webcam
- Python >3.8 


# Project Overview
**Aim**: To create a blackjack agent that can play against a human dealer.  

**Objectives**
1. Computer Vision
    - Playing Card detection and classification
2. Reinforcement Learning
    - Algorithm to provide an action based on the agent and dealer hand values. 

## Computer Vision
Playing card detection and classification

### Dataset Creation
Manually captured images of individual cards and labelled them as "'Value''Suit'" e.g. AS, 2C, QD, KH

Use [dataset.py](https://github.com/JamesSpr/AI/tree/main/Blackjack/original_university_project_files) to augment and split the dataset into train, validation and test folders.

To build the dataset run:
`python dataset.py`

Run the following command to view options such as paths, split ratios and augmentation
`python dataset.py --help`

### Feature Extraction


### Object Recognition


## Reinforcement Learning
Reinforcement learning algorithm to output an action to the agent.

## Project Improvements
- Card Detection for Overlapping Cards