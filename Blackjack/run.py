import cv2
import sys
import argparse
import object_detection as od

parser = argparse.ArgumentParser(description="Play Blackjack")
parser.add_argument('-camin', '--card-area-min', metavar="card_area_min", type=int, default=0, help="Minimum card area threshold value")
parser.add_argument('-camax', '--card-area-max', metavar="card_area_max", type=int, default=0, help="Maximum card area threshold value")

if __name__ == "__main__":
    args = parser.parse_args()
    
    print("Initialising...", end="\r")
    # Hands
    dealer = []
    player = []

    # Get Video Stream
    cap = cv2.VideoCapture(0)
    _, frame = cap.read()
    height, width, _ = frame.shape

    print("Welcome to BlackJack!")
    print("Key Bindings\nM: Menu\nP: Play\nC: Card Calibration Mode\nESC: Exit")
    mode = "Menu"

    # Initialise Calibration mode variables
    cal_min = sys.maxsize
    cal_max = 0
    
    while(True):
        # Capture each frame and crop to usable area
        ret, frame = cap.read()
        key_press = cv2.pollKey()
        # print(key_press)

        # Mode Triggers
        if key_press == 109: # M
            mode = 'Menu'
        if key_press == 112: # P
            mode = 'Play'
        if key_press == 99: # C
            mode = 'Calibration'
        if key_press == 27: # ESC
            break

        
        if mode == "Menu":
            title = "Welcome to Blackjack!"
            
            # Get center position for text
            titlesize = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            textX = (width - titlesize[0]) // 2
            textY = (height + titlesize[1]) // 2

            menu = cv2.putText(frame, title, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2, cv2.LINE_AA)
            cv2.imshow("Video Feed", menu)

        elif mode == "Play":
            cv2.imshow("Video Feed", frame)

        elif mode == "Calibration":
            frame, min, max = od.calibrate_card_area(frame)

            if min < cal_min:
                cal_min = min
            if max > cal_max:
                cal_max = max
            
            # Get center position for text
            title = "Calibration Mode"
            titlesize = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 2)[0]
            titleX = (width - titlesize[0]) // 2
            titleY = (height + titlesize[1]) // 2
            
            subtitle = f"Max: {cal_max}"
            subsize = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            subX = (width - subsize[0]) // 2
            subY = (height + subsize[1]) // 2

            calibration_mode = cv2.putText(frame, title, (titleX, titleY), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 2, cv2.LINE_AA)
            calibration_mode = cv2.putText(frame, subtitle, (subX + titlesize[0]//2, subY + titlesize[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            calibration_mode = cv2.putText(frame, f"Min: {cal_min}", (subX - titlesize[0]//4, subY + titlesize[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow("Video Feed", calibration_mode)

        else:
            cv2.imshow("Video Feed", frame)

    cap.release()
    cv2.destroyAllWindows()