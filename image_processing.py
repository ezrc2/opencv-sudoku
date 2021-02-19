import cv2
import numpy as np

def detect_puzzle(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sudoku_outline = find_puzzle_contour(contours)
    
    if sudoku_outline is not None:
        output = image.copy()
        cv2.drawContours(output, [sudoku_outline], -1, (0, 255, 0), 3)
        cv2.imshow("Outline", output)
        cv2.waitKey(0) 
        return True
    else:
        print("No sudoku found.")
        return False
 
    

def find_puzzle_contour(contours):
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        shape = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(shape) == 4:
            return shape
  
    return None
  
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) >= 0:
            if detect_puzzle(frame):
                pass
            break

    cv2.imshow("Solved puzzle", frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()  