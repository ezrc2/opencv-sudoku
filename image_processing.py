import cv2
import numpy as np
import imutils
from imutils.perspective import four_point_transform

def find_puzzle_outline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    #cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find sudoku outline
    puzzle_outline = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        shape = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(shape) == 4:
            puzzle_outline = shape
  
    if puzzle_outline == None:
        print("no puzzle detected")
        return None, None
    
    puzzle = four_point_transform(image, puzzle_outline.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_outline.reshape(4, 2))

    cv2.imshow("puzzle", puzzle)
    cv2.waitKey(0)
    
    return (puzzle, warped)


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) >= 0:
            break

    cv2.imshow("Solved puzzle", frame)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()  