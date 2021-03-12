import cv2
import numpy as np
from imutils.perspective import four_point_transform
from imutils.convenience import grab_contours
from tensorflow.keras.models import load_model

def find_puzzle_outline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)

    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)

    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    puzzle_outline = None
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        shape = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(shape) == 4:
            puzzle_outline = shape
            break

    if not puzzle_outline.all():
        print("no puzzle detected")
        return None, None

    # output = image.copy()
    # cv2.drawContours(output, [puzzle_outline], -1, (0, 255, 0), 2)
    # cv2.imshow("Puzzle Outline", output)
    # cv2.waitKey(0)
    
    puzzle = four_point_transform(image, puzzle_outline.reshape(4, 2))
    warped = four_point_transform(gray, puzzle_outline.reshape(4, 2))

    cv2.imshow("puzzle", puzzle)
    cv2.waitKey(0)
    
    return (puzzle, warped)

def extract_cells(image):
    pass

def main():
    cap = cv2.VideoCapture(0)
    image = cv2.imread("pictures/sudoku.png")
    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.flip(frame, 1)
    #     cv2.imshow("Camera", frame)
    #     find_puzzle_outline(frame)

    #     if cv2.waitKey(1) >= 0:
    #         break
    find_puzzle_outline(image)
    # cv2.imshow("Solved puzzle", frame)
    # cv2.waitKey(0)

if __name__ == "__main__":
    main()  