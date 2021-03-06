import cv2
import numpy as np

def find_puzzle_outline(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 3)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    #cv2.imshow("thresh", thresh)
    contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    # Find sudoku outline
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        shape = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(shape) == 4:
            return shape
  
    return None
    
def get_corners(corners):
    corners = [(corner[0][0], corner[0][1]) for corner in corners]
    # top left, top right, bottom right, bottom left
    return corners[0], corners[1], corners[2], corners[3]

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