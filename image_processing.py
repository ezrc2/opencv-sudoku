import cv2
import numpy as np
from sudoku import Sudoku
from imutils.perspective import four_point_transform
from imutils.convenience import grab_contours
from skimage.segmentation import clear_border
from tensorflow.keras.models import load_model

# Credit to https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/ for 
# the tutorial on how to find the sudoku puzzle using OpenCV

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

    # cv2.imshow("puzzle", puzzle)
    # cv2.waitKey(0)
    
    return (puzzle, warped)

def extract_digit(cell):
    thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = grab_contours(contours)

    if len(contours) == 0: # empty cell
        return None
	
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [largest], -1, 255, -1)

    percentFilled = cv2.countNonZero(mask) / float(thresh.shape[0] * thresh.shape[1])
    if percentFilled < 0.03: # noise
        return None
	
    digit = cv2.bitwise_and(thresh, thresh, mask=mask) # apply mask
    
    return digit

def main():
    board_size = 9
    # cap = cv2.VideoCapture(0)
    image = cv2.imread("pictures/sudoku.png")
    # while True:
    #     ret, frame = cap.read()
    #     frame = cv2.flip(frame, 1)
    #     cv2.imshow("Camera", frame)

    #     if cv2.waitKey(1) >= 0:
    #         break
    puzzle, warped = find_puzzle_outline(image)

    model = load_model("model.h5")
    board = -1 * np.ones((board_size, board_size), dtype="int")
    dx = warped.shape[1] // board_size
    dy = warped.shape[0] // board_size
    cell_locations = []

    for y in range(board_size):
        cell_row = []
        for x in range(board_size):
            x1 = x * dx
            y1 = y * dy
            x2 = (x + 1) * dx
            y2 = (y + 1) * dy
            cell_row.append((x1, y1, x2, y2))

            cell = warped[y1:y2, x1:x2]
            digit = extract_digit(cell)

            if digit is not None:
                # cv2.imshow("digit", digit)
                # cv2.waitKey(0)
                ROI = cv2.resize(digit, (28, 28))
                ROI = ROI.astype("float") / 255.0
                ROI = ROI.reshape(1, 28, 28, 1)
                prediction = model.predict(ROI).argmax(axis=1)[0]
                board[y, x] = prediction

        cell_locations.append(cell_row)
    
    original = board.copy()
    solver = Sudoku(board)
    solver.solve()
    solution = puzzle.copy()

    for i in range(len(cell_locations)):
        row = cell_locations[i]
        for j in range(len(row)):
            if original[i][j] == -1:
                location = row[j]
                x1, y1, x2, y2 = location[0], location[1], location[2], location[3]
                x = int(0.35 * (x2 - x1) + x1)
                y = int(0.7 * (y2 - y1) + y1)
                cv2.putText(solution, str(solver.board[i, j]), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow("Solved Sudoku", solution)
    cv2.waitKey(0)
    

if __name__ == "__main__":
    main()  