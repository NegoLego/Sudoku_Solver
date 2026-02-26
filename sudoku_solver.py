import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def blur(image):
    return cv.GaussianBlur(image, (7, 7), 3)

def threshold(image):
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)

def get_contours(image):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours

def get_grid_corners(contours):
    sorted_contours = sorted(contours, key=cv.contourArea, reverse=True)
    grid_contour = None

    for contour in sorted_contours:
        peri = cv.arcLength(contour, True)
        poly = cv.approxPolyDP(contour, 0.02 * peri, True)
        if len(poly) == 4:
            grid_contour = poly
            break

    if grid_contour is not None:
        points = grid_contour.reshape(4, 2)

        # Sort: Top-Left, Top-Right, Bottom-Right, Bottom-Left
        s = points.sum(axis=1)  # x + y
        diff = np.diff(points, axis=1)  # y - x (Note: np.diff is [y-x])

        sorted_points = np.zeros((4, 2), dtype="float32")

        sorted_points[0] = points[np.argmin(s)]  # Top-left (min sum)
        sorted_points[2] = points[np.argmax(s)]  # Bottom-right (max sum)
        sorted_points[1] = points[np.argmin(diff)]  # Top-right (min diff: x-y is max)
        sorted_points[3] = points[np.argmax(diff)]  # Bottom-left (max diff: x-y is min)

        return sorted_points
    return None

def perspective_warp(image, points):
    side = 450

    dest_points = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")

    matrix = cv.getPerspectiveTransform(points.astype("float32"), dest_points)
    warped_img = cv.warpPerspective(image, matrix, (side, side))
    return warped_img, matrix

def split_into_cells(image):
    cells = []
    cell_size = image.shape[0] // 9
    margin = cell_size // 8

    for i in range(9):
        for j in range(9):
            x1, y1 = cell_size * j + margin, cell_size * i + margin
            x2, y2 = cell_size * (j + 1) - margin, cell_size * (i + 1) - margin
            cells.append(image[y1:y2, x1:x2])
    return cells

def process_cell(cell):
    gray_cell = cv.cvtColor(cell, cv.COLOR_BGR2GRAY)
    average = np.average(gray_cell)
    _, thresholded_cell = cv.threshold(gray_cell, average * 0.5, 255, cv.THRESH_BINARY_INV)
    thresholded_resized_cell = cv.resize(thresholded_cell, (28, 28), interpolation=cv.INTER_AREA)
    center = thresholded_resized_cell[5: 24, 5: 24]
    average_on_thresholded = np.average(center)
    diff = center.astype('float32') - average_on_thresholded

    if np.count_nonzero(diff > 100) < 10:
        return {'image': thresholded_resized_cell, 'isNumber': False}
    else:
        return {'image': thresholded_resized_cell, 'isNumber': True}

class Model(nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()

        # Layer 1: Learn basic shapes (curves, lines)
        # Input: (1, 28, 28) -> Output: (32, 28, 28)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Pool reduces to (32, 14, 14)

        # Layer 2: Learn combinations of shapes (loops, corners)
        # Input: (32, 14, 14) -> Output: (64, 14, 14)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        # Pool reduces to (64, 7, 7)

        # Fully Connected Layers
        # We flattened 64 channels * 7 * 7 pixels = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(0.5) # Crucial to prevent overfitting on the 8 fonts
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1) # Flatten (Batch_Size, 3136)

        # Classifier
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = Model()
model.load_state_dict(torch.load('models/CNN_Ultra.pth', map_location=torch.device('cpu')))
model.eval()

def predict(image):
    return model(torch.tensor(image / 255.0, dtype=torch.float32).reshape(1, 1, 28, 28)).argmax().item()

def isSafe(mat, row, col, num):
    for x in range(9):
        if mat[row][x] == num:
            return False

    for x in range(9):
        if mat[x][col] == num:
            return False

    startRow = row - (row % 3)
    startCol = col - (col % 3)
    for i in range(3):
        for j in range(3):
            if mat[i + startRow][j + startCol] == num:
                return False
    return True

def solveSudokuRec(mat, row, col):
    if row == 8 and col == 9:
        return True

    # Jump to the next row
    if col == 9:
        row += 1
        col = 0

    if mat[row][col] != 0:
        return solveSudokuRec(mat, row, col + 1)

    for num in range(1, 10):
        if isSafe(mat, row, col, num):
            mat[row][col] = num
            if solveSudokuRec(mat, row, col + 1):
                return True
            mat[row][col] = 0

def solveSudoku(sudoku):
    mat = [[nr for nr in row] for row in sudoku]
    solveSudokuRec(mat, 0, 0)
    return mat

def draw_solution_on_original(original_image, matrix, sudoku_solved, sudoku_original):
    # 1. Create a blank image with the same size as the warped grid (450x450)
    side = 450
    flat_overlay = np.zeros((side, side, 3), dtype="uint8")
    cell_size = side // 9

    # 2. Draw the missing numbers on the flat overlay
    for i in range(9):
        for j in range(9):
            # Only draw if the cell was originally empty (0)
            if sudoku_original[i][j] == 0:
                text = str(sudoku_solved[i][j])

                # Calculate text size to center it perfectly
                font = cv.FONT_HERSHEY_SIMPLEX
                font_scale = 1.5
                thickness = 3
                (text_w, text_h), _ = cv.getTextSize(text, font, font_scale, thickness)

                # Calculate x, y to center the text in the cell
                # x = cell_left + half_cell_width - half_text_width
                x = int((j * cell_size) + (cell_size / 2) - (text_w / 2))
                y = int((i * cell_size) + (cell_size / 2) + (text_h / 2))

                # Draw the number in Green (BGR format: 0, 255, 0)
                cv.putText(flat_overlay, text, (x, y), font, font_scale, (0, 255, 0), thickness)

    # 3. Calculate the Inverse Matrix
    # We want to go from Warped -> Original, so we invert the matrix
    inv_matrix = np.linalg.inv(matrix)

    # 4. Warp the overlay back to the original image shape
    h, w, _ = original_image.shape
    warped_overlay = cv.warpPerspective(flat_overlay, inv_matrix, (w, h))

    # 5. Combine the images
    # Since the background of warped_overlay is black (0,0,0),
    # adding it to the original will just overlay the green pixels.
    result = cv.addWeighted(original_image, 1, warped_overlay, 1, 0)

    return result