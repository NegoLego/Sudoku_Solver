import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

x, y = torch.load('data/MNIST/processed/training.pt')
torch.set_printoptions(precision=4, sci_mode=False)
# Neural Network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return F.softmax(x, dim=1)

model = CNN()
model.load_state_dict(torch.load('models/best_CNN_model.pth', map_location=torch.device('cpu')))
model.eval()

# returns list of 4 tuples (x, y)
def get_grid_corners(img):
    # 2. Find all corners
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # 3. Sort by area and look for square-ish one
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    grid_contour = None
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)

        # If the contour has 4 points, it's likely our grid
        if len(approx) == 4:
            grid_contour = approx
            break

    if grid_contour is not None:
        # Reshape to a simple list of 4 (x, y) coordinates
        points = grid_contour.reshape(4, 2)
        return points

    return None

def visualize_corners(img, points):
    vis_img = img.copy()

    points = points.astype(int)

    for i, pt in enumerate(points):
        cv.circle(vis_img, tuple(pt), 10, (0, 0, 255), -1)
        cv.putText(vis_img, str(i), tuple(pt), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5, cv.LINE_AA)

    return vis_img

def sort_corners(points):
    new_list = [0] * 4
    min_sum = 9000
    max_sum = 0
    min_dif = 9000
    max_dif = 0
    for i in range(4):
        x = points[i][0]
        y = points[i][1]
        sum = x + y
        dif = x - y
        if sum < min_sum:
            min_sum = sum
            new_list[0] = points[i]
        if sum > max_sum:
            max_sum = sum
            new_list[2] = points[i]
        if dif < min_dif:
            min_dif = dif
            new_list[3] = points[i]
        if dif > max_dif:
            max_dif = dif
            new_list[1] = points[i]
    return np.array(new_list)

def perspective_warp(img, points):
    side = 450

    dest_points = np.array([
        [0, 0],
        [side - 1, 0],
        [side - 1, side - 1],
        [0, side - 1]
    ], dtype="float32")

    matrix = cv.getPerspectiveTransform(points.astype("float32"), dest_points)
    warped_img = cv.warpPerspective(img, matrix, (side, side))
    return warped_img

def extract_cells(img):
    cells = []
    cell_size = img.shape[0] // 9
    margin = cell_size // 12

    for i in range(9):
        for j in range(9):
            x1, y1 = cell_size * j + margin, cell_size * i + margin
            x2, y2 = cell_size * (j + 1) - margin, cell_size * (i + 1) - margin
            cells.append(img[y1:y2, x1:x2])
    return cells

def show_first_row(cells):
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(cells[i], cmap='gray')
    plt.show()

if __name__ == '__main__':
    img_name = "img1.jpeg"
    img = cv.imread(f'static/img/{img_name}')

    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blured = img_gray

    # 1. Use adaptive thresholding
    # 11 is the block size, 2 is the constant subtracted from mean
    thresh = cv.adaptiveThreshold(img_blured, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 6)

    # get corners
    corners = get_grid_corners(thresh)

    if corners is not None:
        corners = sort_corners(corners)
        # fit box to size
        kernel = np.ones((2, 2), np.uint8)
        # Opening removes small noise (white dots) from the background
        thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel)
        result = perspective_warp(thresh, corners)
        # split into cells
        cells = extract_cells(result)
        show_first_row(cells)
        sudoku = []
        for cell in cells:
            resized_cell = cv.resize(cell, (28, 28), interpolation=cv.INTER_AREA).astype(float).tolist()
            if np.average(resized_cell) < 0.02:
                sudoku.append('.')
            else:
                tensor_img = torch.tensor(resized_cell).reshape(1, 1, 28, 28)
                sudoku.append(model(tensor_img).argmax().item())
        for i, cell in enumerate(sudoku):
            print(cell, end=" ")
            if (i+1) % 9 == 0: print()

    else:
        print('No corners found!')