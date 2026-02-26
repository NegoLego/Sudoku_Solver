const pyFuncs = [
    `def gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
`,
    `def blur(image):
    # GaussianBlur(src, kernel_size, standard_deviation)
    return cv.GaussianBlur(image, (7, 7), 3)
`,
    `def threshold(image):
    '''
    adaptiveThreshold(src, 
                      true_value, 
                      adaptiveMethod, 
                      thresholdType, 
                      blockSize, 
                      constant)
    '''
    return cv.adaptiveThreshold(image, 
                                255, 
                                cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv.THRESH_BINARY_INV, 
                                11, 
                                2)
`,
    `def get_contours(image):
    '''
    findContours(image, 
                 mode, 
                 method) -> contours, hierarchy
    '''
    contours, _ = cv.findContours(image, 
                                  cv.RETR_EXTERNAL, 
                                  cv.CHAIN_APPROX_SIMPLE)
    return contours
`,
    `def get_grid_corners(contours):
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
        s = points.sum(axis=1)  
        diff = np.diff(points, axis=1)  # Note: np.diff is [y-x]

        sorted_points = np.zeros((4, 2), dtype="float32")

        sorted_points[0] = points[np.argmin(s)]  # Top-left (min sum)
        sorted_points[2] = points[np.argmax(s)]  # Bottom-right (max sum)
        sorted_points[1] = points[np.argmin(diff)]  # Top-right (min diff)
        sorted_points[3] = points[np.argmax(diff)]  # Bottom-left (max diff)

        return sorted_points
    return None
`,
    `# Here, we take the initial image and use the corners identified
# to cut a part of it and warp it to a square

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
`,
    `def split_into_cells(image):
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
`,
    `class Model(nn.Module):
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
        self.dropout = nn.Dropout(0.5) # Crucial to prevent overfitting
        self.fc2 = nn.Linear(128, num_classes)
`,
    `# Classical backtracking algorithm
    
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
`];

export default pyFuncs;