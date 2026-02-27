import os
import time
import numpy as np
import cv2 as cv
from flask import Flask, render_template, request, url_for, jsonify
import base64
import sudoku_solver
from sudoku_solver import perspective_warp, split_into_cells, process_cell


def image_to_base64(img):
    # 1. Encode image to memory buffer (like saving to a virtual file)
    _, buffer = cv.imencode('.jpg', img)
    # 2. Convert bytes to Base64 string
    img_str = base64.b64encode(buffer).decode('utf-8')
    # 3. Add the HTML prefix so the browser understands it
    return f"data:image/jpeg;base64,{img_str}"

def visualize_contours(img, contours):
    vis_img = img.copy()
    return cv.drawContours(vis_img, contours, -1, (0, 255, 0), 3)

def visualize_corners(img, points):
    vis_img = img.copy()
    points = points.astype(int)

    for i, pt in enumerate(points):
        cv.circle(vis_img, tuple(pt), 18, (0, 0, 255), -1)
        cv.putText(vis_img, str(i), tuple(pt), cv.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 5, cv.LINE_AA)

    return vis_img

def stage(img, dtime):
    return {
        'image': image_to_base64(img),
        'time': f"{dtime:.5f}s"
    }

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    # just serve the static HTML page
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # convert file stream to numpy array
    np_img = np.frombuffer(file.read(), np.uint8)
    image = cv.imdecode(np_img, cv.IMREAD_COLOR)

    stages = {
        'extract_sudoku': [],
        'cell_images': {},
        'predictions': {},
        'final_image': {}
    }

    prev_time = time.time()
    gray_image = sudoku_solver.gray(image)
    now_time = time.time()
    stages['extract_sudoku'].append(stage(gray_image, now_time - prev_time))

    prev_time = now_time
    blurred_image = sudoku_solver.blur(gray_image)
    now_time = time.time()
    stages['extract_sudoku'].append(stage(blurred_image, now_time - prev_time))

    prev_time = now_time
    thresholded_image = sudoku_solver.threshold(blurred_image)
    now_time = time.time()
    stages['extract_sudoku'].append(stage(thresholded_image, now_time - prev_time))

    prev_time = now_time
    contours = sudoku_solver.get_contours(thresholded_image)
    contours_image = visualize_contours(image, contours)
    now_time = time.time()
    stages['extract_sudoku'].append(stage(contours_image, now_time - prev_time))

    prev_time = now_time
    corners = sudoku_solver.get_grid_corners(contours)
    if corners is None:
        return jsonify(stages), 200
    else:
        corners_image = visualize_corners(image, corners)
        now_time = time.time()
        stages['extract_sudoku'].append(stage(corners_image, now_time - prev_time))

    prev_time = now_time
    grid_image, persp_matrix = sudoku_solver.perspective_warp(image, corners)
    now_time = time.time()
    stages['extract_sudoku'].append(stage(grid_image, now_time - prev_time))

    prev_time = now_time
    cells = [process_cell(cell) for cell in split_into_cells(grid_image)]
    now_time = time.time()
    stages['cell_images'] = {
        'images': [image_to_base64(cell['image']) for cell in cells],
        'isNumber': [cell['isNumber'] for cell in cells],
        'time': f"{(now_time - prev_time):.5f}s"
    }

    if len([cell for cell in cells if cell['isNumber']]) < 15:
        return jsonify(stages), 200

    prev_time = now_time
    sudoku = [sudoku_solver.predict(cell['image']) if cell['isNumber'] else 0 for cell in cells]

    now_time = time.time()
    stages['predictions'] = {
        'predictions': sudoku,
        'time': f"{(now_time - prev_time):.5f}s"
    }
    sudoku = [sudoku[i: i + 9] for i in range(0, len(sudoku), 9)]  # make matrix from list

    prev_time = now_time
    solved_sudoku = sudoku_solver.solveSudoku(sudoku)
    final_image = sudoku_solver.draw_solution_on_original(image, persp_matrix, solved_sudoku, sudoku)
    now_time = time.time()
    stages['final_image'] = {
        'image': image_to_base64(final_image),
        'time': f"{(now_time - prev_time):.5f}s"
    }

    return jsonify(stages), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=7860, debug=False)