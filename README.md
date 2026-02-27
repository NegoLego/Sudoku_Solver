---
title: Sudoku Solver
emoji: 🔥
colorFrom: gray
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Upload a photo of your sudoku to find its solution
---
# Sudoku Solver

Just upload a photo of your Sudoku and find its solution, along with all the steps it went through to get there.
Try it here: **[:rocket: Sudoku Solver!](https://huggingface.co/spaces/NegoLego/Sudoku_Solver)**

### How It Works

- The front-end saves your image in a form and makes a request to the back-end.  
- The back-end is a Python Flask server that processes the image and builds a response with all the stages.
- The response is then unpacked by the front-end to fill the HTML page with the images. 

All the functions called for solving the sudoku are found in the [`sudoku_solver.py`](./sudoku_solver.py) file, which uses the weights saved at [`models/CNN_Ultra.pth`](./models/CNN_Ultra.pth).
Downloading these 2 files means that you can `import sudoku_solver` into your own project, just like I do in [`app.py`](./app.py).

### Why This Project?

I know many of you enjoy solving sudoku by yourselves, but this is a great way to see computer vision in action, understand a little bit about AI and maybe make you curious about building your own. :robot:

### About the Model

The model is built with PyTorch and has:
- convolutional stage:
  - Thirty-two 3x3 filters to detect basic shapes, like simple curves and straight edges.
  - Steps up to sixty-four 3x3 filters to combine those basic shapes into more complex features, like loops and sharp corners. → 3136 values
- fully connected stage:
  - Dense layer maps those 3136 values down to 128. (I also used a Dropout technique to ensure the model doesn't overfit on the custom fonts)
  - Final output layer maps the 128 values to 10 (representing our possible digit classes, 0-9)
  
The model was trained in 7 minutes, locally, on a laptop, on the MNIST dataset and a dataset created by taking eight fonts and augmenting with various transformations.  
See the training notebook: [`notebooks/CNN_Ultra_model.ipynb`](./notebooks/CNN_Ultra_model.ipynb) and the dataset creation notebook: [`notebooks/augment_dataset.ipynb`](./notebooks/augment_dataset.ipynb). They are a lot of fun:

<img width="544" height="707" alt="notebooks" src="https://github.com/user-attachments/assets/fc5a31bb-819c-4b35-abac-4732b06e5735" />

### App Requirements

To be able to detect your sudoku, its edges should be clearly visible and uninterrupted.

### Known Limitations

Sometimes the predictions are off because of background noise (stains, spots on the paper). This means there is still room for improvement. The dataset can be augmented further with different types of background textures, and have a slightly bigger model to understand that.

### Getting Started

Download the project and play with the notebooks to dig deeper into this project and understand its creation process better.
Or just use it here: **[:rocket: Sudoku Solver!](https://huggingface.co/spaces/NegoLego/Sudoku_Solver)**

Take what you learned and make it better! You can contact me anytime at negoitaa10@gmail.com .
