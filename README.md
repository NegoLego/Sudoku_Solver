# Sudoku Solver

Just upload a photo of a Sudoku and find its solution, along with all the steps it went through to get there.
Try it [here](https://sudoku-solver-app.herokuapp.com/)!

## How It Works

The app identifies the sudoku using OpenCV functions and uses a small model to understand the digits in each written cell.

## Requirements

To be able to detect your sudoku, its edges should be clearly visible and uninterrupted.

## Why This Project?

I know many of you enjoy solving sudoku by yourselves, but this is a great way to see computer vision in action, understand a little bit about AI and maybe make you curious about building your own.

## Model Training

The model was trained in 7 minutes, locally, on a laptop, on the MNIST dataset and a dataset created by taking 8 fonts and augmenting with different transformations.

## Known Limitations

Sometimes the predictions are off because of background noise (stains, spots on the paper), so the dataset can still be augmented with different types of background textures, and have a slightly bigger model to understand that.

## Getting Started

Download the project and play with the notebooks to dig deeper into this project and understand its creation process better.

Take what you learned and make it better! You can contact me anytime at negoitaa10@gmail.com .
