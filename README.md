# Hopfield Network

## Overview

This repository contains an implementation of a Hopfield Network for associative memory tasks, specifically focusing on retrieving images even when they are blurred or corrupted. The network is trained on images generated from the VisualCube API.

## Features

- **Associative Memory**: Retrieve original images from blurred or noisy versions.
- **Image Processing**: Apply Gaussian blur and convert images to binary format for training.
- **Dynamic URL Generation**: Fetch images with random rotations using the VisualCube API.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Code Overview](#code-overview)
4. [License](#license)
5. [Contributing](#contributing)
6. [References](#references)

## Installation

Ensure you have the following libraries installed: NumPy, Matplotlib, Pillow, and Requests for various functionalities.

## Usage

### Importing Required Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from PIL import Image, ImageFilter
import requests
from io import BytesIO
```

### Loading and Preprocessing Images

```python
def load_and_preprocess_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    imgblur = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    imgblur = imgblur.convert('L').resize((128, 128))
    img = img.convert('L').resize((128, 128))

    img_array = np.asarray(img) / 255.0
    binary_image = np.where(img_array > 0.3, 1, -1)

    img_arrayblur = np.asarray(imgblur) / 255.0
    binary_imageblur = np.where(img_arrayblur > 0.3, 1, -1)
    
    return binary_image.flatten(), binary_imageblur.flatten()
```

### Creating the Hopfield Network Class

```python
class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def retrieve(self, pattern, max_iterations=500):
        state = pattern.copy()
        for _ in range(max_iterations):
            for i in range(self.num_neurons):
                update = np.dot(self.weights[i], state)
                state[i] = 1 if update >= 0 else -1
        return state
```

### Training and Retrieving Images

```python
# Train the network
hopfield_net = HopfieldNetwork(num_neurons=128*128)
hopfield_net.train(imgs)

# Display and retrieve a blurred image
blurred_image = imgsblur[0]
plot_image(blurred_image, "Blurred Image 1")
retrieved_image = hopfield_net.retrieve(blurred_image)
plot_image(retrieved_image, "Retrieved Image 1")
```

## License

This project is licensed under the Apache License 2.0. See the LICENSE file for more details.

## Contributors

- Vyom
- Abhisar
- Vihaan

## References

- Hopfield, J.J. (1982). "Neural networks and physical systems with emergent collective computational abilities". *Proceedings of the National Academy of Sciences*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
