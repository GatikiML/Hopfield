# Hopfield Network

## Overview

This repository contains an implementation of a Hopfield Network for associative memory tasks, specifically focusing on retrieving images even when they are blurred or corrupted. The network is trained on images generated from the VisualCube API.

## Features

- **Associative Memory**: Retrieve original images from blurred or noisy versions.
- **Image Processing**: Train the network on images and retrive orignal images using blured or lower resolution versions of the image.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [License](#license)
4. [Contributors](#contributors)
5. [References](#references)


## Installation

To set up the Hopfield Network project, first install the python packages. Run the following command in your terminal/powershell:

```bash
pip install hopfield-memory
```
or
```bash
python -m pip install hopfield-memory
```


## Usage

### Import Packages

```python
import hopfield
from matplotlib import pyplot as plt
```

### Initializing the Hopfield Network

```python
hf = HopfieldNetwork(resolution=128*128)
```

### Training and Retrieving Images

```python
#Image URLs / Path / PIL Image Object list
images = ['https://random.com/image.jpg', 'https://random.com/image2.jpg']

# Train the network
hf.train(images, url=True)

# Display a Blurred/Noisy image
blurred_image = 'https://random.com/blurredimage.jpg'
blur = plt.imread(blurred_image)
plt.imshow(blur)
plt.show()

#Retrive the Orignal image from the Blurred/Noisy Image
retrieved_state = hf.retrieve(blurred_image, url=True)
plot_state(retrieved_image, "Retrived Image", pixel=128)
```

### Complete Code

```python
import hopfield
from matplotlib import pyplot as plt

hf = HopfieldNetwork(resolution=128*128)

#Image URLs / Path / PIL Image Object list
images = ['https://random.com/image.jpg', 'https://random.com/image2.jpg']

# Train the network
hf.train(images, url=True)

# Display a Blurred/Noisy image
blurred_image = 'https://random.com/blurredimage.jpg'
blur = plt.imread(blurred_image)
plt.imshow(blur)
plt.show()

#Retrive the Orignal image from the Blurred/Noisy Image
retrieved_state = hf.retrieve(blurred_image, url=True)
plot_state(retrieved_image, "Retrived Image", pixel=128)
```
For Example Code File Refer To ``Example.py``

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributors

- Vyom Nishant Patel¹
- Abhisar Mehta
- Vihaan Mishra

  1 - Equal Contribution  

## References

- Hopfield, J.J. (1982). "Neural networks and physical systems with emergent collective computational abilities". *Proceedings of the National Academy of Sciences*.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

## Thanks 😊

``
Thank you for your patience, and let us know for any suggestions ✨️🤗
``
