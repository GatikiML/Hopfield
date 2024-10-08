# -*- coding: utf-8 -*-
"""
Hopfield.ipynb
Original file is located at
    https://colab.research.google.com/drive/1YWYdjbddQZJUtcpn7BoGAQWOnQDiBTT9?usp=sharing
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import random
from PIL import Image, ImageFilter
import requests
from io import BytesIO

# Function to display an image
def plot_image(image, title="Image"):
    plt.imshow(image.reshape((128, 128)), cmap='gray')
    plt.title(title)
    plt.show()

# Base URL for the visualcube API
base_url = "https://visualcube.api.cubing.net/visualcube.php?fmt=jpg&size=150&pzl=2"

# List to store URLs
image_urls = []

# Generate URLs with random x and y rotations
for _ in range(3):
    # Generate random angles for x and y rotations
    x_angle = random.randint(-180, 180)  # Random angle between -180 and 180 degrees for x-axis
    y_angle = random.randint(-180, 180)  # Random angle between -180 and 180 degrees for y-axis

    # Construct the URL with the random x and y angles
    url = f"{base_url}&r=x{x_angle}y{y_angle}"

    # Append the generated URL to the list
    image_urls.append(url)

# Check the list of generated URLs
print(image_urls[0])  # Print the first 10 URLs as a sample

# Function to load an image from a URL and convert it to a binary 32x32 image
def load_and_preprocess_image_from_url(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    imgblur = img.filter(ImageFilter.GaussianBlur(radius=1.5))
    imgblur = imgblur.convert('L')
    imgblur = imgblur.resize((128,128))
    img = img.convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to 32x32

    # Convert the image to a binary format (1 for white, -1 for black)
    img_array = np.asarray(img) / 255.0  # Normalize pixel values to [0, 1]
    binary_image = np.where(img_array > 0.3, 1, -1)  # Convert to binary: 1 or -1

    img_arrayblur = np.asarray(imgblur) / 255.0  # Normalize pixel values to [0, 1]
    binary_imageblur = np.where(img_arrayblur > 0.3, 1, -1)  # Convert to binary: 1 or -1
    return binary_image.flatten(), binary_imageblur.flatten()  # Flatten into 1D array

imgs = []
imgsblur = []

for url in image_urls:
  image, image_blured = load_and_preprocess_image_from_url(url)
  image_flattened = preprocess_image(image)
  image_flattenedblur = preprocess_image(image_blured)
  imgs.append(image_flattened)
  imgsblur.append(image_flattenedblur)

class HopfieldNetwork:
    def __init__(self, num_neurons):
        self.num_neurons = num_neurons
        self.weights = np.zeros((num_neurons, num_neurons))

    # Hebbian learning
    def train(self, patterns):
        for pattern in patterns:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # No self-connections
        self.weights /= len(patterns)

    def retrieve(self, pattern, max_iterations=500):
        state = pattern.copy()
        for _ in range(max_iterations):
            for i in range(self.num_neurons):
                update = np.dot(self.weights[i], state)
                state[i] = 1 if update >= 0 else -1
        return state


def preprocess_image(image):
    return np.where(image.flatten() > 0.5, 1, -1)

def add_noise(image, noise_level=0.1):
    noisy_image = image.copy()
    num_flips = int(noise_level * len(image))
    flip_indices = np.random.choice(range(len(image)), num_flips, replace=False)
    noisy_image[flip_indices] *= -1
    return noisy_image

# Train
hopfield_net = HopfieldNetwork(num_neurons=128*128)
hopfield_net.train(imgs)

# Add blur or noise to the image to simulate degradation
blurred_image = imgsblur[0]
plot_image(blurred_image, "Blurred Image 1")

# Retrieve the original image from the blurred input
retrieved_image = hopfield_net.retrieve(blurred_image)
plot_image(retrieved_image, "Retrieved Image 1")
