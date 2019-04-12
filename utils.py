import numpy as np
from PIL import Image

class GrayscaleToRgb:
    """Convert a grayscale image to rgb"""
    def __call__(self, image):
        print(image)
        image = np.array(image)
        print(image)
        image = np.dstack([image, image, image])
        return Image.fromarray(image)
