import numpy as np
from PIL import Image
import cv2

def ImportImage(filename):
    try:
        im= Image.open(filename)
    except:
        print("incompatible image file")
filename = "poo.png"
print("hiya")
ImportImage(filename)
