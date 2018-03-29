import tensorflow as tf
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


def crop_image_by_bb(image_url, box):
    img = cv2.imread(image_url, cv2.IMREAD_UNCHANGED)
    x1, y1, x2, y2 = box
    return img[y1:y2, x1:x2]


def resize_image(image, dim):
    return cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)


def crop_n_resize(image, box, dim):
    return resize_image(crop_image_by_bb(image, box), dim)


def display(img, title):
    cv2.imshow(title, img)


def esc2quit():
    while True:
        keycode = cv2.waitKey()
        if keycode != -1:
            if keycode == 27:
                break
    cv2.destroyAllWindows()
