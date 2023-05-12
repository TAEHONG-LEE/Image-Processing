import numpy as np
import cv2

def threshold(src, value):
    ret, image = cv2.threshold(src, value, 255, cv2.THRESH_BINARY)
    return image

def Auto_threshold(src):
    ret, image = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    return image

if __name__ == "__main__":

    image = cv2.imread('./threshold_test.png', cv2.IMREAD_GRAYSCALE)

    image_100 = threshold(image, 100)
    image_150 = threshold(image, 150)
    auto_image = Auto_threshold(image)
    cv2.imshow('original', image)
    cv2.imshow('threshold_100', image_100)
    cv2.imshow('threshold_150', image_150)
    cv2.imshow('auto threshold', auto_image)
    cv2.waitKey()