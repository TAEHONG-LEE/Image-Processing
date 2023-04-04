import cv2
import numpy as np

def gaussian_2D_mask(filter_size, sigma):
    mask = cv2.getGaussianKernel(filter_size, sigma)
    gaussian_mask = mask @ mask.T
    return gaussian_mask

def gaussian_1D_mask(filter_size, sigma):
    mask = cv2.getGaussianKernel(filter_size, sigma)
    return mask



if __name__ == '__main__':

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    gaussian_2d = gaussian_2D_mask(filter_size=5, sigma=3)
    dst_gaus_2d = cv2.filter2D(src, -1, gaussian_2d)

    cv2.imshow('original', src)
    cv2.imshow('gaus 2d filtering', dst_gaus_2d)
    cv2.waitKey()
    cv2.destroyAllWindows()

    print("gaussian_2d")
    print(gaussian_2d)


    gaussian_1d = gaussian_1D_mask(filter_size=5, sigma=1)
    print("gaussian_1d")
    print(gaussian_1d)







