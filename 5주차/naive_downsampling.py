import numpy as np
import matplotlib.pyplot as plt
import cv2


def my_naive_downsampling(src, ratio):
    return src[::ratio, ::ratio]

def my_navie_pyramid(src, ratio, pyramid_lvl):

    pyramid = [src]
    downsampled_imgs = []

    for i in range(pyramid_lvl):
        downsampled_img = my_naive_downsampling(pyramid[-1], ratio)
        downsampled_imgs.append(downsampled_img)
        pyramid.append(downsampled_img)
    return pyramid, downsampled_imgs


def main():

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    pyramid, downsampled_imgs = my_navie_pyramid(src, ratio=2, pyramid_lvl=3)

    cv2.imshow('original', src)
    cv2.imwrite('original_gray.png',src)


    for level in range(len(downsampled_imgs)):
        cv2.imshow('down level {}'.format(level), downsampled_imgs[level])
        cv2.imwrite('down_level{}.png'.format(level), downsampled_imgs[level])


    cv2.waitKey()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()