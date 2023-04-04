import cv2
import numpy as np

def my_average_filter1(src):

    """
    3 x 3 행렬
    필터의 총 합은 1

    mask

    [[1/9,1/9,1/9],
    [1/9,1/9,1/9],
    [1/9,1/9,1/9]]

    """
    mask = np.ones((3,3))
    mask = mask / np.sum(mask)
    print(mask)
    print("mask 총합 : {}".format(np.sum(mask)))
    dst = cv2.filter2D(src,  -1, mask)
    return dst


def my_average_filter2(src):

    """
    3 x 3 행렬
    필터의 총 합이 1보다 낮은 경우

    mask

    [[1/12, 1/12,1/12],
    [1/12, 1/12, 1/12],
    [1/12, 1/12, 1/12]]

    """
    mask = np.ones((3, 3)) / 12
    print("mask 총합 : {}".format(np.sum(mask)))
    dst = cv2.filter2D(src,-1,mask)
    return dst


def my_average_filter3(src):

    """
    3 x 3 행렬
    필터의 총 합이 1보다 큰 경우

    mask

    [[1/4, 1/4, 1/4],
    [1/4, 1/4, 1/4],
    [1/4, 1/4, 1/4]]

    """
    mask = np.ones((3, 3)) / 4
    print("mask 총합 : {}".format(np.sum(mask)))
    dst = cv2.filter2D(src,-1,mask)
    return dst

def my_sharpening_filter(src):
    """
    3 x 3 행렬
    필터의 총 합이 1보다 큰 경우

    mask

    [[-1/9, -1/9, -1/9],
    [-1/9, -17/9, -1/9],
    [-1/9, -1/9, -1/9]]

    """
    mask1 = np.zeros((3, 3))
    mask1[1,1] = 2

    # 평균 필터
    mask2 = np.ones((3, 3)) / 9

    # sharpening mask
    # 평균 밝기는 변화 없음
    mask = mask1 - mask2
    print("mask 총합 : {}".format(np.sum(mask)))

    dst = cv2.filter2D(src,-1, mask)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png',cv2.IMREAD_GRAYSCALE)

    dst1 = my_average_filter1(src)
    dst2 = my_average_filter2(src)
    dst3 = my_average_filter3(src)
    dst4 = my_sharpening_filter(src)

    cv2.imshow('original',src)
    cv2.imshow('my filtering img1', dst1)
    cv2.imshow('my filtering img2', dst2)
    cv2.imshow('my filtering img3', dst3)
    cv2.imshow('my sharpening img', dst4)

    cv2.waitKey()
    cv2.destroyAllWindows()

