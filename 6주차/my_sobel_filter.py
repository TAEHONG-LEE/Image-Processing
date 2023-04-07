import cv2
import numpy as np


def get_my_sobel():
    sobel_x = np.dot(np.array([[1],[2],[1]]),np.array([[-1,0,1]]))
    sobel_y = np.dot(np.array([[-1],[0],[1]]),np.array([[1,2,1]]))

    return sobel_x,sobel_y


if __name__ == '__main__':

    tower_img = cv2.imread('sobel_test.png', cv2.IMREAD_GRAYSCALE)
    sobel_x , sobel_y = get_my_sobel()

    tower_x = cv2.filter2D(tower_img, -1, sobel_x, borderType=cv2.BORDER_CONSTANT)
    tower_y = cv2.filter2D(tower_img, -1, sobel_y, borderType=cv2.BORDER_CONSTANT)

    cv2.imshow('original', tower_img)
    cv2.imshow('tower x', tower_x)
    cv2.imshow('tower y', tower_y)
    cv2.waitKey()
    cv2.destroyAllWindows()

    src = cv2.imread("./edge_detection_img.png", cv2.IMREAD_GRAYSCALE).astype(np.float32)
    dst_x = cv2.filter2D(src, -1, sobel_x, borderType=cv2.BORDER_CONSTANT) / 255
    dst_y = cv2.filter2D(src, -1, sobel_y, borderType=cv2.BORDER_CONSTANT) / 255

    abs_x = np.abs(dst_x)
    abs_y = np.abs(dst_y)
    #dst = np.sqrt(dst_x**2 + dst_y**2) 이것과 같다
    #dst = np.hypot(dst_x,dst_x)
    dst = abs_x + abs_y

    # min-max scaler
    # 0 ~ 1 사이의 값으로 변경
    dst_x_Norm = (dst_x - np.min(dst_x)) / (np.max(dst_x) - np.min(dst_x))
    dst_y_Norm = (dst_y - np.min(dst_y)) / (np.max(dst_y) - np.min(dst_y))

    dst_Norm = dst_x_Norm + dst_y_Norm / 2.0

    #참고 : 실수로 표현하는 경우 0 이하의 값은 전부 검은 색 1 이상의 값은 전부 흰색으로 사용 중간 값은 회색

    cv2.imshow('original',src)
    cv2.imshow('dst_x',dst_x)
    cv2.imshow('dst_y',dst_y)
    cv2.imshow('abs_x',abs_x)
    cv2.imshow('abs_y',abs_y)
    cv2.imshow('dst_x_Norm',dst_x_Norm)
    cv2.imshow('dst_y_Norm',dst_y_Norm)
    cv2.imshow('dst_Norm',dst_Norm)
    cv2.imshow('dst',dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


