import cv2
import numpy as np

def get_my_sobel():
    sobel_x = np.dot(np.array([[1],[2],[1]]),np.array([[-1,0,1]]))
    sobel_y = np.dot(np.array([[-1],[0],[1]]),np.array([[1,2,1]]))

    return sobel_x,sobel_y


if __name__ == '__main__':

    src = cv2.imread("./Lena.png",cv2.IMREAD_GRAYSCALE)
    sobel_x , sobel_y = get_my_sobel()
    dst_x = cv2.filter2D(src.astype(np.float32), -1, sobel_x, borderType=cv2.BORDER_CONSTANT)
    dst_y = cv2.filter2D(src.astype(np.float32), -1, sobel_y, borderType=cv2.BORDER_CONSTANT)

    dst_magnitude = np.sqrt((dst_x ** 2) + (dst_y ** 2)) / 255
    dst = np.abs(dst_x) + np.abs(dst_y)

    # min-max scaler
    # 0 ~ 1 사이의 값으로 변경
    dst_x_Norm = (dst_x - np.min(dst_x)) / (np.max(dst_x) - np.min(dst_x))
    dst_y_Norm = (dst_y - np.min(dst_y)) / (np.max(dst_y) - np.min(dst_y))

    dst = dst / 255

    #참고 : 실수로 표현하는 경우 0 이하의 값은 전부 검은 색 1 이상의 값은 전부 흰색으로 사용 중간 값은 회색

    cv2.imshow('original',src)
    cv2.imshow('dst_x_Norm',dst_x_Norm)
    cv2.imshow('dst_y_Norm',dst_y_Norm)
    cv2.imshow('dst_magnitude', dst_magnitude)
    cv2.imshow('dst',dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


