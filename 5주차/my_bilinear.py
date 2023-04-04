import cv2
import numpy as np

def my_bilinear(src, scale):

    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst), dtype=np.float32)

    print('original shape : {}'.format(src.shape))
    print('dst shape : {}'.format(dst.shape))

    for row in range(h_dst):
        for col in range(w_dst):
            # 스케일링 되기 전 original 좌표
            y = row / scale
            x = col / scale

            # bilinear interpolation
            # 1.(y,x)를 기준으로 좌측위, 좌측아래, 우측아래, 우측위 좌표를 구함.
            # 2. bilinear interplation 식에 따라 해당 row,col좌표에 값을 대입
            y_up = int(y) # 버림
            y_down = min(int(y+1), h-1) # 반올림 단 src의 최대 좌표값보다는 같거나 작게
            x_left = int(x) # 버림
            x_right = min(int(x+1), w-1) # 반올림 단 src의 최대 좌표값보다는 같거나 작게

            t = y - y_up
            s = x - x_left

            intensity = ((1 - s) * (1 - t) * src[y_up, x_left]) \
                        + (s * (1 - t) * src[y_up, x_right]) \
                        + ((1 - s) * t * src[y_down, x_left]) \
                        + (s * t * src[y_down, x_right])
            dst[row, col] = intensity

    dst = np.round(dst).astype(np.uint8)
    return dst

if __name__ == '__main__':
    img = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    down_cv2 = cv2.resize(img, dsize=(0,0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    down_up_cv2 = cv2.resize(down_cv2, dsize=(0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_LINEAR)

    down_my = my_bilinear(img, scale=0.25)
    down_up_my = my_bilinear(down_my, scale=4.0)

    cv2.imshow('original image', img)
    cv2.imshow('down_cv2_n image', down_cv2)
    cv2.imshow('down_up_cv2_n', down_up_cv2)
    cv2.imshow('down_my', down_my)
    cv2.imshow('down_up_my', down_up_my)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

