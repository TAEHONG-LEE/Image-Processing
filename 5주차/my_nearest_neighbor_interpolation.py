import cv2
import numpy as np
import matplotlib.pyplot as plt

def my_nearest_neighbor(src, scale=None,shape=None):

    """
    함수 인자 정보
    src: gray scale 이미지 (H x W)
    scale: 크기를 확대 또는 축소 시킬 비율 또는 구체적인 값
    """

    (h, w) = src.shape

    # scale이 지정된 경우
    if scale is not None:
        h_scale, w_scale = scale
        h_dst = int(h * h_scale + 0.5)
        w_dst = int(w * w_scale + 0.5)

    # scale이 지정 안되고 구체적인 dst 크기가 지정된 경우
    else:
        h_dst, w_dst = shape
        h_scale = h_dst / h
        w_scale = w_dst / w

    dst = np.zeros((h_dst, w_dst), dtype=np.uint8)
    for row in range(h_dst):
        for col in range(w_dst):
            # int(): 소수점 이하를 버림
            r = min(int(row / h_scale), h - 1)
            c = min(int(col / w_scale), w - 1)
            dst[row, col] = src[r, c]

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    print('original shape: {}'.format(src.shape))

    # scale은 사용자 정의 (크기를 줄이는 방향으로)
    h, w = src.shape
    d_h, d_w = int(h / 7), int(w / 7)

    dst_nearest_div2 = cv2.resize(src, dsize=(d_h, d_w),
                                  interpolation=cv2.INTER_NEAREST)
    print('cv2 method out shape : {}'.format(dst_nearest_div2.shape))

    dst_my_nearest_div2 = my_nearest_neighbor(src, shape=(d_h, d_w))

    print('my method out shape : {}'.format(dst_my_nearest_div2.shape))

    print("comparision : {}".format(np.array_equal(dst_nearest_div2, dst_my_nearest_div2)))

    cv2.imshow('original',src)
    cv2.imshow('cv2 dst_nearest 1/7', dst_nearest_div2)
    cv2.imshow('my dst_nearest 1/7', dst_my_nearest_div2)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 크기를 늘리는 방향으로(보통 원래 크기로 돌아오게 설정)
    up_h, up_w = h, w

    dst_nearest = cv2.resize(dst_nearest_div2, dsize=(up_h, up_w),
                             interpolation=cv2.INTER_NEAREST)

    print('cv2 out shape : {}'.format(dst_nearest.shape))

    my_dst_nearest = my_nearest_neighbor(dst_my_nearest_div2, shape=(up_h, up_w))

    print('my dst shape : {}'.format(my_dst_nearest.shape))

    print("comparision: {}".format(np.array_equal(dst_nearest, my_dst_nearest)))

    cv2.imshow('original', src)
    cv2.imshow('cv2 dst_nearest', dst_nearest)
    cv2.imshow('my dst_nearest', my_dst_nearest)

    cv2.waitKey()
    cv2.destroyAllWindows()

