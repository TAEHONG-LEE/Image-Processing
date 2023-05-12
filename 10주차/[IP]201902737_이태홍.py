import cv2
import numpy as np
import matplotlib.pyplot as plt



def my_padding(src, ksize, pad_type='zero'):
    # default - zero padding으로 셋팅
    (h, w) = src.shape
    (f_h, f_w) = ksize
    p_h = f_h // 2
    p_w = f_w // 2
    pad_img = np.zeros((h + p_h * 2, w + p_w * 2))
    pad_img[p_h:h + p_h, p_w: w + p_w] = src

    return pad_img


def Spatial2Frequency_mask(n=4):

    """
    :param n:block  크기
    :return: full_mask

    변수 정보
    v, u: 주파수 도메인에서의 좌표
    y, x: 공간 도메인에서의 좌표
    y:
    array([[0, 0, 0, 0],
       [1, 1, 1, 1],
       [2, 2, 2, 2],
       [3, 3, 3, 3]])
    x:
    array([[0, 1, 2, 3],
       [0, 1, 2, 3],
       [0, 1, 2, 3],
       [0, 1, 2, 3]])
    """

    # 4 x 4
    v, u = n, n
    y, x = np.mgrid[0:v, 0:u]

    # mask shape : 16 x 16

    full_mask = np.zeros((n * n, n * n))

    for v_ in range(v):
        for u_ in range(u):

            ##########################################################################
            # TODO
            # TODO mask 만들기
            # TODO sub mask shape : 4 x 4
            # TODO full mask shape = 16 x 16
            # TODO DCT에서 사용된 mask는 4 x 4 mask가 16개 있음 (u, v) 별로 1개씩 있음 u=4, v=4
            # TODO submask 마다 0 ~ 255의 범위를 갖도록 변환 (my_transform 함수 사용)
            # TODO full mask는 각 sub mask로 구성되어있음
            ##########################################################################
            submask = my_transform( np.cos( ((2*x + 1)*u_*np.pi) / (2*n) )  * np.cos( ((2*y+1)*v_*np.pi) / (2*n) ) )
            full_mask[v_ * n:(v_ + 1) * n, u_ * n:(u_ + 1) * n] = submask

    return full_mask


def my_transform(src):
    """

    :param src: sub mask
    :return: dst
    """

    ##############################################################################
    # TODO
    # TODO my_normalize
    # TODO mask를 normalization(0 ~ 1)후 (0 ~ 255)의 값을 갖도록 변환
    ##############################################################################
    dst = src.copy()
    if np.min(src) != np.max(src):
        dst = (dst - np.min(dst)) / (np.max(dst) - np.min(dst)) * 255
    else:
        dst = dst / np.max(dst) * 255

    # dst = np.round(dst).astype(np.uint8)
    return dst


if __name__ == '__main__':

    block_size = 4

    mask = Spatial2Frequency_mask(n=block_size)
    mask = mask.astype(np.uint8)

    true_mask = np.load('mask.npy')
    print(true_mask)
    print("결과 비교 : {}".format(np.array_equal(true_mask, mask)))
    print('transform mask : \n{}'.format(mask))

    # Mask 시각화 (16x16) -> (320x320)
    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)
    mask_visualization = np.zeros((400, 400), dtype=np.uint8)
    ratio = 320 // block_size
    for row in range(block_size):
        for col in range(block_size):

            # 80 x 80 sub mask
            sub = mask[(row * ratio):((row + 1) * ratio),
                  (col * ratio):((col + 1) * ratio)]
            # 100 x 100 sub mask
            extended_sub = my_padding(sub, ksize=(20, 20))

            extended_sub[0:5, :] = 255
            extended_sub[95:, :] = 255

            extended_sub[:, 0:5] = 255
            extended_sub[:, 95:] = 255

            mask_visualization[(row * 100):((row + 1) * 100),
            (col * 100):((col + 1) * 100)] = extended_sub

    cv2.imshow('mask visual', mask_visualization)
    cv2.waitKey()
    cv2.destroyAllWindows()

