import numpy as np
import cv2
import matplotlib.pyplot as plt



def get_average_filter(mask_size):

    return np.ones((mask_size, mask_size)) / (mask_size ** 2)

def my_padding(src, pad_shape, pad_type='zero'):
    # zero padding인 경우
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        #########################################################
        # TODO                                                  #
        # repetition padding 완성                                #
        #########################################################
        # up
        pad_img[:p_h, p_w:p_w + w] = src[0, :]
        # down
        pad_img[p_h + h:, p_w:p_w + w] = src[h - 1, :]
        # left
        pad_img[:, :p_w] = pad_img[:, p_w:p_w + 1]
        # right
        pad_img[:, p_w + w:] = pad_img[:, p_w + w - 1: p_w + w]
    return pad_img


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2), pad_type)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:m_h + row, col:m_w + col] * mask)
    return dst


def my_normalize(src):

    """
    :param src: 정규화를 적용할 이미지
    :return: unsigned int 타입의 8 bit 이미지
    """

    dst = src.copy()
    dst *= 255
    dst = np.clip(dst,0,255)

    return dst.astype(np.uint8)


def add_gaus_noise(src, mean=0, sigma=0.1):

    """
    :param src: gaussian noise를 적용할 이미지 (0 ~ 255) 사이의 값을 가짐
    :param mean: 평균
    :param sigma: 표준편차
    :return: noise가 추가된 이미지
    """

    dst = src / 255
    (h, w) = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise

    return my_normalize(dst)



def main():

    # 랜덤 값을 생성 시 같은 값이 나오도록 하기 위해 시드 값을 설정
    np.random.seed(seed=100)
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # I_g(x,y) = I(x,y) + N(x,y)
    dst_noise = add_gaus_noise(src, mean=0, sigma=0.1)

    mask = get_average_filter(mask_size=5)

    dst_5x5 = my_filtering(dst_noise.astype(np.float32), mask)
    dst_5x5 = np.round(dst_5x5).astype(np.uint8)

    mask = get_average_filter(mask_size=7)
    dst_7x7 = my_filtering(dst_noise.astype(np.float32), mask)
    dst_7x7 = np.round(dst_7x7).astype(np.uint8)

    mask = get_average_filter(mask_size=9)
    dst_9x9 = my_filtering(dst_noise.astype(np.float32), mask)
    dst_9x9 = np.round(dst_9x9).astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('5 x 5 average dst', dst_5x5)
    cv2.imshow('7 x 7 average dst', dst_7x7)
    cv2.imshow('9 x 9 average dst', dst_9x9)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return




if __name__ == '__main__':
    main()
