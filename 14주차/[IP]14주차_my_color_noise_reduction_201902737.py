import numpy as np
import cv2
import matplotlib.pyplot as plt


def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1,
           -(msize // 2):(msize // 2) + 1]

    gaus2D = 1 / (2 * np.pi * sigma ** 2) * \
             np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    gaus2D /= np.sum(gaus2D)
    return gaus2D


def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)


def my_padding(src, pad_shape):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src
    return pad_img


def my_filtering(src, mask):
    """

    :param src: filtering 할 이미지
    :param mask: filtering 할 mask
    :return: filtering된 결과 이미지
    """

    (h, w) = src.shape
    # mask의 크기
    (m_h, m_w) = mask.shape
    # 직접 구현한 my_padding 함수를 이용
    pad_img = my_padding(src, (m_h // 2, m_w // 2))
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            val = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
            dst[row, col] = val
    return dst


def add_gaus_noise(src, mean=0, sigma=0.1):
    # src : 0 ~ 255, dst : 0 ~ 1
    dst = src / 255
    h, w, c = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w, c))
    dst += noise
    return my_normalize(dst)


def my_median_filtering(src, msize):
    """
    중앙값 필터링을 수행하는 함수
    :param src: 필터링할 이미지
    :param msize: 필터 크기
    :return: 필터링된 이미지
    """

    h, w = src.shape[:2]
    padded_image = my_padding(src, (msize // 2, msize // 2))
    dst = np.zeros_like(src)

    for row in range(h):
        for col in range(w):
            neighbors = padded_image[row:row + msize, col:col + msize].flatten()
            median = np.sort(neighbors)[len(neighbors) // 2]
            dst[row, col] = median

    return dst

if __name__ == '__main__':
    src = cv2.imread('canoe.png')
    np.random.seed(seed=100)

    # noise image: uint8 형
    noise_image = add_gaus_noise(src, mean=0.1, sigma=0.1)
    # noise image: float 형
    src_noise = noise_image / 255

    ######################################################
    # TODO
    # TODO RGB에서 Gaussian, Median filter 진행
    # TODO Filtering된 결과 이미지는 uint8형 타입이어야함
    #  -> (my_normalize 사용)
    ######################################################

    gaus2D = my_get_Gaussian2D_mask(msize=3, sigma=5)

    rgb_gaussian_dst = np.zeros_like(src_noise)
    rgb_median_dst = np.zeros_like(src_noise)

    for channel in range(3):
        rgb_gaussian_dst[:, :, channel] = my_filtering(src_noise[:, :, channel], gaus2D)
        rgb_median_dst[:, :, channel] = my_median_filtering(src_noise[:, :, channel], msize=3)

    # normalize rgb_gaussian_dst, rgb_median_dst
    rgb_gaussian_dst = my_normalize(rgb_gaussian_dst)
    rgb_median_dst = my_normalize(rgb_median_dst)

    ######################################################
    # TODO
    # TODO YUV에서 Gaussian, Median filter 진행
    # TODO Filtering된 결과 이미지는 uint8형 타입이어야함
    #  -> (my_normalize 사용)
    # TODO yuv noise 이미지는 내장 모듈 사용 (cv2.COLOR_BGR2YUV)
    # TODO yuv_noise에서 첫번째 채널이 Y
    ######################################################

    gaus2D = my_get_Gaussian2D_mask(msize=3, sigma=5)
    yuv_noise = cv2.cvtColor(noise_image, cv2.COLOR_BGR2YUV)

    yuv_gaussian_dst = yuv_noise / 255
    yuv_gaussian_dst[:, :, 0] = my_filtering(yuv_gaussian_dst[:, :, 0], gaus2D)
    yuv_gaussian_dst = my_normalize(yuv_gaussian_dst)

    yuv_median_dst = yuv_noise / 255
    yuv_median_dst[:, :, 0] = my_median_filtering(yuv_median_dst[:, :, 0], msize=3)
    yuv_median_dst = my_normalize(yuv_median_dst)

    cv2.imshow('original', src)
    cv2.imshow('noise image', noise_image)
    cv2.imshow('RGB + gaussian filtering', rgb_gaussian_dst)
    cv2.imshow('RGB + median filtering', rgb_median_dst)
    cv2.imshow('YUV + gaussian filtering', cv2.cvtColor(yuv_gaussian_dst, cv2.COLOR_YUV2BGR))
    cv2.imshow('YUV + median filtering', cv2.cvtColor(yuv_median_dst, cv2.COLOR_YUV2BGR))
    cv2.waitKey()
    cv2.destroyAllWindows()