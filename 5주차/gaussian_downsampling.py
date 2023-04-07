import numpy as np
import matplotlib.pyplot as plt
import cv2
import gaussian_blur
def my_gaussian_downsampling(src, ratio, mask):

    """
    함수 인자 정보
    :param src: gray 이미지( H x W)
    :param ratio: downsampling 할 비율
    :param mask: gaussian filter

    변수 정보
    blur_src : gray 이미지에 gaussian filter을 적용한 이미지
    downsampled_blur_img: blur_src에 ratio만큼 downsamling을 적용한 결과 이미지

    :return: downsampled_blur_img
    """

    # gaussian filtering - 내장 함수 사용
    blur_src = cv2.filter2D(src.astype(np.float32), -1, mask)


    # (h, w) = src.shape
    # blur_src = gaussian_blur.my_gaussian_filter(src, (h, w), sigma=3)
    # blur_src = np.round(blur_src).astype(np.uint8)

    # ratio 만큼 행과 열을 추출
    downsampled_blur_img = blur_src[::ratio, ::ratio]

    return downsampled_blur_img


def my_gaussian_pyramid(src, ratio, pyramid_lvl, filter_size, sigma):

    """
    인자 정보
    :param src: gray 이미지( H x W)
    :param ratio: downsampling 할 비율
    :param pyramid_len:  피라미드의 층 개
    :param filter_size: gaussian filter 크
    :param sigma: gaussian filter sigma
    :return:
    """

    pyramid = [src.astype(np.float32)]
    gaussian_filter = cv2.getGaussianKernel(ksize=filter_size, sigma=sigma)
    gaussian_filter = gaussian_filter @ gaussian_filter.T
    previous_imgs = [src]
    filtered_imgs = []
    downsample_imgs = []

    for i in range(pyramid_lvl):

        ######################################################
        # Gaussian pyramid 진행 절차 (교수님 PPT 41page 참고)
        # 1. filtering
        # 2. subsample(즉 downsampling 이미지 크기 줄이기)
        ######################################################

        # 1. filtering
        filtered_img = cv2.filter2D(pyramid[-1], -1, gaussian_filter)
        # 1-1. filtering 이미지 저장
        filtered_imgs.append(np.round(filtered_img).astype(np.uint8))

        # 2. subsample(즉 downsampling 이미지 크기 줄이기)
        downsampled_img = filtered_img[::ratio, ::ratio]
        downsample_imgs.append(np.round(downsampled_img).astype(np.uint8))
        pyramid.append(downsampled_img)
        previous_imgs.append(np.round(downsampled_img).astype(np.uint8))

    return previous_imgs, filtered_imgs, downsample_imgs

def main():

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # pyramid level 3
    previous_imgs, filtered_imgs, downsampled_imgs = my_gaussian_pyramid(src,
                                                                         ratio=2,
                                                                         pyramid_lvl=3,
                                                                         filter_size=3,
                                                                         sigma=1)

    cv2.imshow('original', src)

    for level in range(len(downsampled_imgs)):
        cv2.imshow('gaussian down level {}'.format(level), downsampled_imgs[level])
        cv2.imwrite('gaussian_down_level_{}.png'.format(level), downsampled_imgs[level])

    cv2.waitKey()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()