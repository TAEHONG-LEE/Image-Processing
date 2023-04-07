import numpy as np
import matplotlib.pyplot as plt
import cv2


def my_downsampling(src, ratio):

    """
    함수 인자 정보
    :param src: gray 이미지( H x W )
    :param ratio: downsampling 할 비율

    변수 정보
    downsampled_blur_img: src에 ratio만큼 downsamling을 적용한 결과 이미지

    :return: downsampled_img
    """

    # ratio 만큼 행과 열을 추출
    downsampled_img = src[::ratio, ::ratio]

    return downsampled_img


def my_laplacian_pyramid(src, ratio, pyramid_lvl, filter_size, sigma):

    """
    인자 정보
    :param src: gray 이미지( H x W)
    :param ratio: downsampling 할 비율
    :param pyramid_len:  피라미드의 층 개
    :param filter_size: gaussian filter 크
    :param sigma: gaussian filter sigma

    변수 정보

    residuals: residual 이미지를 저장할 변수
    :return: filter_imgs, residuals_visualized, gaussian_pyramid
    """

    previous_imgs = [src]
    after_imgs = []
    filter_imgs = []
    gaussian_pyramid = [src.astype(np.float32)]
    gaussian_filter = cv2.getGaussianKernel(ksize=filter_size, sigma=sigma)
    gaussian_filter = gaussian_filter @ gaussian_filter.T

    residuals = []
    residuals_visualized = []

    downsample_imgs = []

    for level in range(pyramid_lvl):

        ######################################################
        # Laplacian pyramid 진행 절차 (수정된 실습 PPT 참)
        # 1. Gaussian filtering
        # 2. residual 계산 및 저장
        #   2.1 1의 filtering된 이미지를 downsampling 후 upsampling
        #   2.2 residual = gaussian pyramid(G_{i}) - 2.1에서 구한 이미지(G_{i}')
        # 3. 2.1에서 downsampling한 이미지를 다음 피라미드의 입력으로 설정
        # 4. 1 ~ 3 과정을 반복, gaussian의 마지막 이미지의 경우
        ######################################################

        # 1. Gaussian filtering
        filtered_img = cv2.filter2D(gaussian_pyramid[-1], -1, gaussian_filter)

        # 2. residual 계산 및 저장
        down_gaussian_img = my_downsampling(filtered_img, ratio)
        up_gaussian_img = cv2.resize(down_gaussian_img, dsize=(0, 0),
                                     fx=ratio, fy=ratio, interpolation=cv2.INTER_LINEAR)
        after_imgs.append(np.round(up_gaussian_img).astype(np.uint8))
        residual = gaussian_pyramid[-1] - up_gaussian_img

        # residual visualize : min-max scaling을 통해 이미지를 0 ~ 1로 normalization
        # 그 후에 uint형으로 데이터 타입을 바꾸기 위해 255를 곱함
        residual_visualized = ((residual - residual.min()) / (residual.max() - residual.min())) * 255
        residual_visualized = np.round(residual_visualized).astype(np.uint8)
        residuals.append(residual)
        residuals_visualized.append(residual_visualized)

        # 3. 2.1에서 downsampling한 이미지를 다음 피라미드의 입력으로 설정
        gaussian_pyramid.append(down_gaussian_img)
        downsample_imgs.append(np.round(down_gaussian_img).astype(np.uint8))

    return gaussian_pyramid[:-1], after_imgs, residuals_visualized, downsample_imgs


def plot_imgs(previous, after_img, residual_img, downsample_img, level):

    (h, w) = previous.shape
    (d_h, d_w) = downsample_img.shape

    whole_img = np.zeros((h, 4*w), dtype=np.uint8)

    whole_img[:, :w] = previous
    whole_img[:, w: 2 * w] = after_img
    whole_img[:, 2 * w: 3 * w] = residual_img
    whole_img[:d_h, 3 * w: ((3 * w) + d_w)] = downsample_img

    cv2.imshow('gaussian level {} image'.format(level), whole_img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def cal_pyramid_imgs(previous, after_img, residual_img, downsample_img):
    (h, w) = previous.shape
    (d_h, d_w) = downsample_img.shape

    whole_img = np.zeros((h, 4*w), dtype=np.uint8)

    whole_img[:, :w] = previous
    whole_img[:, w: 2 * w] = after_img
    whole_img[:, 2 * w: 3 * w] = residual_img
    whole_img[:d_h, 3 * w: ((3 * w) + d_w)] = downsample_img

    return whole_img

def plot_pyramid_imgs(previous_imgs, after_imgs, residuals_visualized, downsample_imgs):

    level = len(downsample_imgs)

    pyramid_imgs = []
    pyramid_shapes = []
    whole_h = 0

    for i in range(level):

        pyramid = cal_pyramid_imgs(previous_imgs[i], after_imgs[i],
                         residuals_visualized[i], downsample_imgs[i])

        pyramid_imgs.append(pyramid)
        pyramid_shapes.append(pyramid.shape)
        whole_h += pyramid.shape[0]

    whole_w = pyramid_imgs[0].shape[1]

    whole_pyramid = np.zeros((whole_h, whole_w), dtype=np.uint8)

    previous_h = 0

    for i in range(level):
        if i == 0:

            whole_pyramid[:pyramid_shapes[i][0], :pyramid_shapes[i][1]] = pyramid_imgs[i]
            previous_h = pyramid_shapes[i][0]
        else:
            cur_h = previous_h + pyramid_shapes[i][0]
            whole_pyramid[previous_h:cur_h, :pyramid_shapes[i][1]] = pyramid_imgs[i]
            previous_h = cur_h

    cv2.imshow('Whole pyramid', whole_pyramid)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # 편의상 원본 사이즈가 아닌 256 x 256 resize해서 사용
    src = cv2.resize(src, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

    # pyramid level 3
    previous_imgs, after_imgs, residuals_visualized, downsample_imgs = my_laplacian_pyramid(src, ratio=2,
                                                                             pyramid_lvl=3, filter_size=3, sigma=1)

    # 전체 파리미드 시각화
    plot_pyramid_imgs(previous_imgs, after_imgs, residuals_visualized, downsample_imgs)

    lvl = 0
    for previous_img, filter_img, residual_img, down_img in zip(previous_imgs, after_imgs, residuals_visualized, downsample_imgs):
        plot_imgs(previous_img, filter_img, residual_img, down_img, lvl)
        lvl += 1

    return

if __name__ == '__main__':
    main()