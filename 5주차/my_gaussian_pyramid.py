import numpy as np
import matplotlib.pyplot as plt
import cv2


def my_naive_downsampling(src, ratio):
    return src[::ratio, ::ratio]

def my_navie_pyramid(src, ratio, pyramid_lvl):

    pyramid = [src]
    downsampled_imgs = []

    for i in range(pyramid_lvl):
        downsampled_img = my_naive_downsampling(pyramid[-1], ratio)
        downsampled_imgs.append(downsampled_img)
        pyramid.append(downsampled_img)
    return pyramid, downsampled_imgs

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
    blur_src = cv2.filter2D(src.astpye(np.float32), -1, mask)

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

def cal_pyramid_imgs(previous, filtered_img, downsample_img):
    (h, w) = previous.shape
    (d_h, d_w) = downsample_img.shape

    whole_img = np.zeros((h, 3*w), dtype=np.uint8)

    whole_img[:, :w] = previous
    whole_img[:, w: 2 * w] = filtered_img
    whole_img[:d_h, 2 * w: ((2 * w) + d_w)] = downsample_img

    return whole_img

def plot_imgs(previous, filtered_img, downsample_img, level):

    (h, w) = previous.shape
    (d_h, d_w) = downsample_img.shape

    whole_img = np.zeros((h, 3 * w), dtype=np.uint8)

    whole_img[:, :w] = previous
    whole_img[:, w: 2 * w] = filtered_img
    whole_img[:d_h, 2 * w: ((2 * w) + d_w)] = downsample_img

    cv2.imshow('gaussian level {} image'.format(level), whole_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plot_navie_imgs(previous, downsample_img, level):

    (h, w) = previous.shape
    (d_h, d_w) = downsample_img.shape

    whole_img = np.zeros((h, 2 * w), dtype=np.uint8)

    whole_img[:, :w] = previous
    whole_img[:d_h, w: (w + d_w)] = downsample_img

    cv2.imshow('gaussian level {} image'.format(level), whole_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def plot_pyramid_imgs(previous_imgs, filter_imgs, downsample_imgs):

    level = len(downsample_imgs)

    pyramid_imgs = []
    pyramid_shapes = []
    whole_h = 0

    for i in range(level):

        pyramid = cal_pyramid_imgs(previous_imgs[i], filter_imgs[i],
                                   downsample_imgs[i])

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

    cv2.imshow('gaussian whole pyramid', whole_pyramid)
    cv2.waitKey()
    cv2.destroyAllWindows()


def cal_navie_pyramid_imgs(previous, downsample_img):
    (h, w) = previous.shape
    (d_h, d_w) = downsample_img.shape

    whole_img = np.zeros((h, 2*w), dtype=np.uint8)

    whole_img[:, :w] = previous
    whole_img[:d_h,  w: (w + d_w)] = downsample_img

    return whole_img

def plot_navie_pyramid_imgs(previous_imgs, downsample_imgs):

    level = len(downsample_imgs)

    pyramid_imgs = []
    pyramid_shapes = []
    whole_h = 0

    for i in range(level):

        pyramid = cal_navie_pyramid_imgs(previous_imgs[i],
                                   downsample_imgs[i])

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

    cv2.imshow('navie whole pyramid', whole_pyramid)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # 편의상 원본 사이즈가 아닌 256 x 256 resize해서 사용
    src = cv2.resize(src, dsize=(256, 256), interpolation=cv2.INTER_NEAREST)

    # pyramid level 3
    previous_imgs, filtered_imgs, downsampled_imgs = my_gaussian_pyramid(src,
                                                                         ratio=2,
                                                                         pyramid_lvl=3,
                                                                         filter_size=3,
                                                                         sigma=1)

    # 전체 gaussian pyramid 시각화
    plot_pyramid_imgs(previous_imgs, filtered_imgs, downsampled_imgs)

    lvl = 0
    for previous_img, filter_img, down_img in zip(previous_imgs, filtered_imgs, downsampled_imgs):
        plot_imgs(previous_img, filter_img, down_img, lvl)
        lvl += 1

    pyramid, downsampled_imgs = my_navie_pyramid(src, ratio=2, pyramid_lvl=3)

    # 전체 navie pyramid 시각화
    plot_navie_pyramid_imgs(pyramid, downsampled_imgs)

    lvl = 0
    for previous_img, down_img in zip(pyramid, downsampled_imgs):
        plot_navie_imgs(previous_img, down_img, lvl)
        lvl += 1

    cv2.waitKey()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()