import cv2
import numpy as np

def my_bilinear(src, scale):

    #######################################################################
    # TODO blinear interpolation 완성
    # TODO blinear interpolation 실습 코드 참고
    #######################################################################
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst), dtype=np.float32)

    for row in range(h_dst):
        for col in range(w_dst):


            dst[row, col] = ???

    return dst


def my_nearest_neighbor(src, scale=None, scale_shape=None):

    """
    함수 인자 정보
    src: gray scale 이미지 (H x W)
    scale: 축소 또는 확대할 비율
    scale_shape: 축소 또는 확대할 구체적인 높이와 너비 값
    """

    (h, w) = src.shape

    # scale이 지정된 경우
    if scale is not None:
        h_scale, w_scale = scale
        h_dst = int(h * h_scale + 0.5)
        w_dst = int(w * w_scale + 0.5)

    # scale이 지정 안되고 구체적인 dst 크기가 지정된 경우
    else:
        h_dst, w_dst = scale_shape
        h_scale = h_dst / h
        w_scale = w_dst / w

    dst = np.zeros((h_dst, w_dst), dtype=np.float32)
    for row in range(h_dst):
        for col in range(w_dst)


            #######################################################################
            # TODO nearest interpolation 완성
            # TODO nearest interpolation 실습 코드 참고
            #######################################################################

            dst[row, col] = ???

    return dst


def my_upsampling_laplacian(src, ratio, residuals=None, upsampling_type='bilinear',
                            pyramid_level=3):

    """

    :param src: gaussian pyramid의 마지막 feature
    :param ratio: upsampling할 비율
    :param residuals: residual 값
    :param upsampling_type: upsampling할 유형
    :param pyramid_level: 피라미드 층 개수
    :return: x
    """

    x = src
    if residuals is not None:
        residuals = residuals[::-1]

    for i in range(pyramid_level):

        if upsampling_type == 'nearest':
            x = my_nearest_neighbor(x, scale=(ratio, ratio))

        else:
            # bilinear interpolation
            x = my_bilinear(x, scale=ratio)

        if residuals is not None:
            #######################################################################
            # TODO residual 정보 반영
            # TODO residuals 변수 사용
            #######################################################################
            x = ???

    x = np.clip(x, 0, 255)
    x = np.round(x).astype(np.uint8)
    return x

def my_downsampling_pyramid(src, ratio, pyramid_lvl, filter_size, sigma, interpolation_type='nearest'):

    """
    인자 정보
    :param src: gray 이미지( H x W)
    :param ratio: downsampling 할 비율
    :param pyramid_len:  피라미드의 층 개수
    :param filter_size: gaussian filter 크기
    :param sigma: gaussian filter sigma
    :param interpolation_type: interpolation 타입( i.e nearest, bilinear ...)

    변수 정보
    gaussian_pyramid: gaussian pyramid
    laplacian_residuals: residual 이미지를 저장할 변수
    :return: residuals_visualized, gaussian_pyramid
    """

    gaussian_pyramid = [src.astype(np.float32)]
    if filter_size is not None:
        gaussian_filter = cv2.getGaussianKernel(ksize=filter_size, sigma=sigma)
        gaussian_filter = gaussian_filter @ gaussian_filter.T

    laplacian_residuals = []

    if filter_size is not None:
        for level in range(pyramid_lvl):

            ###########################################################################
            # TODO Laplacian pyramid 진행 절차
            # TODO 1. Gaussian filtering cv2.filter2D 내장함수 사용할
            # TODO 2. residual 계산 및 저장
            # TODO  2.1 1의 filtering된 이미지를 downsampling 후 upsampling G_{i}'
            # TODO  2.2 residual = gaussian pyramid(G_{i}) - 2.1에서 구한 이미지(G_{i}')
            # TODO 3. 2.1에서 downsampling한 이미지를 다음 피라미드의 입력으로 설정
            # TODO 4. 1 ~ 3 과정을 반복
            # TODO interpolation의 경우 nearest 와 bilinear 중 경우에 맞는 한 가지를 택해서 사용
            # TODO interpolation_type 변수를 활용
            ###########################################################################

            # 1. filtering
            filtered_img = cv2.filter2D(gaussian_pyramid[-1], -1, gaussian_filter)
            # 2. residual 계산 및 저장

            residual = ???
            laplacian_residuals.append(laplacian_residual)


            # 3. 2.1에서 downsampling한 이미지를 다음 피라미드의 입력으로 설정
            gaussian_pyramid.append(???)

        return laplacian_residuals, gaussian_pyramid[-1]


    if filter_size is None:
        for level in range(pyramid_lvl):

            ###########################################################################
            # TODO naive pyramid
            # TODO subsample(즉 downsampling 이미지 크기 줄이기)
            # TODO nearest 와 bilinear 중 경우에 맞는 한 가지를 택해서 사용
            # TODO interpolation_type 변수를 활용
            ###########################################################################

            ???

            gaussian_pyramid.append(???)

        return gaussian_pyramid[-1]

def main():

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    #######################################################################
    # TODO Upsampling 과정시 Residual 사용 유무 결과 비교
    # TODO down-sampling시 두 과정 모두 기본적으로 Residual + Gaussian filter 사용
    # TODO
    #######################################################################
    ratio = 2
    pyramid_level = 3

    #######################################################################
    # TODO 1.1 Nearest interpolation + Residual 적용
    #######################################################################
    laplacian_residuals, gaussian_pyramid = my_downsampling_pyramid(src, ratio=ratio, pyramid_lvl=pyramid_level,
                                                         filter_size=3, sigma=1, interpolation_type='nearest')

    near_output = my_upsampling_laplacian(gaussian_pyramid, ratio,
                                          residuals=laplacian_residuals,
                                          upsampling_type='nearest',
                                          pyramid_level=pyramid_level)


    # 원본 이미지와 회복된 이미지 비교
    # True : 원본이미지와 동일
    print("원본과 nearest 보간법을 이용해서 복원한 경우 비교 결과: {}".format(np.array_equal(src, near_output)))


    #######################################################################
    # TODO 1.2 Nearest interpolation + Residual 적용 x
    #######################################################################
    _, gaussian_pyramid = my_downsampling_pyramid(src, ratio=ratio,
                                                         pyramid_lvl=pyramid_level,
                                                         filter_size=3, sigma=1, interpolation_type='nearest')

    no_res_near_output = my_upsampling_laplacian(gaussian_pyramid,
                                                 ratio,
                                                 residuals=None,
                                                 upsampling_type='nearest',
                                                 pyramid_level=pyramid_level)

    #######################################################################
    # TODO 1.3 Naive up&down sampling Gaussian 적용 x and Residual 적용 x
    #######################################################################
    downsample_imgs = my_downsampling_pyramid(src, ratio=ratio,pyramid_lvl=pyramid_level,
                                             filter_size=None, sigma=None, interpolation_type='nearest')

    naive_output = my_upsampling_laplacian(downsample_imgs, ratio, residuals=None, upsampling_type='nearest',
                                            pyramid_level=pyramid_level)


    #######################################################################
    # TODO 2 Bilinear Interpolation 적용
    # TODO 2.1 Bilinear Interpolation 적용 + Residual 적용
    #######################################################################
    laplacian_residuals, gaussian_pyramid = my_downsampling_pyramid(src, ratio=ratio, pyramid_lvl=pyramid_level,
                                                         filter_size=3, sigma=1,
                                                         interpolation_type='bilinear')


    bilinear_output = my_upsampling_laplacian(gaussian_pyramid, ratio,
                                              residuals=laplacian_residuals,
                                              upsampling_type='bilinear',
                                              pyramid_level=pyramid_level)
    # 원본 이미지와 회복된 이미지 비교
    # True : 원본이미지와 동일
    print("원본과 bilinear 보간을 이용해서 복원한 경우 비교 결과: {}".format(np.array_equal(src, bilinear_output)))

    #######################################################################
    # TODO 2 Bilinear Interpolation 적용
    # TODO 2.2 Bilinear Interpolation 적용 + Residual x
    #######################################################################
    _, gaussian_pyramid = my_downsampling_pyramid(src, ratio=ratio, pyramid_lvl=pyramid_level,
                                                         filter_size=3, sigma=1,
                                                         interpolation_type='bilinear')

    no_res_bilinear_output = my_upsampling_laplacian(gaussian_pyramid, ratio, residuals=None,
                                                     upsampling_type='bilinear',
                                                     pyramid_level=pyramid_level)

    #######################################################################
    # TODO 2 Bilinear Interpolation 적용
    # TODO 2.3 Bilinear Interpolation 적용 + Gaussian 적용 x and Residual 적용 x
    #######################################################################
    downsample_imgs = my_downsampling_pyramid(src, ratio=ratio,
                                                         pyramid_lvl=pyramid_level,
                                                         filter_size=None, sigma=None,
                                                         interpolation_type='bilinear')

    naive_bilinear_output = my_upsampling_laplacian(downsample_imgs,
                                                 ratio,
                                                 residuals=None,
                                                 upsampling_type='bilinear',
                                                    pyramid_level=pyramid_level)

    cv2.imshow('original', src)
    cv2.imshow('nearest', near_output)
    cv2.imshow('no residual nearest', no_res_near_output)
    cv2.imshow('naive nearest output', naive_output)

    cv2.imshow('bilinear + residual', bilinear_output)
    cv2.imshow('no_res_bilinear_output', no_res_bilinear_output)
    cv2.imshow('naive_bilinear_output', naive_bilinear_output)
    cv2.waitKey()
    cv2.destroyAllWindows()


    # 보고서 출력용 이미지
    cv2.imwrite('nearest+residual.png',near_output)
    cv2.imwrite('nearest+no+residual.png', no_res_near_output)
    cv2.imwrite('nearest_naive.png', naive_output)
    cv2.imwrite('bilinear+residual.png',bilinear_output)
    cv2.imwrite('bilinear+no+residual.png',no_res_bilinear_output)
    cv2.imwrite('bilinear_navie.png', naive_bilinear_output)

    return


if __name__ == '__main__':
    main()