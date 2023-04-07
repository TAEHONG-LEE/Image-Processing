import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def my_padding(src,mask,pad_type = 'zero'):

    # default - zero padding으로 셋팅
    (h,w) = src.shape
    (f_h, f_w) = mask.shape
    p_h = f_h // 2
    p_w = f_w // 2
    pad_img = np.zeros((h + p_h * 2, w + p_w * 2))
    pad_img[p_h:h + p_h, p_w : w + p_w] = src
    if pad_type == 'repetition':
        print('repetition padding')
        #up
        pad_img[:p_h, p_w:p_w + w] = src[0,:]

        #down
        pad_img[p_h + h:, p_w:p_w + w] = src[h-1,:]

        #left
        pad_img[:,:p_w] = pad_img[:,p_w:p_w + 1]

        #right
        pad_img[:,p_w + w :] = pad_img[:,p_w + w -1 : p_w + w]

    else:
        # else is zero padding
        print('zero padding')
    return pad_img


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (f_h, f_w) = mask.shape
    pad_img = my_padding(src, mask, pad_type)
    dst = np.zeros((h, w))

    #########################################
    # TODO 3. Filtering 2중 for문 구현
    #########################################

    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row+f_h, col:col+f_w]*mask)
    dst = np.round(dst).astype(np.uint8)
    return dst


def my_get_Gaussian_filter(fshape, sigma=1):

    (f_h, f_w) = fshape
    ############################################################################
    # TODO 2 2D Gaussian filter 구현
    # TODO 2 np.mrid를 사용하면 y, x 모두 구할 수 있음
    # TODO hint
    #     y, x = np.mgrid[-1:2, -1:2]
    #     y => [[-1,-1,-1],
    #           [ 0, 0, 0],
    #           [ 1, 1, 1]]
    #     x => [[-1, 0, 1],
    #           [-1, 0, 1],
    #           [-1, 0, 1]]
    ############################################################################

    y, x = np.mgrid[-(f_w//2):f_w//2+1, -(f_w//2):f_w//2+1]

    # 2차 gaussian mask 생성
    gaussian_filter = (np.exp(-(x*x+y*y)/(2*sigma**2)))/(2*np.pi*sigma**2)

    # mask 총합 1 : 평균 밝기의 변화가 없도록 하기 위함
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)
    return gaussian_filter


def my_gaussian_filter(src, fshape, sigma=15, verbose=False):
    (h, w) = src.shape
    if verbose:
        print('Gaussian filtering')

    filter = my_get_Gaussian_filter(fshape, sigma=sigma)

    if verbose:
        print('<Gaussian filter> - shape:', fshape, '-sigma:', sigma)
        print(filter)

    dst = my_filtering(src, filter)
    return dst, filter


if __name__ == '__main__':

    src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)

    ##################################################################################
    # TODO 1 kernel 크기와 sigma 변화에 따른 2D Gaussian filter 분석
    # TODO filter_size sigma 변수를 다음과 같이 6가지 경우로 나누어서 결과 이미지 및 필터 시각화 결과 확인
    # TODO 보고서에는 6가지 경우에 대한 결과들의 분석 내용이 있어야함
    ##################################################################################
    verbose = False

    # Case1
    filter_size = 5
    sigma = 1
    dst_gaussian_2D_1, filter1 = my_gaussian_filter(src, (filter_size, filter_size),
                                                    sigma=sigma, verbose=verbose)

    # 필터 시각화 확인
    filter1 = ((filter1 - filter1.min()) / (filter1.max() - filter1.min())) * 255
    filter1 = np.clip(filter1, 0, 255)
    filter1 = np.round(filter1).astype(np.uint8)

    plt.title('fsize=5, sigma=1')
    plt.imshow(filter1, cmap='gray')
    plt.show()

    dst_gaussian_2D_1 = np.round(dst_gaussian_2D_1).astype(np.uint8)

    # Case2
    filter_size = 5
    sigma = 3
    dst_gaussian_2D_2, filter2 = my_gaussian_filter(src, (filter_size, filter_size),
                                           sigma=sigma, verbose=verbose)
    dst_gaussian_2D_2 = np.round(dst_gaussian_2D_2).astype(np.uint8)

    # 필터 시각화 확인
    filter2 = ((filter2 - filter2.min()) / (filter2.max() - filter2.min())) * 255
    filter2 = np.clip(filter2, 0, 255)
    filter2 = np.round(filter2).astype(np.uint8)

    plt.title('fsize=5, sigma=3')
    plt.imshow(filter2, cmap='gray')
    plt.show()

    # Case3
    filter_size = 5
    sigma = 0.1
    dst_gaussian_2D_3, filter3 = my_gaussian_filter(src, (filter_size, filter_size),
                                           sigma=sigma, verbose=verbose)
    dst_gaussian_2D_3 = np.round(dst_gaussian_2D_3).astype(np.uint8)

    # 필터 시각화 확인
    filter3 = ((filter3 - filter3.min()) / (filter3.max() - filter3.min())) * 255
    filter3 = np.clip(filter3, 0, 255)
    filter3 = np.round(filter3).astype(np.uint8)

    plt.title('fsize=5, sigma=0.1')
    plt.imshow(filter3, cmap='gray')
    plt.show()

    # Case4
    filter_size = 7
    sigma = 3
    dst_gaussian_2D_4, filter4 = my_gaussian_filter(src, (filter_size, filter_size),
                                           sigma=sigma, verbose=verbose)
    dst_gaussian_2D_4 = np.round(dst_gaussian_2D_4).astype(np.uint8)

    # 필터 시각화 확인
    filter4 = ((filter4 - filter4.min()) / (filter4.max() - filter4.min())) * 255
    filter4 = np.clip(filter4, 0, 255)
    filter4 = np.round(filter4).astype(np.uint8)

    plt.title('fsize=7, sigma=3')
    plt.imshow(filter4, cmap='gray')
    plt.show()

    # Case5
    filter_size = 11
    sigma = 3
    dst_gaussian_2D_5, filter5 = my_gaussian_filter(src, (filter_size, filter_size),
                                           sigma=sigma, verbose=verbose)
    dst_gaussian_2D_5 = np.round(dst_gaussian_2D_5).astype(np.uint8)

    # 필터 시각화 확인
    filter5 = ((filter5 - filter5.min()) / (filter5.max() - filter5.min())) * 255
    filter5 = np.clip(filter5, 0, 255)
    filter5 = np.round(filter5).astype(np.uint8)

    plt.title('fsize=11, sigma=3')
    plt.imshow(filter5, cmap='gray')
    plt.show()

    # Case6
    filter_size = 15
    sigma = 3
    dst_gaussian_2D_6, filter6 = my_gaussian_filter(src, (filter_size, filter_size),
                                           sigma=sigma, verbose=verbose)
    dst_gaussian_2D_6 = np.round(dst_gaussian_2D_6).astype(np.uint8)

    # 필터 시각화 확인
    filter6 = ((filter6 - filter6.min()) / (filter6.max() - filter6.min())) * 255
    filter6 = np.clip(filter6, 0, 255)
    filter6 = np.round(filter6).astype(np.uint8)

    plt.title('fsize=15, sigma=3')
    plt.imshow(filter6, cmap='gray')
    plt.show()

    cv2.imshow('original', src.astype(np.uint8))
    cv2.imshow('Case 1 Gaussian 2D image', dst_gaussian_2D_1)
    cv2.imshow('Case 2 Gaussian 2D image', dst_gaussian_2D_2)
    cv2.imshow('Case 3 Gaussian 2D image', dst_gaussian_2D_3)
    cv2.imshow('Case 4 Gaussian 2D image', dst_gaussian_2D_4)
    cv2.imshow('Case 5 Gaussian 2D image', dst_gaussian_2D_5)
    cv2.imshow('Case 6 Gaussian 2D image', dst_gaussian_2D_6)

    cv2.waitKey()
    cv2.destroyAllWindows()

    # 보고서 첨부용 결과 이미지
    cv2.imwrite('Case1.png',dst_gaussian_2D_1)
    cv2.imwrite('Case2.png',dst_gaussian_2D_2)
    cv2.imwrite('Case3.png',dst_gaussian_2D_3)
    cv2.imwrite('Case4.png',dst_gaussian_2D_4)
    cv2.imwrite('Case5.png',dst_gaussian_2D_5)
    cv2.imwrite('Case6.png',dst_gaussian_2D_6)

