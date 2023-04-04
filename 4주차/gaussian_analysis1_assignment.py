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
            # dst = ???
            dst[row, col] = np.sum(pad_img[row:row+f_h, col:col+f_w]*mask)
    dst = np.round(dst).astype(np.uint8)
    return dst

def my_get_Gaussian_filter(fshape, sigma=1): # fshape (filter_size, 1) or (1, filter_size)

    (f_h, f_w) = fshape
    sigma=sigma
    ############################################################################
    # TODO 2. 1D Gaussian으로 2번 Filtering
    # TODO gaussian_filter 식 채우기
    # TODO 참고) np.mgrid 사용하면 쉽게 구현 가능
    # TODO hint
    #     y, x = np.mgrid[-1:2, -1:2]
    #     y => [[-1,-1,-1],
    #           [ 0, 0, 0],
    #           [ 1, 1, 1]]
    #     x => [[-1, 0, 1],
    #           [-1, 0, 1],
    #           [-1, 0, 1]]
    ############################################################################

    if (f_h == 1):
        print(f_w//2)
        y, x = np.mgrid[-(f_w//2):f_w//2+1, -(f_w//2):f_w//2+1]
        print(x)
        gaussian_filter = np.exp((-x*x)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    elif (f_w == 1):
        y, x = np.mgrid[(f_h/2)-f_h:f_h/2, (f_h/2)-f_h:f_h/2]
        gaussian_filter = np.exp((-y*y)/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)


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

    # 사용중인 필터를 확인하고 싶으면 True로 변경, 보기 싫으면 False로 변경
    verbose = True

    ############################################################################
    # TODO 1. sigma 변화에 따른 1D Gaussian filter 시각화
    # TODO gaussian_filter 식 채우기 및 그래프 그리기
    # TODO 교수님 이론 PPT 40 page 참고
    ############################################################################
    x = np.linspace(-5, 5, 1000)
    mean = x.mean()
    sigma1 = 0.5
    gaussian_filter1 = np.exp((-x*x)/(2*sigma1**2))/(np.sqrt(2*np.pi)*sigma1)

    sigma2 = 1
    gaussian_filter2 = np.exp((-x*x)/(2*sigma2**2))/(np.sqrt(2*np.pi)*sigma2)

    sigma3 = 2
    gaussian_filter3 = np.exp((-x*x)/(2*sigma3**2))/(np.sqrt(2*np.pi)*sigma3)

    sigma4 = 3
    gaussian_filter4 = np.exp((-x*x)/(2*sigma4**2))/(np.sqrt(2*np.pi)*sigma4)

    # xxxxxxxxx : 학번 적을 것
    # 안적으면 감점
    plt.title('201902737 Gaussian filter visualization')
    plt.plot(x, gaussian_filter1, label='sigma=0.5')
    plt.plot(x, gaussian_filter2, label='sigma=1')
    plt.plot(x, gaussian_filter3, label='sigma=2')
    plt.plot(x, gaussian_filter4, label='sigma=3')
    plt.legend()
    plt.show()

    ############################################################################
    # TODO 2. 1D Gaussian으로 2번 Filtering
    # TODO gaussian_filter 식 채우기
    ############################################################################
    filter_size = 5
    sigma = 1
    dst_gaussian_1D, _ = my_gaussian_filter(src, (filter_size, 1), sigma=sigma,
                                         verbose=verbose)
    dst_gaussian_1D, _ = my_gaussian_filter(dst_gaussian_1D, (1, filter_size), sigma=sigma,
                                         verbose=verbose)

    dst_gaussian_1D = np.round(dst_gaussian_1D).astype(np.uint8)

    cv2.imshow('original', src.astype(np.uint8))
    cv2.imshow('Gaussian 1D filtering image', dst_gaussian_1D)
    cv2.waitKey()
    cv2.destroyAllWindows()


