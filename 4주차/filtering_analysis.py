import cv2
import numpy as np
import time

def generate_average_filter(ksize):
    """
    인자 정보
    ksize: 커널 크기

    return kernel
    """
    #########################################################
    # TODO                                                  #
    # average filter 구현
    #########################################################

    kernel = np.ones((ksize,ksize)) / ksize**2
    return kernel

def my_padding(src,ksize,pad_type ='zero'):

    # default - zero padding으로 셋팅
    (h,w) = src.shape
    (f_h, f_w) = ksize
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

def my_filtering(image, kernel):

    """
    Arguments info
    image: gray-scale image (H x W)
    kernel: kernel size(tuple) -> k_h(kernel height), k_w(kernel width)

    Variables info
    pad_img: padding된 이미지
    dst: filtering된 결과 이미지

    return dst
    """

    h, w = image.shape
    k_h, k_w = kernel.shape
    pad_image = my_padding(image, (k_h, k_w))
    # filtering 계산을 위해 dst 타입을 float형으로 지정
    dst = np.zeros((h, w), dtype=np.float32)

    #########################################################
    # TODO                                                  #
    # Filtering 구현                                         #
    # dst : filtering 결과 image                             #
    # 유의미한 시간 측정을 위해 4중 for 문으로 구현 할 것
    # 교수님 이론 PPT 4page 수식 참고
    #########################################################
    for m in range(0, h):
        for n in range(0, w):
            for k in range(k_h):
                for l in range(k_w):
                    dst[m,n] += pad_image[m+k,n+l]*kernel[k,l]

    # float32 -> uint8(unsigned int)로 변경
    dst = np.round(dst).astype(np.uint8)
    # print(pad_image[:9,:9])
    # print(kernel)
    # print(np.sum(pad_image[:3,:3]*kernel))
    # print(dst[:3,:3])
    return dst


def measure_filtering_time(image, kernel):

    h, w = kernel.shape
    start = time.perf_counter()
    output = my_filtering(image, kernel)
    print('{} X {} Filter Time : {}'.format(h, w ,(time.perf_counter() - start)))

    return output


def compare_speed_filtering():

    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    my_3x3_average_filter = generate_average_filter(ksize=3)
    my_7x7_average_filter = generate_average_filter(ksize=7)
    my_15x15_average_filter = generate_average_filter(ksize=15)

    output1 = measure_filtering_time(src, my_3x3_average_filter)
    output2 = measure_filtering_time(src, my_7x7_average_filter)
    output3 = measure_filtering_time(src, my_15x15_average_filter)

    # filtering 결과 이미지 확인
    cv2.imshow('original', src)
    cv2.imshow('3x3 filter output', output1)
    cv2.imshow('7x7 filter output', output2)
    cv2.imshow('15x15 filter output', output3)

    cv2.waitKey()
    cv2.destroyAllWindows()


def main():

    compare_speed_filtering()

    return

if __name__ == "__main__":
    main()