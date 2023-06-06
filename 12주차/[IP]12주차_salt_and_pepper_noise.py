import numpy as np
import cv2
import time

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


def my_median_filtering(src, msize):

    """
    :param src: noise 이미
    :param msize: mask 크기
    :return: median filtering된 결과 이미지
    """

    h,w = src.shape
    dst = np.zeros((h,w))

    for row in range(h):
        for col in range(w):

            # (row,col)를 중심으로 mask를 씌웠을때 index를 초과하는 영역이 생겨남
            # 이 index를 초과하는 범위에 대해서는 median filter 적용시 해당 사항이 없도록 하기 위해서
            # 다음과 같은 row, col를 조정
            r_start = np.clip(row - (msize // 2), 0, h)
            r_end = np.clip(row + (msize // 2) , 0, h)

            c_start = np.clip(col-msize // 2, 0, h)
            c_end = np.clip(col+msize // 2, 0 , h)
            mask = src[r_start:r_end, c_start:c_end]

            dst[row,col] = np.median(mask)

    return dst.astype(np.uint8)

def add_Snp_noise(src,prob):

    """
    :param src: noise를 적용 할 이미지
    :param prob: noise를 적용 할 확률 값
    :return: noise가 추가된 이미지
    """

    h,w = src.shape

    # np.random.rand = 0 ~ 1 사이의 값이 나옴

    noise_prob = np.random.rand(h,w)
    dst = np.zeros((h,w),dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            if noise_prob[row,col] < prob:
                # pepper noise
                dst[row,col] = 0
            elif noise_prob[row,col] > 1 - prob:
                # salt noise
                dst[row,col] = 255
            else:
                dst[row,col] = src[row,col]

    return dst



def main():

    np.random.seed(seed=100)
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    # 원본 이미지에 노이즈를 추가
    snp_noise = add_Snp_noise(src, prob=0.05)

    average_start = time.time()

    # nxn average mask
    # 평균 필터
    mask_size = 5
    mask = np.ones((mask_size,mask_size)) / (mask_size ** 2)

    # noise가 추가된 이미지에 평균 필터 적용
    dst_aver = my_filtering(snp_noise,mask)
    dst_aver = dst_aver.astype(np.uint8)
    print('average filtering time : ',time.time() - average_start)


    # 노이즈가 추가된 이미지에 median filter 적용
    # 결과를 보면 median filter를 적용한 것이 결과가 좋다.
    median_start = time.time()
    dst_median = my_median_filtering(snp_noise, mask_size)
    print('median filtering time : ',time.time() - median_start)

    cv2.imshow('original',src)
    cv2.imshow('Salt and Pepper noise',snp_noise)
    cv2.imshow('noise removal(average fileter)',dst_aver)
    cv2.imshow('noise removal(median filter)',dst_median)
    cv2.waitKey()
    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()

