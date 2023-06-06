import numpy as np
import cv2

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

    dst = src.copy()
    dst *= 255
    dst = np.clip(dst,0,255)

    return dst.astype(np.uint8)


def add_gaus_noise(src,mean=0,sigma=0.1):


    dst = src / 255
    (h,w) = dst.shape
    noise = np.random.normal(mean,sigma,size=(h,w))
    print('noise : \n{}'.format(noise))
    dst += noise

    return my_normalize(dst)


def main():

    np.random.seed(seed=100)
    src = cv2.imread('Lena.png',cv2.IMREAD_GRAYSCALE)

    # I_g(x,y) = I(x,y) + N(x,y)
    dst_noise = add_gaus_noise(src,mean=0,sigma=0.1)

    (h,w) = src.shape
    num = 100

    imgs = np.zeros((num,h,w)) # 100 x 512 x 512

    for i in range(num):
        imgs[i] = add_gaus_noise(src,mean=0,sigma=0.1)

    dst = np.mean(imgs, axis=0).astype(np.uint8)

    cv2.imshow('original',src)
    cv2.imshow('add gaus noise',dst_noise)
    cv2.imshow('noise removal',dst)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return
if __name__ == '__main__':
    main()

