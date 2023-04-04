import cv2
import numpy as np

def my_padding(src, pad_shape, pad_type = 'zero'):

    # default - zero padding으로 셋팅
    (h,w) = src.shape
    p_h, p_w = pad_shape
    pad_img = np.zeros((h + p_h * 2, w + p_w * 2), dtype=np.uint8)
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

if __name__ == '__main__':
    src = cv2.imread('Lena.png',cv2.IMREAD_GRAYSCALE)

    my_zero_pad_img = my_padding(src,(20,20))
    my_repetition_pad_img = my_padding(src, (20,20),pad_type='repetition')

    cv2.imshow('original',src)
    cv2.imshow('my zero pad img', my_zero_pad_img)
    cv2.imshow('my repetition pad img', my_repetition_pad_img)

    cv2.waitKey()
    cv2.destroyAllWindows()

