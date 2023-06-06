import numpy as np
import cv2
import matplotlib.pyplot as plt



def YIQ2BGR(src):
    """

    :param src: YIQ 이미지
    :return: BGR 이미
    """
    src = src.astype(np.float32)

    Y = src[:, :, 0]
    I = src[:, :, 1]
    Q = src[:, :, 2]

    R = 1 * Y + 0.956 * I + 0.621 * Q
    G = 1 * Y + (-0.272) * I + (-0.647) * Q
    B = 1 * Y + (-1.106) * I + (1.703) * Q

    BGR = np.clip(np.round(np.dstack((B, G, R))), 0, 255).astype(np.uint8)

    return BGR


def BGR2YIQ(src):
    """

    :param src: BGR 이미지
    :return:YIQ 이미지지
    """
    src = src.astype(np.float32)

    B = src[:, :, 0]
    G = src[:, :, 1]
    R = src[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    I = 0.596 * R + (-0.274) * G + (-0.322) * B
    Q = 0.211 * R + (-0.523) * G + (0.312) * B

    YIQ = np.clip(np.round(np.dstack((Y, I, Q))), 0, 255).astype(np.uint8)

    return YIQ


def main():

    # BGR uint8 image
    src = cv2.imread('Lena.png')
    yiq = BGR2YIQ(src)
    bgr = YIQ2BGR(yiq)

    cv2.imshow('original', src)
    cv2.imshow('bgr -> yiq', yiq)
    cv2.imshow('yiq -> bgr', bgr)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()