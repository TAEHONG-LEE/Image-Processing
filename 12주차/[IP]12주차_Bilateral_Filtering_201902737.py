import numpy as np
import cv2


def convert_uint8(src):
    return np.round((((src - src.min()) / (src.max() - src.min())) * 255)).astype(np.uint8)

def my_padding(src, pad_shape, pad_type='zero'):

    # zero padding인 경우
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w), dtype=np.float32)
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')
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

    dst = np.zeros((h, w), dtype=np.float32)
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:m_h + row, col:m_w + col] * mask)
    return dst

def my_get_Gaussian2D_mask(msize, sigma=1):

    y, x = np.mgrid[-(msize // 2) : (msize // 2) + 1, -(msize // 2) : (msize // 2) + 1]

    # 2차 gaussian mask 생성
    gaus2D = 1 / ( 2 * np.pi * sigma**2) * np.exp(-((x**2 + y**2) / (2 * sigma ** 2)))
    # mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    return gaus2D

def get_DoG_filter_by_expression(fsize, sigma):

    y, x = np.mgrid[-(fsize // 2): (fsize // 2) + 1, -(fsize // 2): (fsize // 2) + 1]

    DoG_x = (-x / 2 * np.pi * sigma ** 4) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))
    DoG_y = (-y / 2 * np.pi * sigma ** 4) * np.exp(-((x ** 2 + y ** 2) / (2 * sigma ** 2)))

    return DoG_y, DoG_x

def my_normalize(src):
    dst = src.copy()
    dst *= 255
    dst = np.clip(dst, 0, 255)
    return dst.astype(np.uint8)

def add_gaus_noise(src, mean=0, sigma=0.1):

    #src : 0 ~ 255, dst : 0 ~ 1
    dst = src/255
    h, w = dst.shape
    noise = np.random.normal(mean, sigma, size=(h, w))
    dst += noise
    return my_normalize(dst)

def my_bilateral(src, msize, sigma_s, sigma_r,sigma_d, sigma_dog, pos_x, pos_y, pad_type='zero', is_dog=False):
    """

    :param src: gaussian noise가 있는 이미
    :param msize:지 bilater filter의 크기
    :param sigma_s: 기준 좌표와 기준 좌표에 마스크 영역을 잡았을 때의 좌표들 간에 거리를 구할 때 사용되는 시그마
                    -> 실습 PPT 35 Page의 수식에서 첫번째 항( 각 x, y 모두 같게 설정)
    :param sigma_r: 기준 좌표에서의 픽셀 값과 기준 좌표에 마스크 영역을 잡았을 때의 좌표 값들 간에 차이를 구할 때 사용되는 시그마
                    -> 실습 PPT 35 Page의 수식에서 두번째 항
    :param sigma_d: Dog를 이용한 픽셀 값들의 차이를 구할 때 사용 -> > 실습 PPT 35 Page의 수식에서 세번째 항
    :param sigma_dog: Dog filter를 생성할 때 사용하는 sigma(신경 안써도 됨)
    :param pos_x: 관심 패치의 x좌표
    :param pos_y: 관심 패치의 y좌표
    :param pad_type: padding 유형
    :param is_dog: bilateral filtering을 할 때 Dog항의 사용 유무를 결정할 값
                    Default는 False -> 이것을 if문으로 활용하여 사용한 경우와 아닌 경우 나눌 것
    :return:
    """
    ############################################################
    # TODO
    # TODO my_bilateral 완성
    ############################################################

    (h, w) = src.shape
    # filter size만큼의 y,x 좌표 생성

    y, x = np.mgrid[-(msize // 2): (msize // 2) + 1, -(msize // 2): (msize // 2) + 1]

    # filte size만큼 padding 이미지 생성
    img_pad = my_padding(src, (msize // 2, msize // 2), 'zero')
    # padding 폭
    (p_h, p_w) = (msize // 2, msize // 2)

    dog_1_y, dog_1_x = get_DoG_filter_by_expression(5, sigma_dog)
    dog_y_image = cv2.filter2D(src.astype(np.float32), -1, dog_1_y, borderType=cv2.BORDER_CONSTANT)
    dog_x_image = cv2.filter2D(src.astype(np.float32), -1, dog_1_x, borderType=cv2.BORDER_CONSTANT)

    dog_img = np.sqrt(dog_x_image ** 2 + dog_y_image ** 2)
    dog_img = dog_img / dog_img.max()

    dog_pad = my_padding(dog_img, pad_shape=(p_h, p_w))
    # dst
    dst = np.zeros((h, w), dtype=np.float32)

    mask_spatial = np.exp(-((x**2)+(y**2))/(2*sigma_s**2))

    print(x)
    print(p_h)

    print("pos_x입니다", pos_x)
    for i in range(h):
        print('\r%d / %d ...' %(i, h), end="")
        for j in range(w):
            mask = mask_spatial*np.exp(-((src[i,j] - img_pad[i+y+p_h,j+x+p_w])**2 / (2*sigma_r**2)))

            if is_dog:
                mask = mask*np.exp(-((dog_img[i, j] - dog_pad[i+y+p_h, j+x+p_w])**2 / (2*sigma_d**2)))

            mask = mask / np.sum(mask)

            if i == pos_x and j == pos_y:
                import matplotlib.pyplot as plt
                mask_visual = mask
                mask_visual = mask_visual - mask_visual.min()
                mask_visual = (mask_visual / mask_visual.max() * 255).astype(np.uint8)
                cv2.imshow('mask', mask_visual)
                img = img_pad[i:i + msize - 1, j:j + msize - 1]
                img = my_normalize(img)
                plt.imshow(img, cmap='gray')
                plt.show()
                plt.imshow(mask_visual, cmap='gray')
                plt.show()
            # 한 픽셀에 대한 mask filtering 결과를 반영한다.
            dst[i, j] = np.sum(img_pad[i: i + msize, j: j + msize] * mask)

    return dst

if __name__ == '__main__':

    src = cv2.imread('baby.png', cv2.IMREAD_GRAYSCALE)

    np.random.seed(seed=100)

    # 관심 패치의 좌표
    pos_y = 213
    pos_x = 310

    src_noise = add_gaus_noise(src, mean=0, sigma=0.1)
    src_noise = src_noise / 255
    src_noise_uint8 = convert_uint8(src_noise)

    ######################################################
    # TODO
    # TODO my_bilateral, gaussian mask 채우기
    # TODO filter size, sigma, sigma_s, sigma_r 값 채우기
    # TODO DoG를 활용하는 부분도 채우기
    ######################################################

    msize = 7
    ratio = 0.1  # 조정 가능한 비율 값

    dst = my_bilateral(src_noise, 15, sigma_s=5, sigma_r=0.2, sigma_d=0.08, sigma_dog=3, pos_x=pos_x, pos_y=pos_y)
    dst = my_normalize(dst)

    dog_dst = my_bilateral(src_noise, 15, sigma_s=5, sigma_r=0.2, sigma_d=0.08, sigma_dog=3, pos_x=pos_x, pos_y=pos_y,
                           pad_type='zero', is_dog=True)
    dog_dst = my_normalize(dog_dst)

    import matplotlib.pyplot as plt
    gaus2D = my_get_Gaussian2D_mask(15, sigma=5)
    plt.imshow(gaus2D,cmap='gray')
    plt.show()
    dst_gaus2D = my_filtering(src_noise, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    # 보고서 출력용 - 주석 풀어서 확인해 볼 것
    cv2.imwrite('./baby_gaussian_noise.png', src_noise_uint8)
    cv2.imwrite('./baby_gaussian_filtering.png', dst_gaus2D)
    cv2.imwrite('./baby_my_bilateral.png', dst)
    cv2.imwrite('./baby_my_dog_bilateral.png', dog_dst)

    # stretching difference image
    image1 = cv2.imread('baby_my_bilateral.png', cv2.IMREAD_GRAYSCALE) / 255
    image2 = cv2.imread('baby_my_dog_bilateral.png', cv2.IMREAD_GRAYSCALE) / 255
    diff_image = image2 - image1
    diff_image = diff_image / diff_image.max()
    cv2.imshow('image', diff_image)
    cv2.waitKey()
    cv2.destroyAllWindows()

