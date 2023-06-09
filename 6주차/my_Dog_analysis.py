import cv2
import numpy as np
import matplotlib.pyplot as plt

def filtering(src, kernel):
    filtering_image = cv2.filter2D(src, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return filtering_image

def generate_sobel_filter_2D():
    sobel_x = np.dot(np.array([[1], [2], [1]]), np.array(([[-1, 0, 1]])))
    sobel_y = np.dot(np.array([[-1], [0], [1]]), np.array(([[1, 2, 1]])))
    return sobel_x, sobel_y


def my_get_Gaussian_filter(fshape, sigma=1):

    (f_h, f_w) = fshape

    ############################################################################
    # TODO 2D Gaussian filter 구현
    # TODO np.mrid를 사용하면 y, x 모두 구할 수 있음
    # TODO 이 함수에서는 정규화를 사용하지 않음.
    # TODO hint
    #     y, x = np.mgrid[-1:2, -1:2]
    #     y => [[-1,-1,-1],
    #           [ 0, 0, 0],
    #           [ 1, 1, 1]]
    #     x => [[-1, 0, 1],
    #           [-1, 0, 1],
    #           [-1, 0, 1]]
    ############################################################################

    y, x = np.mgrid[-(f_h//2):f_h//2+1, -(f_w//2):f_w//2+1]
    gaussian_filter = (np.exp(-(x*x+y*y)/(2*sigma**2))/(sigma**2))
    return gaussian_filter

def my_padding(src, pad_shape, pad_type='zero'):
    # default - zero padding으로 셋팅
    (h,w) = src.shape
    p_h, p_w = pad_shape
    pad_img = np.zeros((h + p_h * 2, w + p_w * 2), dtype=np.float32)
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    return pad_img


def my_filtering(image, filter):
    # image : (height, width), filter : (f_h, f_w)
    # (h_pad, w_pad) = padding size
    h_pad = filter.shape[0] // 2
    w_pad = filter.shape[1] // 2

    filtered_image = np.zeros((min(image.shape[0], image.shape[1]), min(image.shape[0], image.shape[1])))

    for i in range(min(image.shape[0], image.shape[1])):
        for j in range(min(image.shape[0], image.shape[1])):

            filtered_image[i, j] = np.sum(image[i:i + filter.shape[0], j:j + filter.shape[1]] * filter)

    return filtered_image

def get_DoG_filter_by_filtering(fsize, sigma):

    """
    :param fsize: dog filter 크기
    :param sigma: gaussian filter 생성시 사용하는 sigma

    :return: dog_y_filer, dog_x_filter
    """

    ############################################################################
    # TODO 2 filtering 이용한 DoG 필터 마스크 구현
    # TODO 2 절차
    # TODO  2.1 y, x 각 방향에 대한 크기 3을 갖는 미분 vector(1차원) 생성
    # TODO      (예를들어, derivate_x = [[-1, 0, 1]])
    # TODO  2.2 gaussian filter 생성 (임의의 크기 고려) ( my_get_Gaussian_filter 함수 사용)
    # TODO  2.3 2.1에서 구한 1차원 미분 filter를 사용하여 2.2에서 구한 gaussian_filter를 filtering
    # TODO  2.4 y,x 각각의 방향으로 미분 filtering한 결과 값을 반환 -> dog_y_filer, dog_x_filter
    # TODO NOTE filtering시 내장 함수 사용 금지 (cv2.filter2D)
    ############################################################################
    derivated_y = np.array([[-1],
                            [0],
                            [1]])
    derivated_x = derivated_y.T

    gaussian_filter_y = my_get_Gaussian_filter((fsize+2, fsize), sigma)
    gaussian_filter_x = my_get_Gaussian_filter((fsize, fsize+2), sigma)
    # dog_y_filter = filtering(gaussian_filter, derivated_y)
    # dog_x_filter = filtering(gaussian_filter, derivated_x)


    dog_y_filter = my_filtering(gaussian_filter_y, derivated_y) / 2
    dog_x_filter = my_filtering(gaussian_filter_x, derivated_x) / 2

    # dog_y_filter = dog_y_filter * sigma**2
    # dog_x_filter = dog_x_filter * sigma**2

    print("==================여기서 부터는 Filtering===============")
    print(dog_y_filter)
    print(dog_x_filter)
    return dog_y_filter, dog_x_filter

def get_DoG_filter_by_expression(fsize, sigma):
    """
    
    :param fsize: dog filter 크
    :param sigma: sigma 값
    :return: DoG_y, DoG_x
    """

    ############################################################################
    # TODO 1 수식을 이용한 DoG 필터 마스크 구현
    # TODO 1 np.mrid를 사용하면 y, x 모두 구할 수 있음
    # TODO hint
    #     y, x = np.mgrid[-1:2, -1:2]
    #     y => [[-1,-1,-1],
    #           [ 0, 0, 0],
    #           [ 1, 1, 1]]
    #     x => [[-1, 0, 1],
    #           [-1, 0, 1],
    #           [-1, 0, 1]]
    # TODO 수식은 이론 및 실습 ppt를 참고하여 구현.
    ############################################################################

    y, x = np.mgrid[-(fsize//2):(fsize//2)+1, -(fsize//2):(fsize//2)+1]

    gaussian_filter = my_get_Gaussian_filter((fsize, fsize), sigma)


    DoG_y = (-y)*gaussian_filter
    DoG_x = (-x)*gaussian_filter


    print("==================여기서 부터는 Expression===============")
    print(DoG_y)
    print(DoG_x)
    # DoG_y = (-y / (2*np.pi*sigma**4)) * np.exp(-(x**2+y**2) / (2*sigma**2))
    # DoG_x = (-x / (2*np.pi*sigma**4)) * np.exp(-(x**2+y**2) / (2*sigma**2))
    # DoG_y = (-y / sigma**2)*my_get_Gaussian_filter((fsize, fsize), sigma)
    # DoG_x = (-x / sigma**2)*my_get_Gaussian_filter((fsize, fsize), sigma)
    return DoG_y, DoG_x

def calculate_magnitude(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude

def make_noise(std, gray):

    height, width = gray.shape
    img_noise = np.zeros((height, width), dtype=float)
    for i in range(height):
        for a in range(width):
            make_noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * make_noise
            img_noise[i][a] = gray[i][a] + set_noise
    return img_noise

if __name__ == "__main__":

    image = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    noise_image = make_noise(10, image)
    cv2.imshow('noise_image', noise_image / 255)

    sobel_x_filter, sobel_y_filter = generate_sobel_filter_2D()
    sobel_x_image = filtering(noise_image, sobel_x_filter)
    sobel_y_image = filtering(noise_image, sobel_y_filter)

    cv2.imshow('Sobel_magnitude', calculate_magnitude(sobel_x_image / 255., sobel_y_image / 255.))
    cv2.waitKey()
    cv2.destroyAllWindows()


    ############################################################################
    # TODO 1 수식으로 임의의 kernel 크기를 갖는 DoG 필터 마스크 구현
    ############################################################################

    dog_1_y, dog_1_x = get_DoG_filter_by_expression(5, 1)
    dog_y_image = cv2.filter2D(image, -1, dog_1_y, borderType=cv2.BORDER_CONSTANT)
    dog_x_image = cv2.filter2D(image, -1, dog_1_x, borderType=cv2.BORDER_CONSTANT)


    ############################################################################
    # TODO 2 filtering으로 임의의 kernel 크기를 갖는 DoG 필터 마스크 구현
    ############################################################################

    dog_2_y, dog_2_x = get_DoG_filter_by_filtering(5, 1)
    dog_y_image2 = cv2.filter2D(image, -1, dog_2_y, borderType=cv2.BORDER_CONSTANT)
    dog_x_image2 = cv2.filter2D(image, -1, dog_2_x, borderType=cv2.BORDER_CONSTANT)

    # 수식으로 만든 것과 filtering으로 만든 Dog filter 사이의 Magnitude 비교
    cv2.imshow('Dog Magnitude by expression', calculate_magnitude(dog_x_image / 255., dog_y_image / 255.))
    cv2.imshow('Dog Magnitude by filtering', calculate_magnitude(dog_x_image2/255., dog_y_image2/255.))
    cv2.waitKey()

