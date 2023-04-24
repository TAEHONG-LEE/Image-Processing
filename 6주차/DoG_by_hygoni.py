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
    y = fshape[0] // 2 + 1
    x = fshape[1] // 2 + 1

    y, x = np.mgrid[-y + 1:y, -x + 1:x]
    gaussian_filter = (1 / (sigma * sigma)) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    gaussian_filter /= gaussian_filter.sum()
    return gaussian_filter


def get_DoG_filter_by_filtering(fsize, sigma):
    """
    :param fsize: dog filter 크기
    :param sigma: gaussian filter 생성시 사용하는 sigma

    :return: dog_y_filer, dog_x_filter
    """

    dx = np.array([[-1, 0, 1]])
    dy = ([[-1], [0], [1]])

    # gaussian x,y filter 생성
    gaussian_x_filter = my_get_Gaussian_filter((fsize, fsize), sigma)
    gaussian_y_filter = my_get_Gaussian_filter((fsize, fsize), sigma)

    # gaussian filter 패딩
    f_h = 3
    f_w = 1
    p_h = f_h // 2
    p_w = f_w // 2
    gaussian_y_filter_padded = np.zeros((fsize + p_h * 2, fsize + p_w * 2))
    gaussian_y_filter_padded[p_h:fsize + p_h, p_w:fsize + p_w] = gaussian_y_filter

    # gaussian y filter 미분
    dog_y_filter = np.zeros((fsize, fsize))
    for row in range(fsize):
        for col in range(fsize):
            dog_y_filter[row, col] = np.sum(dy * gaussian_y_filter_padded[row:row + f_h, col:col + f_w])

    # gaussian x filter 패딩
    f_h = 1
    f_w = 3
    p_h = f_h // 2
    p_w = f_w // 2
    gaussian_x_filter_padded = np.zeros((fsize + p_h * 2, fsize + p_w * 2))
    gaussian_x_filter_padded[p_h:fsize + p_h, p_w:p_w + fsize] = gaussian_x_filter

    # gaussian y filter 미분
    dog_x_filter = np.zeros((fsize, fsize))
    for row in range(fsize):
        for col in range(fsize):
            dog_x_filter[row, col] = np.sum(dx * gaussian_x_filter_padded[row:row + f_w, col:col + f_w])

    print(dog_y_filter)
    return dog_y_filter, dog_x_filter


def get_DoG_filter_by_expression(fsize, sigma):
    """

    :param fsize: dog filter 크
    :param sigma: sigma 값
    :return: DoG_y, DoG_x
    """

    y, x = np.mgrid[-fsize + 1:fsize, -fsize + 1:fsize]
    DoG_y = (-(y / ((sigma ** 2)))) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    DoG_x = (-(x / ((sigma ** 2)))) * np.exp(-(x * x + y * y) / (2 * sigma * sigma))

    print(DoG_y)
    return DoG_y, DoG_x


def calculate_magnitude(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude


def make_noise(std, gray):
    height, width = gray.shape
    img_noise = np.zeros((height, width), dtype=np.float64)
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
    cv2.imshow('Dog Magnitude by filtering', calculate_magnitude(dog_x_image2 / 255., dog_y_image2 / 255.))
    cv2.waitKey()
    cv2.destroyAllWindows()