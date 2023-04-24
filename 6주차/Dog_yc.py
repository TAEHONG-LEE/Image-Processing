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

    y, x = np.mgrid[-(f_h // 2):f_h // 2 + 1, -(f_w // 2):f_w // 2 + 1]

    gaussian_filter = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (sigma ** 2)

    return gaussian_filter

def my_filtering(image, filter):
    filtered_image = np.zeros((min(image.shape[0], image.shape[1]), min(image.shape[0], image.shape[1])))

    for i in range(min(image.shape[0], image.shape[1])):
        for j in range(min(image.shape[0], image.shape[1])):
            filtered_image[i, j] = np.sum(image[i:i + filter.shape[0], j:j + filter.shape[1]] * filter)

    return filtered_image


def get_DoG_filter_by_filtering(fsize, sigma):
    derivate_y = np.array([[-1 / 2],
                          [0],
                          [1 / 2]])
    derivate_x = derivate_y.T

    gaussian_filter_y = my_get_Gaussian_filter((fsize + 2, fsize), sigma)
    gaussian_filter_x = my_get_Gaussian_filter((fsize, fsize + 2), sigma)

    dog_y_filter = my_filtering(gaussian_filter_y, derivate_y)
    dog_x_filter = my_filtering(gaussian_filter_x, derivate_x)
    print(dog_x_filter)

    return dog_y_filter, dog_x_filter

def get_DoG_filter_by_expression(fsize, sigma):
    y, x = np.mgrid[-(fsize // 2):fsize // 2 + 1, -(fsize // 2):fsize // 2 + 1]

    DoG_x = (-x / sigma ** 2) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))
    DoG_y = (-y / sigma ** 2) * (np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)))

    return DoG_y, DoG_x

def calculate_magnitude(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude

def make_noise(std, gray):

    height, width = gray.shape
    img_noise = np.zeros((height, width), dtype=np.float)
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

    dog_2_y, dog_2_x  = get_DoG_filter_by_filtering(5, 1)
    dog_y_image2 = cv2.filter2D(image, -1, dog_2_y, borderType=cv2.BORDER_CONSTANT)
    dog_x_image2 = cv2.filter2D(image, -1, dog_2_x, borderType=cv2.BORDER_CONSTANT)

    # 수식으로 만든 것과 filtering으로 만든 Dog filter 사이의 Magnitude 비교
    cv2.imshow('Dog Magnitude by expression', calculate_magnitude(dog_x_image / 255., dog_y_image / 255.))
    cv2.imshow('Dog Magnitude by filtering', calculate_magnitude(dog_x_image2 / 255., dog_y_image2 / 255.))
    cv2.waitKey()