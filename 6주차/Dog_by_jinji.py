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
    y, x = np.mgrid[-int(f_h / 2):int(f_h / 2) + 1, -int(f_w / 2):int(f_w / 2) + 1]  # ???

    # 2차 gaussian mask 생성
    gaussian_filter = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # ???

    # mask 총합 1 : 평균 밝기의 변화가 없도록 하기 위함
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter) # ???
    return gaussian_filter


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
    # TODO      (예를 들어, derivate_x = [[-1, 0, 1]])
    # TODO  2.2 gaussian filter 생성 (임의의 크기 고려) ( my_get_Gaussian_filter 함수 사용)
    # TODO  2.3 2.1에서 구한 1차원 미분 filter를 사용하여 2.2에서 구한 gaussian_filter를 filtering
    # TODO  2.4 y,x 각각의 방향으로 미분 filtering한 결과 값을 반환 -> dog_y_filer, dog_x_filter
    # TODO NOTE filtering시 내장 함수 사용 금지 (cv2.filter2D)
    ############################################################################
    # ???
    # 2.1
    derivate_x = np.array([np.mgrid[-int(fsize / 2):int(fsize / 2) + 1]])
    derivate_y = np.array([np.mgrid[-int(fsize / 2):int(fsize / 2) + 1]]).T

    # 2.2
    gaussian_filter = my_get_Gaussian_filter(fsize, sigma = sigma)

    # 2.3
    dog_y_filter = gaussian_filter * derivate_x # ???
    dog_x_filter = gaussian_filter * derivate_y # ???

    return dog_y_filter, dog_x_filter

def calculate_magnitude(sobel_x, sobel_y):
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    return magnitude

def get_DoG_filter_by_expression(fsize, sigma):
    """

    :param fsize: dog filter 크기
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
    y, x = np.mgrid[-int(fsize / 2):int(fsize / 2) + 1, -int(fsize / 2):int(fsize / 2) + 1]  # ???

    DoG_x = -x / (2 * np.pi * sigma ** 4) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # ???
    DoG_y = -y / (2 * np.pi * sigma ** 4) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))  # ???

    return DoG_y, DoG_x

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
