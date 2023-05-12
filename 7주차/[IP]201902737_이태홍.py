import cv2
import numpy as np

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (m_h, m_w) = filter.shape
    pad_img =my_padding(src, filter)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * filter)
    return dst

def convert_uint8(img):
    #이미지 출력을 위해서 타입을 변경 수행
    return ((img - np.min(img)) / np.max(img - np.min(img)) * 255).astype(np.uint8)

def get_DoG_filter(fsize, sigma):
    ###################################################
    # TODO                                            #
    # TODO DoG mask 완성                                    #
    # TODO DoG의 경우 과제가 진행중이기에 저장된 배열을 가지고 와서
    # TODO 불러오는 형식으로 진행함.
    # TODO 함수를 고칠 필요는 전혀 없음.
    ###################################################

    DoG_x = np.load('DoG_x.npy')
    DoG_y = np.load('DoG_y.npy')

    return DoG_x, DoG_y

def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    magnitude = np.sqrt(Ix*Ix + Iy*Iy)
    return magnitude

def calcAngle(Ix, Iy):
    #######################################
    # TODO                                #
    # calcAngle 완성                      #
    # angle     : ix와 iy의 angle         #
    #######################################
    row, col = Ix.shape
    angle = np.zeros((row, col))

    for r in range(row):
        for c in range(col):
            if Ix[r, c] == 0:
                if Iy[r, c] > 0:
                    angle[r, c] = 90.0
                elif Iy[r, c] < 0:
                    angle[r, c] = -90.0
                else:
                    angle[r, c] = 0.0
            else:
                angle[r, c] = np.rad2deg(np.arctan(Iy[r, c] / Ix[r, c]))
    return angle

def pixel_bilinear_coordinate(src, pixel_coordinate):
    ####################################################################################
    # TODO                                                                             #
    # TODO Pixel-Bilinear Interpolation 완성
    # TODO 진행과정
    # TODO 저번 실습을 참고로 픽셀 위치를 기반으로 주변 픽셀을 가져와서 Interpolation을 구현
    ####################################################################################

    h, w = src.shape
    y, x = pixel_coordinate

    # 주변 픽셀 위치 4개를 가져옴.
    # 가져오는 방식은 저번주 실습을 참고하여 가져오는 것을 추천.
    y_up = int(y)
    y_down = min(int(y + 1), h - 1)
    x_left = int(x)
    x_right = min(int(x + 1), w - 1)

    # x 비율, y 비율을 계산하는 코드
    # 저번 실습 자료 참고.
    t = y - y_up
    s = x - x_left

    # Bilinear Interpolation 구현 부분
    # 저번 실습 자료 참고.
    intensity = ((1 - s) * (1 - t) * src[y_up, x_left]) \
                + (s * (1 - t) * src[y_up, x_right]) \
                + ((1 - s) * t * src[y_down, x_left]) \
                + (s * t * src[y_down, x_right])

    return intensity

def non_maximum_supression_three_size(magnitude, angle):
    ####################################################################################
    # TODO
    # TODO non_maximum_supression
    # TODO largest_magnitude: non_maximum_supression 결과(가장 강한 edge만 남김)         #
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]

            # gradient의 degree는 edge와 수직방향이다.
            if 0 <= degree and degree < 45:
                rate = np.tan(np.deg2rad(degree))
                left_pixel_coordinate = (row + rate, col + 1)
                right_pixel_coordinate = (row - rate, col - 1)
                left_magnitude = pixel_bilinear_coordinate(magnitude, left_pixel_coordinate)
                right_magnitude = pixel_bilinear_coordinate(magnitude, right_pixel_coordinate)
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif 45 <= degree and degree <= 90:
                rate = np.tan(np.deg2rad(90 - degree))  # cotan = 1/tan
                up_pixel_coordinate = (row + 1, col + rate)
                down_pixel_coordinate = (row - 1, col - rate)
                up_magnitude = pixel_bilinear_coordinate(magnitude, up_pixel_coordinate)
                down_magnitude = pixel_bilinear_coordinate(magnitude, down_pixel_coordinate)
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -45 <= degree and degree < 0:
                rate = -np.tan(np.deg2rad(degree))
                left_pixel_coordinate = (row - rate, col + 1)
                right_pixel_coordinate = (row + rate, col - 1)
                left_magnitude = pixel_bilinear_coordinate(magnitude, left_pixel_coordinate)
                right_magnitude = pixel_bilinear_coordinate(magnitude, right_pixel_coordinate)
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -90 <= degree and degree < -45:
                rate = -np.tan(np.deg2rad(90 - degree))
                up_pixel_coordinate = (row - 1, col + rate)
                down_pixel_coordinate = (row + 1, col - rate)
                up_magnitude = pixel_bilinear_coordinate(magnitude, up_pixel_coordinate)
                down_magnitude = pixel_bilinear_coordinate(magnitude, down_pixel_coordinate)
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            else:
                print(row, col, 'error!  degree :', degree)

    return largest_magnitude

def non_maximum_supression_five_size(magnitude, angle, step = 0.5):
    ####################################################################################
    # TODO
    # TODO non_maximum_supression 완성 5x5 영역
    # TODO largest_magnitude: non_maximum_supression 결과(가장 강한 edge만 남김)
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))
    for row in range(2, h - 2):
        for col in range(2, w - 2):
            degree = angle[row, col]
            coordinates = []
            if 0 <= degree and degree < 45:
                rate = np.tan(np.deg2rad(degree))
                for i in np.arange(-2.0, 2.5, step):
                    row_coordinate = row + i * rate
                    col_coordinate = col + i
                    coordinates.append((row_coordinate, col_coordinate))
            elif 45 <= degree and degree <= 90:
                rate = np.tan(np.deg2rad(90 - degree))
                for i in np.arange(-2.0, 2.5, step):
                    row_coordinate = row + i
                    col_coordinate = col + i * rate
                    coordinates.append((row_coordinate, col_coordinate))
            elif -45 <= degree and degree < 0:
                rate = -np.tan(np.deg2rad(degree))
                for i in np.arange(-2.0, 2.5, step):
                    row_coordinate = row + i * rate
                    col_coordinate = col + i
                    coordinates.append((row_coordinate, col_coordinate))
            elif -90 <= degree and degree < -45:
                rate = -np.tan(np.deg2rad(90 - degree))
                for i in np.arange(-2.0, 2.5, step):
                    row_coordinate = row + i
                    col_coordinate = col + i * rate
                    coordinates.append((row_coordinate, col_coordinate))
            else:
                print(row, col, 'error!  degree :', degree)

            interpolated_magnitude = [pixel_bilinear_coordinate(magnitude, coordinate) \
                                     if coordinate != (row, col) else magnitude[row, col] for coordinate in coordinates]
            if max(interpolated_magnitude) == magnitude[row, col]:
                largest_magnitude[row, col] = magnitude[row, col]
    return largest_magnitude
def double_thresholding(src, high_threshold):

    ####################################################################################
    # TODO
    # TODO double_thresholding 완성
    # TODO Goal : Weak Edge와 Strong Edge를 토대로 연결성을 찾아서 최종적인 Canny Edge Detection 이미지를 도출
    # TODO 이 함수는 건드릴 필요가 없음.
    # TODO largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)
    # TODO double_thresholding 수행 high threshold value는 메인문에서 지정한 값만 사용할 것
    # TODO 3 x 3 non_maximum_supression의 high threshold 값: 40
    # TODO 5 x 5 non_maximum_supression의 high threshold 값: 29
    ####################################################################################

    dst = src.copy()
    # dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)
    (h, w) = dst.shape

    high_threshold_value = high_threshold

    low_threshold_value = high_threshold_value * 0.4

    for row in range(h):
        for col in range(w):
            if dst[row, col] >= high_threshold_value:
                dst[row, col] = 255
            elif dst[row, col] < low_threshold_value:
                dst[row, col] = 0
            else:
                ####################################################################
                # TODO
                # TODO Weak Edge일때 구현
                # TODO search_weak_edge 함수 설명 : Weak Edge를 찾아 배열에 저장하는 함수
                # TODO classify_edge : search_weak_edge를 통해 찾아낸 Weak Edge들을 이용하여 주변에
                #  Strong Edge가 있으면 weak Edge들을 Strong으로 변경 Edge 주변에 Strong이 없으면 Weak Edge를 버림.
                ####################################################################
                weak_edge = []
                weak_edge.append((row, col))
                weak_edge = search_weak_edge(dst, row, col, [], high_threshold_value, low_threshold_value)
                if classify_edge(dst, weak_edge, high_threshold_value):
                    for idx in range(len(weak_edge)):
                        (r, c) = weak_edge[idx]
                        dst[r, c] = 255
                else:
                    for idx in range(len(weak_edge)):
                        (r, c) = weak_edge[idx]
                        dst[r, c] = 0

    return dst

# 실습의 Find_connected_dots_8neighbors를 이용하여 함수를 작성함
def search_weak_edge(dst, row, col, coordinates, high_threshold_value, low_threshold_value):
    ####################################################################################
    # TODO
    # TODO search_weak_edge 함수
    # TODO Goal : 연결된 Weak Edge를 찾아서 저장하는 함수
    # TODO 구현의 자유도를 주기위해 실습을 참고하여 구현해도 되며
    # TODO 직접 생각해서 구현해도 무방함.
    ####################################################################################
    (h, w) = dst.shape

    # 첫번째 조건은 중복을 제거하기 위함
    # 두번째 조건은 128가 아닌 값을 포함시키지 않기 위함
    if ((row, col) in coordinates) or dst[row,col] < low_threshold_value or dst[row, col] >= high_threshold_value:
        return coordinates

    # 위의 두 조건에 해당하지 않으면 connected components에 포함시킨다.(자기 자신, 5번에 해당)
    coordinates.append((row, col))

    # 1번 좌표
    if row > 0 and col > 0:
        coordinates = search_weak_edge(dst, row - 1, col - 1, coordinates, high_threshold_value, low_threshold_value)
    # 2번 좌표
    if row > 0:
        coordinates = search_weak_edge(dst, row - 1, col, coordinates, high_threshold_value, low_threshold_value)
    # 3번 좌표
    if row > 0 and col < w - 1:
        coordinates = search_weak_edge(dst, row - 1, col + 1, coordinates, high_threshold_value, low_threshold_value)
    # 4번 좌표
    if col > 0:
        coordinates = search_weak_edge(dst, row, col - 1, coordinates, high_threshold_value, low_threshold_value)
    # 6번 좌표
    if col < w - 1:
        coordinates = search_weak_edge(dst, row, col + 1, coordinates, high_threshold_value, low_threshold_value)
    # 7번 좌표
    if row < h - 1 and col > 0:
        coordinates = search_weak_edge(dst, row + 1, col - 1, coordinates, high_threshold_value, low_threshold_value)
    # 8번 좌표
    if row < h - 1:
        coordinates = search_weak_edge(dst, row + 1, col, coordinates, high_threshold_value, low_threshold_value)
    # 9번 좌표
    if row < h - 1 and col < w - 1:
        coordinates = search_weak_edge(dst, row + 1, col + 1, coordinates, high_threshold_value, low_threshold_value)

    # 중복 제거
    return list(set(coordinates))


def classify_edge(dst, weak_edge, high_threshold_value):
    ####################################################################################
    # TODO
    # TODO weak edge가 strong edge랑 연결되어 있는지 확인한 후 edge임을 결정하는 함수
    # TODO 구현의 자유도를 주기위해 실습을 참고하여 구현해도 되며
    # TODO 직접 생각해서 구현해도 무방함.
    ####################################################################################
    (h, w) = dst.shape
    for edge in weak_edge:
        row, col = edge
        # 1번 좌표
        if row > 0 and col > 0:
            if (dst[row - 1, col - 1] >= high_threshold_value): return True
        # 2번 좌표
        if row > 0:
            if (dst[row - 1, col] >= high_threshold_value): return True
        # 3번 좌표
        if row > 0 and col < w - 1:
            if (dst[row - 1, col + 1] >= high_threshold_value): return True
        # 4번 좌표
        if col > 0:
            if (dst[row, col - 1] >= high_threshold_value): return True
        # 6번 좌표
        if col < w - 1:
            if (dst[row, col + 1] >= high_threshold_value): return True
        # 7번 좌표
        if row < h - 1 and col > 0:
            if (dst[row + 1, col - 1] >= high_threshold_value): return True
        # 8번 좌표
        if row < h - 1:
            if (dst[row + 1, col] >= high_threshold_value): return True
        # 9번 좌표
        if row < h - 1 and col < w - 1:
            if (dst[row + 1, col + 1] >= high_threshold_value): return True

    return False
def my_canny_edge_detection(src, fsize=3, sigma=1):

    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    DoG_x, DoG_y = get_DoG_filter(fsize, sigma)
    Ix = my_filtering(src, DoG_x)
    Iy = my_filtering(src, DoG_y)

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    cv2.imshow('magnitude', convert_uint8(magnitude))

    angle = calcAngle(Ix, Iy)

    #non-maximum suppression 수행
    larger_magnitude2 = non_maximum_supression_three_size(magnitude, angle)
    cv2.imshow('NMS_Three', convert_uint8(larger_magnitude2))
    larger_magnitude3 = non_maximum_supression_five_size(magnitude, angle)
    cv2.imshow('NMS_Five', convert_uint8(larger_magnitude3))

    #double thresholding 수행
    dst = double_thresholding(larger_magnitude2,40)
    dst2 = double_thresholding(larger_magnitude3,29)
    return dst, dst2

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    dst, dst2 = my_canny_edge_detection(src)

    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.imshow('my canny edge detection2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


