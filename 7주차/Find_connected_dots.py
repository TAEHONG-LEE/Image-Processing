import numpy as np
import matplotlib.pyplot as plt

def Find_connected_dots_4neighbors(src, row, col, coordinates):

    # 첫번째 조건은 중복을 제거하기 위함
    # 두번째 조건은 0.5가 아닌 값을 포함시키지 않기 위함
    if ((row, col) in coordinates) or src[row,col] != 128:
        return coordinates

    # 위의 두 조건에 해당하지 않으면 connected dots에 포함시킨다.
    coordinates.append((row,col))

    h, w = src.shape

    # 4-neighbor
    # 이웃한 4개 좌표
    # -------------------
    # | 0   | 1     |0
    # | 1   | 1     |1
    # | 0   | 1     |0
    # (row,col)을 기준으로 위 아래 왼쪽 오른쪽 방향에 있는 -1값을 모두 labelling하기 위한 작업

    # up
    if row > 0:
        coordinates = Find_connected_dots_4neighbors(src, row - 1, col, coordinates)
    # down
    if row < h-1:
        coordinates = Find_connected_dots_4neighbors(src, row + 1, col, coordinates)
    # left
    if col > 0:
        coordinates = Find_connected_dots_4neighbors(src, row, col - 1, coordinates)
    # right
    if col < w-1:
        coordinates = Find_connected_dots_4neighbors(src, row, col + 1, coordinates)

    # 중복 제거
    return list(set(coordinates))


def Find_connected_dots_8neighbors(src, row, col, coordinates):

    # 첫번째 조건은 중복을 제거하기 위함
    # 두번째 조건은 128가 아닌 값을 포함시키지 않기 위함
    if ((row, col) in coordinates) or src[row, col] != 128:
        return coordinates

    # 8-neighbor
    # 이웃한 8개 좌표
    # -------------------
    # | 1   | 2     |3
    # | 4   | 5     |6
    # | 7   | 8     |9

    # 위의 두 조건에 해당하지 않으면 connected components에 포함시킨다.(자기 자신, 5번에 해당)
    coordinates.append((row, col))
    h, w = src.shape

    # 1번 좌표
    if row > 0 and col > 0:
        coordinates = Find_connected_dots_8neighbors(src, row - 1, col - 1, coordinates)
    # 2번 좌표
    if row > 0:
        coordinates = Find_connected_dots_8neighbors(src, row - 1, col, coordinates)
    # 3번 좌표
    if row > 0 and col < w - 1:
        coordinates = Find_connected_dots_8neighbors(src, row - 1, col + 1, coordinates)
    # 4번 좌표
    if col > 0:
        coordinates = Find_connected_dots_8neighbors(src, row, col - 1, coordinates)
    # 6번 좌표
    if col < w - 1:
        coordinates = Find_connected_dots_8neighbors(src, row, col + 1, coordinates)
    # 7번 좌표
    if row < h - 1 and col > 0:
        coordinates = Find_connected_dots_8neighbors(src, row + 1, col - 1, coordinates)
    # 8번 좌표
    if row < h - 1:
        coordinates = Find_connected_dots_8neighbors(src, row + 1, col, coordinates)
    # 9번 좌표
    if row < h - 1 and col < w - 1:
        coordinates = Find_connected_dots_8neighbors(src, row + 1, col + 1, coordinates)

    # 중복 제거
    return list(set(coordinates))


def connect(src, connection_type='4neighbor'):

    """
    :param src: 입력 이미지
    :param connection_type: connection type
    :return: dst 연결시킨 이미지
    """

    h, w = src.shape
    dst = src.copy()

    value = 255
    for row in range(h):
        for col in range(w):
            if dst[row, col] == 128:
                # neighbor 유형에 따라
                # 4 neighbor
                if connection_type =='4neighbor':
                    coordinates = Find_connected_dots_4neighbors(dst, row, col, [])
                else:
                    # 8 neighbor
                    coordinates = Find_connected_dots_8neighbors(dst, row, col, [])

                if len(coordinates) != 0:
                    # coordinates에 값이 들어있을때 까지 반복
                    while coordinates:
                        r, c = coordinates.pop()
                        dst[r, c] = value
                    return dst

                else:
                    continue
    return dst

if __name__ == '__main__':

    src = np.array([[0, 128, 128, 0, 0],
                  [0, 128, 128, 0, 0],
                  [0, 0, 128, 0, 0],
                  [0, 0, 0, 128, 128],
                  [0, 0, 0, 128, 128]], dtype=np.uint8)

    plt.title('input')
    plt.imshow(src, cmap='gray', vmin=0, vmax=255)
    plt.show()

    neighbor4_output = connect(src, connection_type='4neighbor')
    neighbor8_output = connect(src, connection_type='8neighbor')

    plt.title('4 neighbor')
    plt.imshow(neighbor4_output, cmap='gray', vmin=0, vmax=255)
    plt.show()

    plt.title('8 neighbor')
    plt.imshow(neighbor8_output, cmap='gray', vmin=0, vmax=255)
    plt.show()

