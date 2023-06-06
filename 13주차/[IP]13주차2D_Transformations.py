import numpy as np
import cv2
import matplotlib.pyplot as plt

# ceil은 수학기호로 [x]를 의미 즉 i >= x인 최소 정수 i를 의미.
# 예를들어, x가 3.6이면 i >= 3.6인 최소 정수를 의미하므로 4
# int() 소수점 이하를 버림.
# 예를들어, int(3.6) == 3

def forward(src, M, fit=False, m_type=''):

    print('< forward >')
    print('M')
    print(M)
    h, w = src.shape
    print('original img shape')
    print('h : {}, w : {}'.format(h, w))

    if fit == False:

        dst = np.zeros((h, w), dtype=np.float32)
        if m_type == 'scaling':
            ratio = M[0, 0]
            dst = np.zeros((int(ratio * h), int(ratio * w)), dtype=np.float32)
        h_, w_ = dst.shape
        N = np.zeros(dst.shape)

        for row in range(h):
            for col in range(w):
                # P : Point vector (x,y,1)
                P = np.array([
                    [col],  # x
                    [row],  # y
                    [1]
                ])

                P_dst = np.dot(M, P)  # (x,y,1) vector와 Translation matrix를 곱함
                dst_col = P_dst[0][0]  # x
                dst_row = P_dst[1][0]  # y

                # ceil은 수학기호로 [x]를 의미 즉 i >= x인 최소 정수를 의미.
                dst_col_right = int(np.ceil(dst_col))
                dst_col_left = int(dst_col)

                dst_row_bottom = int(np.ceil(dst_row))
                dst_row_top = int(dst_row)

                # index를 초과하는 부분은 값을 채우지 않음.

                if dst_row_top < 0 or dst_col_left < 0 or dst_row_bottom >= h_ or dst_col_right >= w_:
                    continue

                # dst_row,col이 정수이면 original 좌표에서 값을 가져온 후 counting을 한다.
                dst[dst_row_top, dst_col_left] += src[row, col]
                N[dst_row_top, dst_col_left] += 1

                # dst_col 즉 x' 좌표가 소수라면
                if dst_col_right != dst_col_left:
                    dst[dst_row_top, dst_col_right] += src[row, col]
                    N[dst_row_top, dst_col_right] += 1

                # dst_row 즉 y' 좌표가 소수라면
                if dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_left] += src[row, col]
                    N[dst_row_bottom, dst_col_left] += 1

                # dst_col, dst_row 즉 x',y' 모두 좌표가 소수라면
                if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                    dst[dst_row_bottom, dst_col_right] += src[row, col]
                    N[dst_row_bottom, dst_col_right] += 1


        N[N == 0] = 1
        dst = np.round(dst / N)
        dst = dst.astype(np.uint8)

    print('dst shape : {}'.format(dst.shape))
    print('dst min : {} dst max : {}'.format(np.min(dst),np.max(dst)))
    return dst

def backward(src, M, fit=False, m_type=''):

    print('< backward >')
    print('M')
    print(M)

    h, w = src.shape
    # M 역행렬 구하기
    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)

    if fit == False:

        dst = np.zeros((h, w), dtype=np.float32)

        if m_type == 'scaling':
            ratio = M[0, 0]
            dst = np.zeros((int(ratio * h), int(ratio * w)), dtype=np.float32)
        h_, w_ = dst.shape

        for row in range(h_):
            for col in range(w_):
                P_dst = np.array([
                    [col],
                    [row],
                    [1]
                ])
                # original 좌표로 매핑
                P = np.dot(M_inv, P_dst)
                src_col = P[0, 0]
                src_row = P[1, 0]
                # bilinear interpolation

                src_col_right = int(np.ceil(src_col))
                src_col_left = int(src_col)

                src_row_bottom = int(np.ceil(src_row))
                src_row_top = int(src_row)

                # index를 초과하는 부분에 대해서는 값을 채우지 않음
                if src_col_right >= w or src_row_bottom >= h or src_col_left < 0 or src_row_top < 0:
                    continue

                s = src_col - src_col_left
                t = src_row - src_row_top

                intensity = (1 - s) * (1 - t) * src[src_row_top, src_col_left] \
                            + s * (1 - t) * src[src_row_top, src_col_right] \
                            + (1 - s) * t * src[src_row_bottom, src_col_left] \
                            + s * t * src[src_row_bottom, src_col_right]

                dst[row, col] = intensity

        dst = dst.astype(np.uint8)

    print('dst shape : {}'.format(dst.shape))
    print('dst min : {} dst max : {}'.format(np.min(dst),np.max(dst)))

    return dst

def main():

    src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE)
    # src = cv2.resize(src, None, fx=0.5, fy=0.5)

    # translation
    M_tr = np.array([
        [1, 0, 50],
        [0, 1, 100],
        [0, 0, 1]
    ])

    # Rotation
    angle = 20
    M_ro = np.array([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0],
        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],
        [0, 0, 1]])

    # Shearing
    M_sh = np.array([
        [1, 0, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])

    # Scaling
    M_sc = np.array([
        [1.8, 0, 0],
        [0, 1.8, 0],
        [0, 0, 1]
    ])

    # 순서 주의
    # Scaling -> Shearing -> Rotation -> Translation
    M = np.dot(np.dot(np.dot(M_tr,M_ro),M_sh),M_sc)

    # dst_for = forward(src, M_sc, m_type='scaling')
    # dst_back = backward(src, M_sc, m_type='scaling')

    dst_for = forward(src, M_sh)
    dst_back = backward(src, M_sh)

    cv2.imshow('original', src)
    cv2.imshow('forward', dst_for)
    cv2.imshow('backward', dst_back)
    cv2.waitKey()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    main()