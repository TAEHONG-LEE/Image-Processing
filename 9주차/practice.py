import numpy as np
import cv2

def get_hist(src, mask):

    """
    :param src: gray scale 이미지
    :param mask: masking을 하기 위한 값
    :return:
    """

    #######################################################################
    # TODO mask를 적용한 히스토그램 완성
    # TODO mask 값이 0인 영역은 픽셀의 빈도수를 세지 않음
    # TODO histogram을 생성해 주는 내장함수 사용금지. np.histogram, cv2.calHist
    #######################################################################
    hist = np.zeros((256,))
    for i in range(256):
        hist[i] = np.count_nonzero(src == i)
    hist[0] = 0

    return hist

intensity = np.array([i for i in range(256)])

meat = cv2.imread('meat.png', cv2.IMREAD_GRAYSCALE)
# mask 값을 0 또는 1로 만들기 위해 255로 나누어줌
mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE) / 255

# meat.png에서 관심 영역 해당하는 부분만 따로 추출.
src = (meat * mask).astype(np.uint8)

hist = get_hist(src, mask)
hist = hist.astype((np.int32))
intensity = np.array([i for i in range(256)])

p = hist / np.sum(hist)

def get_threshold_by_within_variance(intensity, p):
    """
    :param intensity: pixel 값 0 ~ 255 범위를 갖는 배열
    :param p: 상대도수 값
    :return: k: 최적의 threshold 값
    """

    ########################################################
    # TODO
    # TODO otsu_method 완성
    # TODO  1. within-class variance를 이용한 방법
    # TODO  교수님 이론 PPT 22 page 참고
    ########################################################

    ex1 = intensity * p
    ex2 = intensity * intensity * p

    within_var = np.zeros_like(p)

    for i in intensity:
        q1 = np.sum(p[:i+1])
        q2 = np.sum(p[i+1:])
        if q1 == 0:
            m1 = 0
        else:
            m1 = np.sum(ex1[:i]) / q1
        if q2 == 0:
            m2 = 0
        else:
            m2 = np.sum(ex1[i:]) / q2
        # within_var을 미리 계산하면 아래와 같다. (미리 q1과 q2를 곱해줌)
        within_var[i] = (np.sum(ex2[:i]) - (q1 * m1 ** 2)) + (np.sum(ex2[i:]) - (q1 * m2 ** 2))

    k = np.argmin(within_var)
    return k


def get_threshold_by_inter_variance(intensity, p):
    """
    :param p: 상대도수 값
    :return: k: 최적의 threshold 값
    """

    ########################################################
    # TODO
    # TODO otsu_method 완성
    # TODO  2. inter-class variance를 이용한 방법
    # TODO  Moving average를 이용하여 구현
    # TODO  교수님 이론 PPT 26 page 참고
    ########################################################

    p += 1e-7  # q1과 q2가 0일때 나눗셈을 진행할 경우 오류를 막기 위함

    q = np.cumsum(p)

    ex1 = intensity * p
    ex2 = intensity * intensity * p
    m = np.sum(ex1)

    between_var = np.zeros_like(p)

    for i in intensity:
        q1 = np.sum(p[:i+1])
        q2 = np.sum(p[i+1:])
        if q1 == 0:
            m1 = 0
        else:
            m1 = np.sum(ex1[:i]) / q1
        if q2 == 0:
            m2 = 0
        else:
            m2 = np.sum(ex1[i:]) / q2
        # Between-class variance의 식은 아래와 같다

        between_var[i] = q1 * (m1 - m) ** 2 + q2 * (m2 - m) ** 2

    k = np.argmax(between_var)
    return k
within_k = get_threshold_by_within_variance(intensity, p)
between_k = get_threshold_by_inter_variance(intensity, p)
print("within class를 이용하여 찾은 k : ", within_k)
print("between class를 이용하여 찾은 k : ", between_k)
# p = np.zeros(len(intensity))
#
# for i in intensity:
#     p[i] = np.count_nonzero(src == i)
# p[0] = 0
# p += 1e-7
#
# print(p)
# mn = np.sum(p)
# print(mn)
# minVar = 1000000000000000000
#
# ex1 = intensity * p
# ex2 = intensity * intensity * p
#
#
# for i in intensity:
#     q1 = np.sum(p[:i+1])
#     q2 = np.sum(p[i+1:])
#     m1 = np.sum(ex1[:i+1]) / q1
#     m2 = np.sum(ex1[i:]) / q2
#     var1 = np.abs(np.sum(ex2[:i])/q1 - m1**2)
#     var2 = np.abs(np.sum(ex2[i:])/q2 - m2**2)
#     withinVar = (q1*var1**2) + (q2*var2**2)
#     minVar = min(minVar, withinVar)
#     if minVar == withinVar:
#         withinVarK = i
#
# maxVar = 0
#
# for i in range(len(p)):
#     q1 = np.cumsum(p[:i + 1])[-1]
#     q2 = np.cumsum(p[i:])[-1]
#     m1 = np.cumsum(ex1[:i + 1])[-1] / q1
#     m2 = np.cumsum(ex1[i:])[-1] / q2
#
#     betweenVar = q1*q2*((m1 - m2)**2)
#
#     maxVar = max(maxVar, betweenVar)
#     if maxVar == betweenVar:
#         betweenVarK = i
#
# print(withinVarK, betweenVarK)

