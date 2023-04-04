import cv2
import numpy as np
import matplotlib.pyplot as plt


src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE).astype(np.float32)
pad_img = np.arange(36).reshape([6,6])
mask = np.arange(9).reshape([3,3])
dst = np.zeros([4, 4])
x = np.linspace(-5, 5, 1000)
mean = x.mean()

(h, w) = dst.shape
(f_h, f_w) = mask.shape
x_shape = x.shape
mean = x.mean()

print(pad_img)
print(mask)
print(dst)
print(f_h)
print(x_shape)
print(mean)

print(pad_img[0:3, 0:3]*mask)

for row in range(h):
    for col in range(w):
        dst[row, col] = np.sum(pad_img[row:row+f_h, col:col+f_w]*mask)
print(dst)