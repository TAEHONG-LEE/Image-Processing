import cv2
import numpy as np

src = np.zeros((300, 300, 3), dtype=np.uint8)
src[0, 0] = [1, 2, 3]
src[0, 1] = [4, 5, 6]
src[1, 0] = [7, 8, 9]

src[0, 2] = [255, 0, 0]
src[1, 1] = [255, 255, 255]

print(src.shape)
print(src[0, 0, 0], src[0, 0, 1], src[0, 0, 2])
print(src[0, 0])
print(src[0])
print(src)

cv2.imshow('src', src)
cv2.waitKey()
cv2.destroyAllWindows()




