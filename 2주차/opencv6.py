import cv2

src = cv2.imread('./Lena.png')
rgb = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)



cv2.imshow('original', src)
cv2.imshow('RGB', rgb)
cv2.imshow('gray', gray)

print('[BGR] {0}' .format(src[0, 0]))
print('[RGB] {0}' .format(rgb[0, 0]))
print('[GRAY] {0}' .format(gray[0, 0]))

cv2.waitKey()
cv2.destroyAllWindows()