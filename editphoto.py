import cv2
import numpy as np

img1 = cv2.imread("1.png")
img2 = cv2.imread("2.png")
img3 = cv2.imread("3.png")
img4 = cv2.imread("4.png")
img5 = cv2.imread("5.png")
img6 = cv2.imread("6.png")

vis1 = np.concatenate((img1, img2, img3), axis=1)
vis2 = np.concatenate((img4, img5, img6), axis=1)
vis3 = np.concatenate((vis1, vis2), axis=0)
vis3 = cv2.resize(vis3, None, fx=0.5, fy=0.5)
cv2.imshow("t", vis3)
cv2.waitKey(0)
cv2.imwrite('out.png', vis3)