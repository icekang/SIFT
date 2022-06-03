interval = range(5)
for i in zip(interval, interval[1:], interval[2:]):
    print(i)
# import cv2 as cv
# import pysift

# img = cv.imread('imgg.jpg')
# gray = cv.imread('imgg.jpg', 0)
# kp, des = pysift.computeKeypointsAndDescriptors(gray)

# img = cv.drawKeypoints(gray, kp, img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
# cv.imwrite('sift_keypoints.jpg',img)