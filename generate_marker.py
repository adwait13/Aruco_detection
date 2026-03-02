import cv2
import cv2.aruco as aruco

aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

marker = aruco.generateImageMarker(aruco_dict, 1, 400)

cv2.imwrite("marker_1.png", marker)