import cv2  
import json
import os

ARUCO_DICT = cv2.aruco.DICT_6X6_250

# define the charuco board parameters
SQUARES_VERTICALLY = 5
SQUARES_HORIZONTALLY = 7
SQUARE_LENGTH = 0.035
MARKER_LENGTH = 0.025
   

def get_calibration_parameters(img_dir):
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard((SQUARES_VERTICALLY, SQUARES_HORIZONTALLY), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    # Load images from directory
    image_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".png")]
    print(f"Found {len(image_files)} images in {img_dir}")
    all_charuco_ids = []
    all_charuco_corners = []

    # Loop over images and extraction of corners
    for image_file in image_files:
        image = cv2.imread(image_file)
        #print(image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgSize = (image.shape[1], image.shape[0])

        image_copy = image.copy()
        marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
        
        if len(marker_ids) > 0: # If at least one marker is detected
            # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
            ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, image, board)

            if charucoIds is not None and len(charucoCorners) > 3:
                all_charuco_corners.append(charucoCorners)
                all_charuco_ids.append(charucoIds)
    #print(all_charuco_corners, all_charuco_ids)
    # Calibrate camera with extracted information
    result, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(all_charuco_corners, all_charuco_ids, board, imgSize, None, None)
    print("Reprojection error:", result)

    return result, mtx, dist,rvecs,tvecs


root = os.getcwd()
imgs_dir = os.path.join(root, 'droidcam_images')

result, mtx, dist,rvecs,tvecs = get_calibration_parameters(img_dir=imgs_dir)
print("\n=== Camera Intrinsics ===")

fx = mtx[0, 0]
fy = mtx[1, 1]
cx = mtx[0, 2]
cy = mtx[1, 2]

print(f"fx: {fx:.4f}")
print(f"fy: {fy:.4f}")
print(f"cx: {cx:.4f}")
print(f"cy: {cy:.4f}")

print("\n=== Distortion Coefficients ===")

k1, k2, p1, p2, k3 = dist[0]

print(f"k1 (radial): {k1:.6f}")
print(f"k2 (radial): {k2:.6f}")
print(f"p1 (tangential): {p1:.6f}")
print(f"p2 (tangential): {p2:.6f}")
print(f"k3 (radial): {k3:.6f}")

print(f"\nReprojection error: {result:.6f}")
