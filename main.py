import cv2
import cv2.aruco as aruco
import numpy as np

# -----------------------------
# YOUR CALIBRATED VALUES
# -----------------------------
cv2.namedWindow("Pose Estimation", cv2.WINDOW_NORMAL)
camera_matrix = np.array([
    [1267.5622, 0, 958.4879],
    [0, 1290.0316, 640.0982],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.array([
    [0.059346, -0.130095, 0.013055, 0.000567, 0.086800]
], dtype=np.float32)

# REAL marker side length in meters
marker_length = 0.025   # <-- change this to your real marker size

# -----------------------------

cap = cv2.VideoCapture("http://10.42.0.45:4747/video", cv2.CAP_FFMPEG)


aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()

        # Draw detected markers
        aruco.drawDetectedMarkers(frame, corners, ids)

        # Estimate pose
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
            corners,
            marker_length,
            camera_matrix,
            dist_coeffs
        )

        for i in range(len(ids)):
            # Draw 3D axis
            cv2.drawFrameAxes(
                frame,
                camera_matrix,
                dist_coeffs,
                rvecs[i],
                tvecs[i],
                0.03   # axis length (meters)
            )

            # Distance from camera
            distance = np.linalg.norm(tvecs[i])

            print(f"ID {ids[i]} | Distance: {distance:.3f} m")

            cv2.putText(frame,
                        f"ID:{ids[i]}  Dist:{distance:.2f}m",
                        (10, 40 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2)

    cv2.imshow("Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
