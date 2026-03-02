import cv2
import os

cap = cv2.VideoCapture(2)

num = 0
root = os.getcwd()
images_dir = os.path.join(root, 'droidcam_images')
while cap.isOpened():

    succes, img = cap.read()

    k = cv2.waitKey(5)

    if k == ord('q'):
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        save_path = os.path.join(images_dir, 'img' + str(num) + '.png')
        cv2.imwrite(save_path, img)
        print(f"Image saved at {save_path}")
        print("image saved!")
        num += 1

    cv2.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()