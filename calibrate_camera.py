import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (8, 5)
square_size = 20.0  

objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= square_size 

objpoints = []
imgpoints = []

# ของโจ้(image_dir) fx=1483.96, fy=1487.91 (pixel units)
# ของพี่(calib_imagess) fx=1386.81, fy=1381.46 (pixel units)
image_dir = "calib_images"
images = glob.glob(os.path.join(image_dir, "*.jpg"))

print(f"Found {len(images)} images")
print(images)

valid_images = []
for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Cannot read: {fname}")
    else:
        valid_images.append(img)

print(f"Valid images: {len(valid_images)}")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        imgpoints.append(corners)
    else:
        print(f"Chessboard not found: {fname}")

    print(f"Found corners in {len(imgpoints)} images")

# คำนวณ Calibration
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix (Intrinsic):\n", mtx)
print("Distortion coefficients:\n", dist)

# แปลงค่าที่ได้มา เช่น focal length
fx, fy = mtx[0,0], mtx[1,1]
cx, cy = mtx[0,2], mtx[1,2]

print(f"Focal length: fx={fx:.2f}, fy={fy:.2f} (pixel units)")
print(f"Principal point: ({cx:.2f}, {cy:.2f})")

# Save calibration data to npz file
np.savez('calibration_data.npz', camera_matrix=mtx, dist_coeffs=dist)
print("Calibration data saved to calibration_data.npz")