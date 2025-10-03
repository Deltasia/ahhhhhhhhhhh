# Camera Calibration Script - Cross-platform compatible

import numpy as np
import cv2 as cv
import glob
 
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
 
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
 
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# Look for images in multiple possible directories
image_paths = []
search_dirs = ['*.jpg', '*.png', 'images/*.jpg', 'images/*.png', 'image_dir/*.jpg', 'image_dir/*.png']
for pattern in search_dirs:
    image_paths.extend(glob.glob(pattern))

if not image_paths:
    print("No calibration images found. Please ensure you have .jpg or .png files in the current directory or images/ folder.")
    exit(1)

print(f"Found {len(image_paths)} calibration images")
 
for fname in image_paths:
    print(f"Processing: {fname}")
    img = cv.imread(fname)
    if img is None:
        print(f"Could not read image: {fname}")
        continue
        
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
 
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
 
    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)  # Fixed: was (8,6), should be (7,6)
        
        # Check if we're in a GUI environment before showing windows
        try:
            cv.imshow('img', img)
            cv.waitKey(500)
        except cv.error as e:
            print(f"Cannot display image (no GUI): {e}")
            print("Continuing without display...")
 
try:
    cv.destroyAllWindows()
except Exception:
    pass  # Ignore if no windows were created

# Calibrate camera
if len(objpoints) == 0:
    print("No valid calibration images found. Cannot proceed with calibration.")
    exit(1)

image_shape = None
for fname in image_paths:
    img = cv.imread(fname)
    if img is None:
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    if image_shape is None:
        image_shape = gray.shape[::-1]  # (width, height)
        break

if image_shape is None:
    print("Could not determine image shape from calibration images.")
    exit(1)

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

print("Camera matrix:\n", mtx)        # intrinsic parameters
print("Distortion coefficients:\n", dist)  # distortion coefficients

# Save calibration data
np.savez('calibration_data.npz', camera_matrix=mtx, dist_coeffs=dist, rvecs=rvecs, tvecs=tvecs)
print("Calibration data saved to calibration_data.npz")

# Test undistortion if test image exists
test_images = glob.glob('test.jpg') + glob.glob('test.png')
if test_images:
    test_img_path = test_images[0]
    img = cv.imread(test_img_path)
    if img is not None:
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        
        try:
            cv.imshow("original", img)
            cv.imshow("undistorted", dst)
            cv.waitKey(0)
            cv.destroyAllWindows()
        except cv.error as e:
            print(f"Cannot display images (no GUI): {e}")
            print("Undistortion completed successfully but cannot display.")
        
        # Save undistorted image
        cv.imwrite('undistorted_test.jpg', dst)
        print("Undistorted test image saved as undistorted_test.jpg")
    else:
        print(f"Could not read test image: {test_img_path}")
else:
    print("No test image (test.jpg or test.png) found for undistortion demo.")

# Summary
print("\nCalibration completed successfully!")
print(f"- Processed {len(objpoints)} valid images")
print(f"- Image size: {image_shape}")
print("- Calibration data saved to: calibration_data.npz")

