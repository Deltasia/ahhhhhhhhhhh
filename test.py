import cv2
import platform

# Choose appropriate backend based on OS
system = platform.system()
if system == "Darwin":  # macOS
    backend = cv2.CAP_AVFOUNDATION
elif system == "Windows":
    backend = cv2.CAP_DSHOW
elif system == "Linux":
    backend = cv2.CAP_V4L2
else:
    backend = cv2.CAP_ANY

print(f"Running on {system}, using backend: {backend}")

try:
    cap = cv2.VideoCapture(1, backend)
    if not cap.isOpened():
        print("Camera 1 failed, trying camera 0...")
        cap = cv2.VideoCapture(0, backend)
    
    ret, frame = cap.read()
    print("ret=", ret)
    if ret:
        cv2.imshow("Test", frame)
        cv2.waitKey(0)
    else:
        print("Failed to read frame from camera")
        
except Exception as e:
    print(f"Error accessing camera: {e}")
finally:
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()