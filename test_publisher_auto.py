import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
import requests
import threading
import time

import matplotlib.pyplot as plt

class PID:
    def __init__(self, Kp=0.5, Ki=0.0, Kd=0.1):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt=0.1):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

class YoloRobotController(Node):
    def __init__(self, yolo_api_url="http://localhost:8080/api/state"):
        super().__init__('yolo_robot_controller')
        self.publisher_ = self.create_publisher(Int32, 'keyboard', 10)
        self.yolo_api_url = yolo_api_url

        self.mode = "keyboard"  # default: keyboard, can toggle to 'auto'
        self.mode_lock = threading.Lock()
        self.running = True

        self.pid = PID(Kp=0.5, Ki=0.0, Kd=0.1)
        self.error_history = []

        # Thread สำหรับ loop autonomous
        self.auto_thread = threading.Thread(target=self.autonomous_loop, daemon=True)
        self.auto_thread.start()

    def toggle_mode(self):
        """สลับระหว่าง keyboard กับ auto"""
        with self.mode_lock:
            self.mode = "auto" if self.mode == "keyboard" else "keyboard"
            self.get_logger().info(f"Mode changed to: {self.mode}")

    def autonomous_loop(self):
        """Loop อ่าน YOLO API และ publish key code ในโหมดอัตโนมัติ"""
        while self.running:
            with self.mode_lock:
                if self.mode != "auto":
                    time.sleep(0.1)
                    continue

            try:
                resp = requests.get(self.yolo_api_url, timeout=0.2)
                if resp.status_code != 200:
                    time.sleep(0.1)
                    continue
                data = resp.json()
                latest = data.get("latest", None)
                if not latest:
                    time.sleep(0.1)
                    continue

                key_code = self.decide_movement(latest)
                if key_code:
                    msg = Int32()
                    msg.data = key_code
                    self.publisher_.publish(msg)
                    self.get_logger().info(f"Autonomous move key published: {chr(key_code)}")

                time.sleep(0.1)
            except Exception as e:
                self.get_logger().warn(f"YOLO API request failed: {e}")
                time.sleep(0.2)

    def decide_movement(self, measurement):
        if measurement is None or measurement.get("distance_cm") is None:
            return ord('a')
        frame_center = 320  # เฟรมเราตั้งกว้าง 640
        obj_center_x = measurement["center"]["x"]
        distance_cm = measurement["distance_cm"]

        # คำนวณ error และ PID
        error = obj_center_x - frame_center
        self.error_history.append(error)
        pid_output = self.pid.compute(error)

        # แปลงค่า PID output เป็น key code
        if abs(error) < 10:
            move = 'w' if distance_cm > 20 else 's' 
        elif pid_output > 0:
            move = 'd'  # เลี้ยวขวา
        else:
            move = 'a'  # เลี้ยวซ้าย

        return ord(move)

        # # ----------ปรับค่า logic เพิ่มด้วย---------
        # if distance_cm > 50: 
        #     if obj_center_x < frame_center - 50:
        #         return ord('a')
        #     elif obj_center_x > frame_center + 50:
        #         return ord('d')
        #     else:
        #         return ord('w') 
        # elif distance_cm < 20:
        #     return ord('s') 
        # else:
        #     return None
        # # ----------ปรับค่า logic เพิ่มด้วย---------

    def shutdown(self):
        self.running = False
        self.auto_thread.join()
        if self.error_history:
            plt.plot(self.error_history)
            plt.xlabel("Time step")
            plt.ylabel("PID error (pixels)")
            plt.title("PID error over time (oscillation)")
            plt.show()


def main(args=None):
    rclpy.init(args=args)
    controller = YoloRobotController()

    try:
        print("Press 't' + Enter to toggle mode (keyboard/auto). Ctrl+C to exit.")
        while rclpy.ok():
            rclpy.spin_once(controller, timeout_sec=0.1)
            user_input = input()
            if user_input.lower() == 't':
                controller.toggle_mode()
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        controller.shutdown()
        controller.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
