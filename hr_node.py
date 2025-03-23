#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque

class HeartRateNode(Node):
    def __init__(self):
        super().__init__('heart_rate_node')
        self.publisher_ = self.create_publisher(Float32, 'heart_rate', 10)
        self.timer = self.create_timer(0.033, self.timer_callback)  # 30 FPS
        
        # Load haarcascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize deque for storing intensity values
        self.window_size = 150  # Number of frames to consider
        self.intensities = deque(maxlen=self.window_size)
        
        # Bandpass filter parameters (for heart rate range)
        self.lowcut = 0.75  # 0.75 Hz (45 bpm)
        self.highcut = 4.0  # 4.0 Hz (240 bpm)
        self.frame_rate = 30  # Assuming 30 fps
        
        # Moving average for heart rate smoothing
        self.hr_history = deque(maxlen=10)
        
        # Long-term baseline adjustment parameters
        self.baseline_alpha = 0.1  # Weight for updating the baseline
        self.min_signal_quality = 0.5  # Minimum signal quality to update baseline
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(2)
        
        # Calibrate the system
        self.baseline_hr = self.calibrate()
        
        self.get_logger().info('Heart Rate Node has been initialized')

    def butter_bandpass(self, lowcut, highcut, fs, order=3):
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def bandpass_filter(self, data, lowcut, highcut, fs, order=3):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        return filtfilt(b, a, data)

    def calibrate(self, calibration_time=10):
        self.get_logger().info("Calibrating... Please remain still.")
        calibration_data = []
        start_time = cv2.getTickCount()

        while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < calibration_time:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))
                (x, y, w, h) = face
                roi = frame[y:y+h//4, x:x+w]

                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                calibration_data.append(np.mean(roi[:, :, 2]))

        if not calibration_data:
            self.get_logger().warn("No calibration data collected. Using default baseline.")
            return 72.0

        calibration_signal = self.bandpass_filter(calibration_data, self.lowcut, self.highcut, self.frame_rate)
        calibration_signal = (calibration_signal - np.mean(calibration_signal)) / np.std(calibration_signal)

        peaks, _ = find_peaks(calibration_signal, height=0.5, distance=self.frame_rate//2)
        peak_times = np.array(peaks) / self.frame_rate

        if len(peak_times) >= 2:
            baseline_hr = 60 / np.mean(np.diff(peak_times))
            self.get_logger().info(f"Calibration complete. Baseline heart rate: {baseline_hr:.2f} bpm")
        else:
            baseline_hr = 72.0
            self.get_logger().warn("Calibration failed. Using default baseline heart rate.")

        return baseline_hr

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to read from webcam")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))
            (x, y, w, h) = face
            roi = frame[y:y+h//4, x:x+w]

            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            self.intensities.append(np.mean(roi[:, :, 2]))

            if len(self.intensities) == self.window_size:
                filtered_signal = self.bandpass_filter(list(self.intensities), self.lowcut, self.highcut, self.frame_rate)
                normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)
                signal_quality = np.max(normalized_signal) - np.min(normalized_signal)

                peaks, _ = find_peaks(normalized_signal, height=0.5, distance=self.frame_rate//2)
                peak_times = np.array(peaks) / self.frame_rate

                if len(peak_times) >= 2:
                    heart_rate = 60 / np.mean(np.diff(peak_times))
                    heart_rate = np.clip(heart_rate, 40, 180)

                    self.hr_history.append(heart_rate)
                    smoothed_hr = np.mean(self.hr_history)

                    if signal_quality > self.min_signal_quality:
                        self.baseline_hr = self.baseline_alpha * smoothed_hr + (1 - self.baseline_alpha) * self.baseline_hr

                    # Publish the heart rate
                    msg = Float32()
                    msg.data = float(smoothed_hr)
                    self.publisher_.publish(msg)
                    self.get_logger().info(f'Published heart rate: {smoothed_hr:.2f} bpm')

    def __del__(self):
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = HeartRateNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 