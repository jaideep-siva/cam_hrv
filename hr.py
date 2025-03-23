import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from collections import deque

# Load haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize deque for storing intensity values
window_size = 150  # Number of frames to consider
intensities = deque(maxlen=window_size)

# Bandpass filter parameters (for heart rate range)
lowcut = 0.75  # 0.75 Hz (45 bpm)
highcut = 4.0  # 4.0 Hz (240 bpm)
frame_rate = 30  # Assuming 30 fps

# Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=3):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Calibration function
def calibrate(intensities, frame_rate, calibration_time=10):
    print("Calibrating... Please remain still.")
    calibration_data = []
    start_time = cv2.getTickCount()

    while (cv2.getTickCount() - start_time) / cv2.getTickFrequency() < calibration_time:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # Largest face
            (x, y, w, h) = face
            roi = frame[y:y+h//4, x:x+w]  # Forehead region

            # Convert ROI to HSV and extract the V channel (intensity)
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            calibration_data.append(np.mean(roi[:, :, 2]))

    # Filter the calibration data
    calibration_signal = bandpass_filter(calibration_data, lowcut, highcut, frame_rate)
    calibration_signal = (calibration_signal - np.mean(calibration_signal)) / np.std(calibration_signal)

    # Calculate baseline heart rate
    peaks, _ = find_peaks(calibration_signal, height=0.5, distance=frame_rate//2)
    peak_times = np.array(peaks) / frame_rate

    if len(peak_times) >= 2:
        baseline_hr = 60 / np.mean(np.diff(peak_times))
        print(f"Calibration complete. Baseline heart rate: {baseline_hr:.2f} bpm")
    else:
        baseline_hr = 72.0  # Default baseline if calibration fails
        print("Calibration failed. Using default baseline heart rate.")

    return baseline_hr

# Start webcam
cap = cv2.VideoCapture(2)

# Calibrate the system
baseline_hr = calibrate(intensities, frame_rate)

# Moving average for heart rate smoothing
hr_history = deque(maxlen=10)

# Long-term baseline adjustment parameters
baseline_alpha = 0.1  # Weight for updating the baseline (smaller = slower adjustment)
min_signal_quality = 0.5  # Minimum signal quality to update baseline

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # Largest face
        (x, y, w, h) = face
        roi = frame[y:y+h//4, x:x+w]  # Forehead region

        # Convert ROI to HSV and extract the V channel (intensity)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        intensities.append(np.mean(roi[:, :, 2]))  # Append intensity value

        # Calculate heart rate if enough data is collected
        if len(intensities) == window_size:
            # Bandpass filter the signal
            filtered_signal = bandpass_filter(list(intensities), lowcut, highcut, frame_rate)

            # Normalize the signal
            normalized_signal = (filtered_signal - np.mean(filtered_signal)) / np.std(filtered_signal)

            # Calculate signal quality (e.g., amplitude of the filtered signal)
            signal_quality = np.max(normalized_signal) - np.min(normalized_signal)

            # Find peaks in the filtered signal
            peaks, _ = find_peaks(normalized_signal, height=0.5, distance=frame_rate//2)
            peak_times = np.array(peaks) / frame_rate  # Convert peak indices to time in seconds

            # Calculate heart rate if at least 2 peaks are detected
            if len(peak_times) >= 2:
                heart_rate = 60 / np.mean(np.diff(peak_times))  # Average time between peaks
                heart_rate = np.clip(heart_rate, 40, 180)  # Clip to realistic range

                # Smooth the heart rate using a moving average
                hr_history.append(heart_rate)
                smoothed_hr = np.mean(hr_history)

                # Update baseline if signal quality is high
                if signal_quality > min_signal_quality:
                    baseline_hr = baseline_alpha * smoothed_hr + (1 - baseline_alpha) * baseline_hr

                print(f'Estimated heart rate: {smoothed_hr:.2f} bpm | Baseline: {baseline_hr:.2f} bpm')
            else:
                print("Not enough peaks detected. Waiting for more data...")
    else:
        print("No face detected. Waiting for face...")

# Release the webcam
cap.release()