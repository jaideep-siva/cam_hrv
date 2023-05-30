# RealTimeHeartRateMonitor

RealTimeHeartRateMonitor is a Python application that demonstrates real-time heart rate monitoring using a webcam.

## Introduction

This application accesses the user's webcam to capture video, detects the user's face, and measures the color fluctuation in the facial region to estimate the user's heart rate. The heart rate is then plotted in real-time.

## Technologies Used

* Python
* OpenCV
* NumPy
* SciPy
* scikit-learn
* matplotlib

## Installation and Setup

1. Clone this repository.
2. Install the required Python packages:

    ```shell
    pip install opencv-python-headless numpy scipy sklearn matplotlib
    ```

3. Run the script:

    ```shell
    python HR.py
    ```

## Usage

Ensure that you are in a well-lit environment and that your face is clearly visible to the webcam. The script will output an estimate of your heart rate in beats per minute and plot the estimated heart rate in real time.

## Limitations

This is a demonstration application and is not intended for medical use. It uses a simplified method of remote photoplethysmography (rPPG), and there are many factors (including lighting conditions, movement, and webcam quality) that can affect the accuracy of the heart rate measurement.

## Author

Mason Kadem

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
