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
 ## Docker Instructions

You can also run this application in a Docker container. To do so, follow these steps:

1. Ensure Docker is installed on your system. If not, you can download and install it from the [official Docker website](https://www.docker.com/products/docker-desktop).

2. Clone this repository to your local system:

    ```shell
    git clone https://github.com/masonkadem/RealTimeHeartRateMonitor.git
    cd RealTimeHeartRateMonitor
    ```

3. Build the Docker image:

    ```shell
    docker build -t real-time-heart-rate-monitor .
    ```

4. Run the Docker container. If you are on a Linux system, you can use the `--device` flag to give Docker access to your webcam. Replace `/dev/video0` with the appropriate device path for your system:

    ```shell
    docker run --device /dev/video0 -p 4000:80 real-time-heart-rate-monitor
    ```

**Note**: Docker may not have access to your webcam, which is required for this script to function. Docker can access hardware devices like the webcam on Linux systems, but it's generally not possible on macOS and Windows. For Windows or macOS, it may be more feasible to run the application directly on your host system or inside a virtual machine that has access to your webcam.
   
 

## Usage

Ensure that you are in a well-lit environment and that your face is clearly visible to the webcam. The script will output an estimate of your heart rate in beats per minute and plot the estimated heart rate in real time.

## Limitations

This is a demonstration application and is not intended for medical use. It uses a simplified method of remote photoplethysmography (rPPG), and there are many factors (including lighting conditions, movement, and webcam quality) that can affect the accuracy of the heart rate measurement.

## Author

Mason Kadem

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
