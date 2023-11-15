# mechanical_mirror
The Pi 4 is responsible for both processing live video and outputting to servos

## Running servos
`servo_test.py` uses micropython to control a series of servos using an I2C connection. [Blinka](https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi) is required to run micropython on the Pi. Before running the servos, test that `blinkatest.py` runs properly as shown in the documentation. 

## MediaPipe 
[Documentation](https://developers.google.com/mediapipe/framework/getting_started/install)
* Install python six library: `pip3 install --user six`
* Install Go: `sudo apt install golang-go`
* Install Bazelisk: `go get github.com/bazelbuild/bazelisk`. Move it to bin: `sudo mv ./go/bin/bazelisk /usr/bin/bazel`
* Install mediapipe: `git clone https://github.com/google/mediapipe.git`

a

## OpenCV
OpenCV is installed via mediapipe by running the `mediapipe/setup_opencv.sh` file. 
  ```console
  chmod +x setup_opencv.sh
  ./setup_opencv.sh
  ```










