# mechanical_mirror
The Pi 4 is responsible for both processing live video and outputting to servos

## Running servos
`servo_test.py` uses micropython to control a series of servos using an I2C connection. [Blinka](https://learn.adafruit.com/circuitpython-on-raspberrypi-linux/installing-circuitpython-on-raspberry-pi) is required to run micropython on the Pi. Before running the servos, test that `blinkatest.py` runs properly as shown in the documentation. 
```console
  sudo pip3 install adafruit-circuitpython-pca9685
  sudo pip3 install adafruit-circuitpython-motor
  ```

## MediaPipe 
[Documentation](https://developers.google.com/mediapipe/framework/getting_started/install)
* Install `pip3 install mediapipe`


## OpenCV
OpenCV is installed via mediapipe by running the `mediapipe/setup_opencv.sh` file. 
  ```console
  chmod +x setup_opencv.sh
  ./setup_opencv.sh
  ```

## Useful Commands
Check i2c connections; `i2cdetect -y 1`
Check USB devices: `dmesg | grep tty`

Controlling a background process: 
* `systemctl status mirror_station.service`
* `sudo systemctl enable --now mirror_station.service`
* creating background file: `sudo systemctl --force --full edit mirror_station.service` 









