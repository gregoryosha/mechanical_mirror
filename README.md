# mechanical_mirror
Connecting: 
home ip: 192.168.86.184
School ip: 10.245.149.172





Installing Dependencies on raspberry Pi:

Found here: https://dydx.me/2021-04-07/kinect-on-raspberry-pi
ran commands:
sudo apt-get install git-core cmake freeglut3-dev pkg-config build-essential libxmu-dev libxi-dev libusb-1.0-0-dev

Main freenect page: https://github.com/OpenKinect/libfreenect2

To search for packages: apt-cache search keyword

___________________________Freenect Dependencies___________________
Installed libusb:
sudo apt-get install libusb-dev

Installed GLFW:
sudo apt-get install libglfw3
sudo apt-get install libglfw3-dev

Installed turbo-jpeg:
sudo apt-get install libturbojpeg0-dev
sudo apt-get install libturbojpeg0

OpenNi2:
sudo apt-get install libopenni2-0
sudo apt-get install libopenni2-dev

Install cmake:
sudo apt-get install cmake 

*note: raspberry pis don't support openGL 3, so special cmake command for going to 2 is required
*Find this command on the freenect2 github page under different operating system 

_________________________Open CV_______________
sudo apt-get install build-essential cmake pkg-config libjpeg-dev libtiff5-dev libjasper-dev libpng-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libfontconfig1-dev libcairo2-dev libgdk-pixbuf2.0-dev libpango1.0-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran libhdf5-dev libhdf5-serial-dev libhdf5-103 python3-pyqt5 python3-dev -y

sudo apt-get install python3-opencv

___________________Running Software____________






