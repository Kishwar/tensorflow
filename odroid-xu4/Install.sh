#!/bin/bash

if [[ "$1" == "1" ]]; then
  # Install global dependencies
  sudo apt-get install libhdf5-serial-dev
  sudo apt-get install libhdf5-dev
  sudo apt-get install openmpi-bin libopenmpi-dev
  sudo apt-get install libblas3 liblapack3 liblapack-dev libblas-dev
  sudo apt-get install gfortran
  sudo apt-get -qq remove ffmpeg x264 libx264-dev
  sudo apt-get -qq install libopencv-dev build-essential checkinstall 
  sudo apt-get -qq install cmake pkg-config yasm libjpeg-dev libjasper-dev 
  sudo apt-get -qq install libavcodec-dev libavformat-dev libswscale-dev 
  sudo apt-get -qq install libdc1394-22-dev libxine2-dev libgstreamer0.10-dev 
  sudo apt-get -qq install libgstreamer-plugins-base0.10-dev libv4l-dev python-dev 
  sudo apt-get -qq install python-numpy libtbb-dev libqt4-dev libgtk2.0-dev 
  sudo apt-get -qq install libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev 
  sudo apt-get -qq install libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils 
  sudo apt-get -qq install ffmpeg cmake qt5-default checkinstall

  # Update enviroment path
  export CPATH="/usr/include/hdf5/serial/"

  # Install Miniconda
  wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh
  sudo chmod +x Miniconda3-latest-Linux-armv7l.sh
  sudo ./Miniconda3-latest-Linux-armv7l.sh
  echo "Please reboot system using 'sudo reboot'"
fi

if [[ "$1" == "2" ]]; then
  # Create conda enviroment
  conda create -n py34Env python=3.4

  # Activate enviroment
  source activate py34Env
  
  # Deactivate enviroment
  source deactivate
fi

if [[ "$1" == "3" ]]; then

  sudo rm tensorflow-1.11.0-cp3*
  
  # Download tensorflow
  wget https://www.piwheels.org/simple/tensorflow/tensorflow-1.11.0-cp34-none-linux_armv7l.whl
  
  # Activate enviroment
  source activate py34Env

  # Install local dependencies
  export CPATH="/usr/include/hdf5/serial/"
  python3.4 -m pip install keras_applications==1.0.6 --no-deps
  python3.4 -m pip install keras_preprocessing==1.0.5 --no-deps
  python3.4 -m pip install h5py==2.8.0

  # Install tensorflow
  python3.4 -m pip install tensorflow-1.11.0-cp34-none-linux_armv7l.whl
  
  # Deactivate enviroment
  source deactivate
fi

if [[ "$1" == "4" ]]; then

  # Activate enviroment
  source activate py34Env
  
  python3.4 -m pip install numpy
  python3.4 -m pip install -U numpy

  # Download Opencv
  version="3.4.3"
  mkdir OpenCV
  cd OpenCV
  wget -O OpenCV-$version.zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/$version/opencv-"$version".zip/download
  unzip OpenCV-$version.zip
  cd opencv-$version

  # Build OpenCV
  mkdir build
  cd build
  cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON \
        -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON \
        -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=ON \
        -D PYTHON3_LIBRARY=/home/odroid/.conda/envs/py34Env/lib/libpython3.4m.so \
        -D PYTHON3_INCLUDE_DIR=/home/odroid/.conda/envs/py34Env/include/python3.4m ..

  make -j8

  # Install it
  sudo make install
  sudo ldconfig
  cd /usr/local/lib/python3.4/site-packages
  sudo mv cv2.cpython-34m.so cv2.so

  cd /home/odroid/.conda/envs/py34Env/lib/python3.4/site-packages
  ln -s /usr/local/lib/python3.4/site-packages/cv2.so cv2.so

  cd ~
  
  # Deactivate enviroment
  source deactivate
fi

echo 'Job finished.'
