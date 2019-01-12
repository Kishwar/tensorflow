{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf200
{\fonttbl\f0\fnil\fcharset0 Menlo-Regular;}
{\colortbl;\red255\green255\blue255;\red27\green31\blue34;\red244\green246\blue249;}
{\*\expandedcolortbl;;\cssrgb\c14118\c16078\c18039;\cssrgb\c96471\c97255\c98039;}
\paperw11900\paperh16840\margl1440\margr1440\vieww15940\viewh8400\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs27\fsmilli13600 \cf2 \cb3 \expnd0\expndtw0\kerning0
sudo apt-get install libhdf5-serial-dev\
sudo apt-get install libhdf5-dev\
export CPATH="/usr/include/hdf5/serial/"\
sudo pip3 install keras_applications==1.0.6 --no-deps\
sudo pip3 install keras_preprocessing==1.0.5 --no-deps\
sudo pip3 install h5py==2.8.0\
sudo apt-get install -y openmpi-bin libopenmpi-dev\
sudo apt install git\
sudo apt-get install libblas3 liblapack3 liblapack-dev libblas-dev\
sudo apt-get install gfortran\
\
wget https://www.piwheels.org/simple/tensorflow/tensorflow-1.11.0-cp34-none-linux_armv7l.whl\
wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-armv7l.sh\
\
sudo chmod +x Miniconda3-latest-Linux-armv7l.sh\
sudo ./Miniconda3-latest-Linux-armv7l.sh\
sudo ldconfig\
\
conda create -n py34Env python=3.4\
source activate py34Env\
\
sudo apt-get -qq remove ffmpeg x264 libx264-dev\
sudo apt-get -qq install libopencv-dev build-essential checkinstall cmake pkg-config yasm libjpeg-dev libjasper-dev libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev python-dev python-numpy libtbb-dev libqt4-dev libgtk2.0-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev libvorbis-dev libxvidcore-dev x264 v4l-utils ffmpeg cmake qt5-default checkinstall\
\
python3.4 -m pip install numpy\
python3.4 -m pip install -U numpy\
\
version="3.4.3"\
mkdir OpenCV\
cd OpenCV\
wget -O OpenCV-$version.zip http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/$version/opencv-"$version".zip/download\
unzip OpenCV-$version.zip\
cd opencv-$version\
\
mkdir build\
cd build\
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local \\\
      -D WITH_TBB=ON -D BUILD_NEW_PYTHON_SUPPORT=ON -D WITH_V4L=ON \\\
      -D INSTALL_C_EXAMPLES=ON -D INSTALL_PYTHON_EXAMPLES=ON \\\
      -D BUILD_EXAMPLES=ON -D WITH_QT=OFF -D WITH_OPENGL=ON \\\
      -D PYTHON3_LIBRARY=/home/odroid/.conda/envs/py34Env/lib/libpython3.4m.so \\\
      -D PYTHON3_INCLUDE_DIR=/home/odroid/.conda/envs/py34Env/include/python3.4m ..\
\
make -j8\
\
sudo make install\
sudo ldconfig\
\
cd /usr/local/lib/python3.4/site-packages\
sudo mv cv2.cpython-34m.so cv2.so\
\
cd /home/odroid/.conda/envs/py34Env/lib/python3.4/site-packages\
ln -s /usr/local/lib/python3.4/site-packages/cv2.so cv2.so\
\
cd ~\
\
\
}