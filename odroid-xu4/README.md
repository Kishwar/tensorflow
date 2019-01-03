
# Tensorflow on Odroid XU4

### To build Tensorflow on Odroid XU4, we will need following hardware.

- Odroid XU4 board
- SD Card / eMMC
- Network cable
- USB (Swap memory)
- Monitor (Optional)
- Keyboard and Mouse (Optional)

<p align = 'center'>
<img src = 'images/Hardware.jpeg' height = '320px'>
</p>

### Step 1: Format SD Card
We need to format SD card because without formatting it, we may not get all available space for OS. I am using software available on https://www.sdcard.org/downloads/formatter_4/eula_mac/
<p align = 'center'>
<img src = 'images/SDCardFormatter.png' height = '320px'>
</p>

### Step 2: Download OS for Odroid XU4
We can download OS from https://wiki.odroid.com/odroid-c1/os_images/ubuntu/v3.0 (https://east.us.odroid.in/ubuntu_18.04lts/ubuntu-18.04-4.14-mate-odroid-xu4-20180501.img.xz)

### Step 3: Load OS on SD Card
We will use <b>Balena Etcher</b> to load OS image on SD Card. Balena Etcher is available on http://etcher.io
<p align = 'center'>
<img src = 'images/loadImage.png' height = '320px'>
</p>

### Step 4: Let's get into the Odroid using SSH
<p align = 'center'>
<img src = 'images/ssh.png' height = '320px'>
</p>

### Step 5: Execute some basic commands required for Odroid XU4
```
sudo apt install libnfs11
sudo apt update
sudo apt upgrade
sudo apt dist-upgrade
sudo reboot
```

### Step 6.1: Preprocessing for Bazel (Build tool for Tensorflow)
Before we build Bazel, run following commands
```
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip default-jdk autoconf automake libtool

# For Python 2.7
sudo apt-get install python-pip python-numpy swig python-dev
sudo pip install wheel

# For Python 3.3+
sudo apt-get install python3-pip python3-numpy swig python3-dev
sudo pip3 install wheel
```
We will use gcc and g++ version 4.8 to build bazel
```
sudo apt-get install gcc-4.8 g++-4.8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 100
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-4.8 100
```
Check gcc and g++ versions before proceeding further.
```
odroid@odroid:~$ gcc --version
gcc (Ubuntu/Linaro 4.8.5-4ubuntu8) 4.8.5
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

odroid@odroid:~$ g++ --version
g++ (Ubuntu/Linaro 4.8.5-4ubuntu8) 4.8.5
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

### Step 6.2: Adding swap memory
Install htop to check swap memory attached with Odroid
```
sudo apt install htop
```
First, put insert your USB drive, and find the /dev/XXX path for the device.
```
sudo blkid
```
As an example, my drive's path was /dev/sda1

Flag your device to be swap:
```
sudo mkswap /dev/XXX
```
Copy the UUID associated with /dev/XXX
Now edit your /etc/fstab file to register your swap file.
```
sudo nano /etc/fstab
```
On a separate line, enter the following information. Replace the X's with the UUID (without quotes)
```
UUID=XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX none swap sw,pri=5 0 0
```
Save /etc/fstab, exit your text editor, and run the following command:
```
sudo swapon -a
```
Ok, we have added swap memory. To check run <b>htop</b>. In <b>Swp</b>, you should now have some memory size.

### Step 6.2: Get Bazel
```
wget https://github.com/bazelbuild/bazel/releases/download/0.15.0/bazel-0.15.0-dist.zip
unzip -d bazel bazel-0.15.0-dist.zip
```
