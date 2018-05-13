from configparser import ConfigParser
from tqdm import tqdm
import numpy as np
import librosa
import os
from socket import *      # Import necessary modules
from sklearn.model_selection import train_test_split
from keras.losses import categorical_crossentropy
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Conv1D
from keras.layers import BatchNormalization, Reshape, LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from scipy.io.wavfile import write
import time
from pydub import AudioSegment
import time

HOST = '192.168.0.7'    # Server(Raspberry Pi) IP address
PORT = 21567
BUFSIZ = 1024             # buffer size
ADDR = (HOST, PORT)

tcpCliSock = socket(AF_INET, SOCK_STREAM)   # Create a socket
tcpCliSock.connect(ADDR)                    # Connect with the server

tcpCliSock.send('speed50'.encode('utf-8'))

# =============================================================================
# The function is to send the command forward to the server, so as to make the 
# car move forward.
# ============================================================================= 
def run(event):
	print('motor ', event)
	if event == 'stop':
		tcpCliSock.send('stop'.encode('utf-8'))
	elif event == 'go':
		tcpCliSock.send('forward'.encode('utf-8'))
	elif event == 'left':
		tcpCliSock.send('left'.encode('utf-8'))
		time.sleep(1)
		tcpCliSock.send('home'.encode('utf-8'))
	elif event == 'right':
		tcpCliSock.send('right'.encode('utf-8'))
		time.sleep(1)
		tcpCliSock.send('home'.encode('utf-8'))

# instantiate
config = ConfigParser()
    
# parse existing file
config.read('config.ini')

in_data = config.get('section_path', 'in_path')
out_data = config.get('section_path', 'out_path')

Keys = config.get('section_keys', 'Keys')
Keys = Keys.split()

Keys_Indx = np.arange(0, len(Keys))

# Lets create X, Y for Test and X, Y for validation
for Idx, Key in enumerate(Keys):
    print(Idx, Key)
    if(Idx == 0):
        X = np.load(out_data + Keys[Idx] + '.npy')
        Y = np.zeros(X.shape[Idx])
    else:
        x = np.load(out_data + Keys[Idx] + '.npy')
        X = np.vstack((X, x))
        Y = np.append(Y, np.full(x.shape[0], fill_value = Idx))
        
    assert X.shape[0] == len(Y)
    
XT, XV, YT, YV = train_test_split(X, Y, test_size= (1 - 0.6), random_state=42, shuffle=True)


def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    model = Sequential()
    
    model.add(LSTM(units=128, input_shape=input_shape, return_sequences = True))
    
    model.add(LSTM(units=128, return_sequences = True))
    
    model.add(LSTM(units=20, return_sequences = False))
    
    model.add(Dense(units=len(Keys), activation='softmax'))
    
    return model
    
model = model(input_shape = XT.shape[1:])

model.load_weights('Car_5Keys_weights.h5')

import pyaudio
from queue import Queue
from threading import Thread
import sys
import time
import wave
from scipy.io import wavfile

chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 44100 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.


THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 44100

def wav2mfcc(file_path, mfcclen=65):
    # read file
    wave, sr = librosa.load(file_path, mono=True, sr=None)
    
    # calculate power spectrum of a sound
    mfcc = librosa.feature.mfcc(wave, sr=16000)
    
    print(mfcc.shape)
    
    # pad if required to get 32 len
    if (mfcclen > mfcc.shape[1]):
        pad_width = mfcclen - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:,:mfcclen]
    return mfcc


import pyaudio
import wave
import numpy as np
 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2 
WAVE_OUTPUT_FILENAME = "file.wav"

while(True):
    audio = pyaudio.PyAudio()
 
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                frames_per_buffer=CHUNK)
    print("recording...")
    frames = []
    npframes = []
 		
    while(True):
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
            npframes.append(np.fromstring(data, dtype=np.int16))
            
        if(np.abs(npframes).mean() < 100):
            frames = []
            npframes = []
            continue
        else:
            for j in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
                if(np.abs(npframes[j:j+16]).mean() < 500):
                	j += 16
                	continue
                else:
                	print(j, j * int(RATE / CHUNK * RECORD_SECONDS), i*int(RATE / CHUNK * RECORD_SECONDS), np.abs(npframes[j:j+16]).mean())
                	break
            break
                
    print("finished recording")
 
    # stop Recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
 
    waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames[j:j+32]))
    waveFile.close()
        
    background = AudioSegment.from_wav(WAVE_OUTPUT_FILENAME)
    background.export(r"fileA.wav", format="wav")
        
    mfcc = wav2mfcc(r"fileA.wav")
    spectrum = np.expand_dims(mfcc, axis=0)
    print(np.max(model.predict(spectrum)))
    print(np.argmax(model.predict(spectrum)))
    run(Keys[np.argmax(model.predict(spectrum))])
        
        
    time.sleep(2)
