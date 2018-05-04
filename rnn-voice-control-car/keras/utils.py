from pydub import AudioSegment
from configparser import ConfigParser
import os
import numpy as np
import random
import sys
import IPython
from scipy.io import wavfile 
import matplotlib
import matplotlib.pyplot as plt
import tempfile

'''
# Data downloaded from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

Copyright note:

APA-style citation: "Warden P. Speech Commands: A public dataset for single-word speech recognition, 2017. Available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz".

BibTeX @article{speechcommands, title={Speech Commands: A public dataset for single-word speech recognition.}, author={Warden, Pete}, journal={Dataset available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz}, year={2017} }
'''

fileIdx = 0

# instantiate
config = ConfigParser()

# parse existing file
config.read('config.ini')

root = config.get('section_path', 'data_path')
data = config.get('section_path', 'training_data')

# path to noise data - 3 seconds each generated by 'gen_noise_data.py'
noise = data + config.get('section_path', 'out_noise_path')

inPath = root + config.get('section_path', 'left')
outPath = data + config.get('section_path', 'left')

Ty = 1375


# From Audio recordings to Spectrograms
def graph_spectrogram(data):
    nfft = 200          # fft window length
    fs = 8000           # sampling frequency
    noverlap = 120      # overlap between windows
    nchannels = data.ndim
    
    if nchannels == 1:
        pxx, freq, bins, im = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, freq, bins, im = plt.specgram(data[:,0], nfft, fs, noverlap=noverlap)
    
    # print('file duration in seconds %s' %(len(data)/rate))
    return pxx    
    
# find random time segment
def get_random_time_segment(seg_ms):
    seg_start = np.random.randint(low=0, high=10000-seg_ms)    # low = 0 sec, high = 3 seconds
    seg_end = seg_start + seg_ms - 1
    #print('end' + str(seg_start))
    return seg_start, seg_end



def is_overlapping(seg_time, prev_segs):
    
    seg_start, seg_end = seg_time
    
    overlap = False
    
    for prev_start, prev_end in prev_segs:
        if seg_start <= prev_end and seg_end >= prev_start:
            overlap = True
            
    return overlap

# method to overlay
def insert_audio_clip(background, audio_clip, prev_segs):
    """
    background     10 sec background noise audio
    audio_clip     clip with positive or negative sample
    prev_seg       time where autdio sample has already placed
    
    return         updated 10 seconds audio
    """
    
    # get time in ms
    seg_ms = len(audio_clip)
    
    # get random place time for this clip
    seg_time = get_random_time_segment(seg_ms)
    
    # check if seg_time overlaps to previous seg, if yes then find another place
    while is_overlapping(seg_time, prev_segs):
        seg_time = get_random_time_segment(seg_ms)
    
    # append new seg to prev seg
    prev_segs.append(seg_time)
    
    # lets overlay
    new_background = background.overlay(audio_clip, position = seg_time[0])
    
    return new_background, seg_time
    
# generate Y by inserting 1 at place where keyword 'left' is detected
def insert_ones(y, seg_end_ms):
    
    seg_end_y = int(seg_end_ms * Ty / 10000.0)
    
    for i in range(seg_end_y + 1, seg_end_y + 51): # 50 1s once keyword detected
        if i < Ty:
            y[0, i] = 1
            
    return y

# amplify it
def match_target_amp(sound, target_dBFS):
    dBFS_Change = target_dBFS - sound.dBFS
    return sound.apply_gain(dBFS_Change)

# create training samples 
def create_training_sample(background, positive, negative, idx):
    
    """
    background      path 3 second audio
    positive        path positive sample audio  -> with left
    
    returns:
    x               spectrogram of training sample
    y               label at each time step of the spectrogram
    """
    
    
    # Make backgound quieter
    background = background - 20
    
    # initialize ouput vector as zeros
    y = np.zeros((1, Ty))
    
    # initialize seg times as empty list
    prev_seg = []
    
    # select 0-4 random positive samples
    nr_of_positve = np.random.randint(0, 5)
    random_indx = np.random.randint(len(positive), size=nr_of_positve)
    random_positives = [positive[i] for i in random_indx]
    
    # lets overlay this sample over background
    for random_positive in random_positives:
        # insert positive sample on background
        background, seg_time = insert_audio_clip(background, random_positive, prev_seg)
        
        # retreive seg_start and seg_end from seg_time
        seg_start, seg_end = seg_time
        
        # insert ones in y
        y = insert_ones(y, seg_end)
        
    # select 0-2 random negative samples
    nr_of_negative = np.random.randint(0, 3)
    random_idx = np.random.randint(len(negative), size=nr_of_negative)
    random_negatives = [negative[i] for i in random_idx]
    
    # lets overlay this sample over background
    for random_negative in random_negatives:
        # insert negative sample on backgound
        background, _ = insert_audio_clip(background, random_negative, prev_seg)
        
    
    # standarize the volume of the audio clip
    background = match_target_amp(background, -20.0)
    
    # reshape background
    background = np.asarray(background.get_array_of_samples())
    background = background.reshape(int(len(background)/2), 2)
    
    # get spectrogram
    x = graph_spectrogram(background)
        
    return x, y