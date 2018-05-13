import pyaudio
import librosa
import numpy as np 
import wave


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 2 
WAVE_OUTPUT_FILENAME = "data/right/file-right-t-"

k = 0

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
 
    waveFile = wave.open(WAVE_OUTPUT_FILENAME + str(k) + '-.wav', 'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames[j:j+32]))
    waveFile.close()
    
    k += 1