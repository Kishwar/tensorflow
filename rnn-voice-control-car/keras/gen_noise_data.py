from pydub import AudioSegment
from configparser import ConfigParser
import os

'''
# Data downloaded from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

Copyright note:

APA-style citation: "Warden P. Speech Commands: A public dataset for single-word speech recognition, 2017. Available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz".

BibTeX @article{speechcommands, title={Speech Commands: A public dataset for single-word speech recognition.}, author={Warden, Pete}, journal={Dataset available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz}, year={2017} }
'''

def generate_noise_files():
    fileIdx = 0
    
    outAudioLen = 10 * 1000
    
    # instantiate
    config = ConfigParser()
    
    # parse existing file
    config.read('config.ini')
    
    root = config.get('section_path', 'data_path')
    data = config.get('section_path', 'training_data')
    
    inPath = root + config.get('section_path', 'in_noise_path')
    outPath = data + config.get('section_path', 'out_noise_path')
    
    if not os.path.exists(outPath):
        os.makedirs(outPath)
        
    for file in os.listdir(inPath):
        noise = AudioSegment.from_wav(inPath + file)
        ln = len(noise)
        for i in range(int(ln/outAudioLen)):
            gnoise = noise[(i*outAudioLen):((i+1)*outAudioLen)]
            gnoise.export(outPath + "gnoise-" + str(fileIdx) + ".wav", format="wav")
            fileIdx += 1