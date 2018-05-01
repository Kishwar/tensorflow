from pydub import AudioSegment

'''
# Data downloaded from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz

Copyright note:

APA-style citation: "Warden P. Speech Commands: A public dataset for single-word speech recognition, 2017. Available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz".

BibTeX @article{speechcommands, title={Speech Commands: A public dataset for single-word speech recognition.}, author={Warden, Pete}, journal={Dataset available from http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz}, year={2017} }
'''

fileIdx = 0
inPath = r"D:\rnn-voice-control-car-data\training_data\_background_noise_"
outPath = r"D:\rnn-voice-control-car-data\training_data\_background_noise_\noise"

# from doing_the_dishes.wav
noise = AudioSegment.from_wav(inPath + r"\doing_the_dishes.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export(outPath + "\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from dude_miaowing.wav
noise = AudioSegment.from_wav(inPath + r"\dude_miaowing.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export(outPath + "\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1

# from exercise_bike.wav
noise = AudioSegment.from_wav(inPath + r"\exercise_bike.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export(outPath + "\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from pink_noise.wav
noise = AudioSegment.from_wav(inPath + r"\pink_noise.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export(outPath + "\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from running_tap.wav
noise = AudioSegment.from_wav(inPath + r"\running_tap.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export(outPath + "\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from white_noise.wav
noise = AudioSegment.from_wav(inPath + r"\white_noise.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export(outPath + "\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1