from pydub import AudioSegment

fileIdx = 0

# from doing_the_dishes.wav
noise = AudioSegment.from_wav("doing_the_dishes.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export("noise\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from dude_miaowing.wav
noise = AudioSegment.from_wav("dude_miaowing.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export("noise\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1

# from exercise_bike.wav
noise = AudioSegment.from_wav("exercise_bike.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export("noise\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from pink_noise.wav
noise = AudioSegment.from_wav("pink_noise.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export("noise\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from running_tap.wav
noise = AudioSegment.from_wav("running_tap.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export("noise\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1
    
# from white_noise.wav
noise = AudioSegment.from_wav("white_noise.wav")
ln = len(noise)
for i in range(int(ln/3000)):
    gnoise = noise[(i*3000):((i+1)*3000)]
    gnoise.export("noise\gnoise-" + str(fileIdx) + ".wav")
    fileIdx += 1