## VOICE CONTROLLED CAR (Python 3.5, Tensorflow LATEST, Keras LATEST)
 Youtube video can be found at [Youtube](https://www.youtube.com/watch?v=X7tyBVSACUM) <br>
 
 - Step 1: Generate New data using <b>Generate_Voice_Data.py</b> <br>
   - Before running / executing file please change <br>
   ```WAVE_OUTPUT_FILENAME = "data/right/file-right-t-"```  --> Example for keyword "RIGHT"
   - Please create required directories manually as code will not generate any directory. <br>
   
 - Step 2: Update <b>config.ini</b> with correct path and keywords. <br>
 - Step 3: Run <b>Voice-Controlled-Car.ipynb</b> to start training. <br>
 - Step 4: Save model after training. <br>
 - Step 5: Change IP to correct IP of car in <b>Test-Voice-Control-Car-Model.py</b>.
 
 ##### Please note that with current data, model doesn't produce good result. 
