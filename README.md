# tensorflow (Python 2.7 / 3.5)

This repo contains my tensorflow codes

## MNIST  <br />
   - mnist training - python mnist_train.py <br />
  Required training and test/validation data is automatically downloaded by code. <br />
   - mnist test - python mnist_test.py <br />
  Image input as arg (not yet implemented, arg for input image, change in code for now)

## EMOTION <br />
   - emotion training - python emo_train.py <br />
 Required training and test/validation data can be downloaded from  <br />
      * https://inclass.kaggle.com/c/facial-keypoints-detector/data --> ./data/
  Image input as arg (not yet implemented, arg for input image, change in code for now)
  
## VGG19-STYLE-TRANSFER (Python 3.5, Tensorflow LATEST)
   - Download model weights from http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat <br />
   - VGG19 STYLE TRANSFER can be executed by - <br />
```
python style-transfer-main.py  --style ./imgstyle/StyleImage.jpg --content ./imgcontent/ContentImage.jpg --out ./output/ --epochs 100000 --print-iterations 100 --learning-rate 0.01 
```
   - VGG19 Style Transfer can be also executed by <b> vgg19-style-transfer/Style_Transfer.ipynb </b> IPYTHONG NOTEBOOK
      * Please change values for <b> --epochs </b> and <b> --print-iterations </b> as per your requirement.
