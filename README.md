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
  
## VGG19-STYLE-TRANSFER FOR IMAGE (Python 3.5, Tensorflow LATEST)
   - Download model weights from http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat <br />
   - VGG19 STYLE TRANSFER can be executed by - <br />
```
python style-transfer-main.py  --style ./imgstyle/StyleImage.jpg --content ./imgcontent/ContentImage.jpg --out ./output/ --epochs 100000 --print-iterations 100 --learning-rate 10
```
   - VGG19 Style Transfer can be also executed by <b> vgg19-style-transfer/Style_Transfer.ipynb </b> IPYTHONG NOTEBOOK
      * Please change values for <b> --epochs </b> and <b> --print-iterations </b> as per your requirement.

<p align = 'center'>
<img src = 'vgg19-style-transfer/imgstyle/StyleImageRain.jpg' height = '246px'>
<img src = 'vgg19-style-transfer/imgcontent/ContentImage.jpg' height = '246px'>
<a href = 'vgg19-style-transfer/output/out.jpg'><img src = 'vgg19-style-transfer/output/out.jpg' width = '627px'></a>
</p>

## VGG19-STYLE-TRANSFER FOR VIDEO (Python 3.5, Tensorflow LATEST) - Expected to be finished 30/04/2018
   - Currently working...
      * https://arxiv.org/abs/1708.04538
```
python style-transfer-video-main.py --style ./imgstyle/StyleImageRain.jpg --content ./train2014/ --epochs 100 --print-iterations 500 --learning-rate 10 --chkpnt ./chkpnt/
```     

## STYLE-TRANSFER FOR MUSIC (Python 3.5, Tensorflow LATEST) - Expected to be finished 30/06/2018
   - NEXT IN QUEUE..

## STYLE-TRANSFER FOR AUDIO - VOICE CLONING (Python 3.5, Tensorflow LATEST) - Expected to be finished 30/11/2018
   - IN QUEUE
   
