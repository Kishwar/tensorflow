## VGG19-STYLE-TRANSFER FOR IMAGE (Python 3.5, Tensorflow LATEST)
   - Download model weights from http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat <br />
   - VGG19 STYLE TRANSFER can be executed by - <br />
```
python style-transfer-main.py  --style ./imgstyle/StyleImage.jpg --content ./imgcontent/ContentImage.jpg --out ./output/ --epochs 100000 --print-iterations 100 --learning-rate 10
```
   - VGG19 Style Transfer can be also executed by <b> vgg19-style-transfer/Style_Transfer.ipynb </b> IPYTHONG NOTEBOOK
      * Please change values for <b> --epochs </b> and <b> --print-iterations </b> as per your requirement.

<p align = 'center'>
<img src = 'imgstyle/StyleImageRain.jpg' height = '246px'>
<img src = 'imgcontent/ContentImage.jpg' height = '246px'>
<a href = 'output/out.jpg'><img src = 'output/out.jpg' width = '627px'></a>
</p>
