Training CNN (Pytorch) and Deployment on the WWW (ONNX.js) on Windows (CPU)


### Requirements
- Windows (or Linux)
- CPU processor (or GPU, CUDA)


### Supported CNN
- MobileNet v2
- EfficientNet Lite


### How to
- Download the code (ZIP)
    ![img](https://github.com/whria78/data-in-paper-out/blob/main/screenshot/00.PNG?raw=true)
- Get Anaconda version 3.8 or newer (https://www.anaconda.com/products/distribution)

	![img](https://github.com/whria78/modelderm_rcnn_api/raw/master/img/download_anaconda.PNG)

	![img](https://github.com/whria78/modelderm_rcnn_api/raw/master/img/ana1.PNG)

	Please be sure to add the system PATH. 
	
	![img](https://github.com/whria78/modelderm_rcnn_api/raw/master/img/ana2.PNG)

- Check the pretrained DEMO

    ![img](https://github.com/whria78/data-in-paper-out/blob/main/screenshot/00-1.PNG?raw=true)

    Run "0. requirement.bat" to install the required libraries.

    Run "2. run_demo_server.bat", and then "3. connect_demo_server.bat", to check the default DEMO.

- Train a custom model and try the DEMO

    Run "1. train_mobilenet.bat" if you want to train a CNN (MobileNet).

    Run "2. run_demo_server.bat", and then "3. connect_demo_server.bat", to check the customized DEMO.

## Screenshot

The default DEMO shows the result of ensemble of efficientnet (/demo/model_eff_30e_0.onnx) and mobilenet (/demo/model_mob_30e_0.onnx). If you want to change or add more models, please check (/demo/dxinfo.js)

![img](https://github.com/whria78/data-in-paper-out/blob/main/screenshot/1.JPG?raw=true)
![img](https://github.com/whria78/data-in-paper-out/blob/main/screenshot/2.JPG?raw=true)

> [LINUX] python train.py --model efficientnet --epoch 30 --step 10 --lr 0.005

![img](https://github.com/whria78/data-in-paper-out/blob/main/screenshot/5.PNG?raw=true)

> [LINUX] python demo.py

> [LINUX] xdg-open http://127.0.0.1:8000

![img](https://github.com/whria78/data-in-paper-out/blob/main/screenshot/6.PNG?raw=true)


## Dataset

Onychomycosis dataset came from the following paper.

Deep neural networks show an equivalent and often superior performance to dermatologists in onychomycosis diagnosis: Automatic construction of onychomycosis datasets by region-based convolutional deep neural network, PLOS One 2018

https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0191493
https://figshare.com/articles/dataset/Model_Onychomycosis_Training_Datasets_JPG_thumbnails_and_Validation_Datasets_JPG_images_/5398573?file=9302506