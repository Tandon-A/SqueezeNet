# SqueezeNet

Tensorflow implementation of SqueezeNet following this [paper](https://arxiv.org/abs/1602.07360). Referenced Caffe code can be found [here](https://github.com/DeepScale/SqueezeNet).

<img src="https://raw.githubusercontent.com/Tandon-A/SqueezeNet/master/assets/Model.PNG" height="400" alt="SqueezeNet Model Definition ">


## Prerequisites

* Python 3.3+
* Tensorflow 
* pillow (PIL)
* (Optional) [Tiny ImageNet Database](https://tiny-imagenet.herokuapp.com/): Tiny ImageNet Database (200 classes)

## Implemented Models 

### SqueezeNet V0

* Use of 8 Fire Modules. 
* Starting convolution filter of kernel size = 7 and output_filters = 96
* Pool Layers used after conv1, fire4 and fire8. 

### SqueezeNet V0 Residual (Using Bypass Connections) 

* Uses bypass connections between layers. 
* Bypass connections between fire2 and fire4, fire6 and pool2, fire8 and fire6, cxonv10 and pool3.

### SqueezeNet V1 

* Model defined in [official repository](https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1)
* Starting convolution filter of kernel size = 3 and output filters = 64
* Pool layers used after conv1, fire3 and fire5. 


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/Tandon-A/Image-Editing-using-GAN/blob/master/LICENSE) file for details

## Author 

Abhishek Tandon/ [@Tandon-A](https://github.com/Tandon-A)
