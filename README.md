# Feature Fusion Via Dual-Resolution Compressive Measurement Matrix Analysis For Spectral Image Classification

[Juan Marcos Ramírez](https://juanmarcosramirez.github.io/ "Juan's Website"), José Ignacio Martínez Torre, [Henry Arguello](http://hdspgroup.com/ "HDSP's Homepage")

## Abstract

In the compressive spectral imaging (CSI) framework, different architectures have been proposed to recover high-resolution spectral images from compressive measurements. Since CSI architectures compactly capture the relevant information of the spectral image, various methods that extract classification features from compressive samples have been recently proposed. However, these techniques require a feature extraction procedure that reorders measurements using the information embedded in the coded aperture patterns. In this paper, a method that fuses features directly from dual-resolution compressive measurements is proposed for spectral image classification. More precisely, the fusion method is formulated as an inverse problem that estimates high-spatial-resolution and low-dimensional feature bands from compressive measurements. To this end, the decimation matrices that describe the compressive measurements as degraded versions of the fused features are mathematically modeled using the information embedded in the coded aperture patterns. Furthermore, we include both a sparsity-promoting and a total-variation (TV) regularization terms to the fusion problem in order to consider the correlations between neighbor pixels, and therefore, improve the accuracy of pixel-based classifiers. To solve the fusion problem, we describe an algorithm based on the accelerated variant of the alternating direction method of multipliers (accelerated-ADMM). Additionally, a classification approach that includes the developed fusion method and a multilayer neural network is introduced. Finally, the proposed approach is evaluated on three remote sensing spectral images and a set of compressive measurements captured in the laboratory. Extensive simulations show that the proposed classification approach outperforms other approaches under various performance metrics.

## Supplementary material

### How to run the code

Download and uncompress the `featurefusion_getfund` folder. To generate Figures and Tables in the paper, under **MATLAB** environment, navigate to the `featurefusion_getfund/pavia_university` folder and follow the instructions described below

#### Demo for Pavia University cropped image

To observe the feature bands as the TV regularization parameter increases, run in MATLAB:

	>> feature_maps_lambda

![Demo image](https://github.com/JuanMarcosRamirez/featurefusion_getfund/blob/master/images/feature_maps.png?raw=true "Demo image")

To observe the labeling maps as the TV regularization parameter increases, run in MATLAB:

	>> classification_maps_lambda

![Demo image](https://github.com/JuanMarcosRamirez/featurefusion_getfund/blob/master/images/class_lambda.png?raw=true "Demo image")

To observe the labeling performance for the different supervised classifiers, run in MATLAB:

	>> classifier_testing

![Demo image](https://github.com/JuanMarcosRamirez/featurefusion_getfund/blob/master/images/variousclassifiers.png?raw=true "Demo image")

### Platform

* Windows 10 OS, MATLAB R2018a. 

* Ubuntu 18.04 OS, MATLAB R2020a.

### License

This code package is licensed under the GNU GENERAL PUBLIC LICENSE (version 3) - see the [LICENSE](LICENSE) file for details.

### Author

* Juan Marcos Ramírez Rondón. GET-Cofund MSCA Postdoctoral Fellow. Computer Science Department. [Universidad Rey Juan Carlos](http://www.urjc.es). Móstoles, 28933, Spain. 

### Contact

[Juan Marcos Ramirez](juanmarcos.ramirez@ujrc.es)

### Date

February 15, 2021

### Acknowledgements

This work has received funding from the European Union’s Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement No 754382, GOT ENERGY TALENT. The content of this article does not reflect the official opinion of the European Union. Responsibility for the information and views expressed herein lies entirely with the authors.
