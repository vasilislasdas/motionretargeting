# Motion Retargeting

Performs motion retargeting between mocap characters from the Mixamo dataset. The characters can have the same number of joints(intra-structure or intra-class retargeting) or different number of joints(cross-structure retargeting). The code has been tested on Ubuntu 20.04 LTS using python 3.8 and pytorch 1.9.0 with cuda 10.2.


## Getting Started

In the next sections you can find instructions on how to:
- Download the training and the test set.
- Preprocess the training set as a preparation step for training.
- Training the retargeter network.
- Retarget between various characters.

All the instructions are based using a linux console. Alternatively, you can add the whole folder/project using your favorite python IDE(e.g. pycharm), and perform the steps from within the IDE.

### Prerequisites

From a console, clone the repository:

```
git clone https://github.com/vasilislasdas/motionretargeting
cd motionretargeting/
```

Download the training and the test and unzip them in the dataset folder.
- [Training set](https://drive.google.com/file/d/1UhXkFfOVaDUdpCmIGkScOoi1IG7c3olH/view?usp=sharing)
- [Test set](https://drive.google.com/file/d/1nc4MRO-QTjqqr96vuhAr4P6HdZdBM3W6/view?usp=sharing)

The structure of the dataset folder should look like this:
![alt text](https://github.com/vasilislasdas/motionretargeting/blob/main/images/dataset_image.png)

The training/test folders contain the *bvh* files from various characters of the mixamo dataset. 

### Preprocessing

In order to preprocess the dataset, do the following, from a python console, assuming you are in the root folder of the repository:
```
        cd utils/
        python preprocess.py
```
The above will create the file **normalized_dataset.txt**, which will be automatically placed in the train_wgan/ folder. Depending on your system, the preprocessing can take around 1-2 minutes. Make sure this file is correctly created. Otherwise, you can download a precreated file from [here](https://drive.google.com/file/d/1es64N10Kh6P78icbVJjxD0S-T_sgoKGV/view?usp=sharing), and again place it in the train_wgan/ folder.

### Retargeting

### Training
The network architecture is based on the Wasserstein GAN. The generator(retargeter) is an encoder-decoder architecture. The encoder is based on transformers, whereas the decoder is a plain FFN network(which is fixed). The discriminator is also based on transformers. The various hyperparameters of the network can be found on the **net_conf.txt** file located in the train_wgan folder. You can also add multiple values for the hyperparameters seperated using commas, which leads to multiple architectures. Training the network, suffices to(assuming your are in the root folder):
```
cd train_wgan
python train_final.py
```
The training process will create with the train_wgan folder a *trials folder*, containing the results of each trial. A successful training will create 3 files as showin in the picture above:

![alt text](https://github.com/vasilislasdas/motionretargeting/blob/main/images/training_result.png)

There are 3 artefacts that are being created for a specific configuration of hyperparameters:
- The network configuration(same for a single set of hyperparameter values).
- The weights of the retargeter network(**retargeter.zip**).
- The losses during training, which can be used later for plotting.

Specifying multiple hypermater values in the *net_conf.txt* file will lead to multiple folders created.









## Authors

  -**Vasilis Lasdas** 


## License

This project is licensed under the *DOWHATEVERYOUWANT* license.


