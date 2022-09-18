# Motion Retargeting

Performs motion retargeting between mocap characters. The characters can have the same number of joints(intra-structure or intra-class retargeting) or different number of joints(cross-structure retargeting). The deployed networks are based on the wasserstein-gan architecture, where both the encoder and the discriminator are built using transformers. The code has been tested on Ubuntu 20.04 LTS using python 3.8 and pytorch 1.9.0 with cuda 10.2.


## Getting Started

In the next sections you can find instructions on how to:
- Download the training and the test set
- Preprocess the training set as a preparation step for training 
- Training the retargeter network
- Retarget between various characters

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






## Authors

  -**Vasilis Lasdas** 


## License

This project is licensed under the *DOWHATEVERYOUWANT* license.


