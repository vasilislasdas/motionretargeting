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

Download the training and the test datasets and unzip them in the dataset folder.
- [Training set](https://drive.google.com/file/d/1UhXkFfOVaDUdpCmIGkScOoi1IG7c3olH/view?usp=sharing)
- [Test set](https://drive.google.com/file/d/1nc4MRO-QTjqqr96vuhAr4P6HdZdBM3W6/view?usp=sharing)

The structure of the dataset folder should look like this:
![alt text](https://github.com/vasilislasdas/motionretargeting/blob/main/images/dataset_image.png)

The training/test folders contain the *bvh* files from various characters of the mixamo dataset.

### Install required packages

The python packages used are contained in the **requirements.txt** file. In order to install them:

```
pip install -r requirements.txt
```

### Preprocessing

In order to preprocess the dataset, do the following, from a terminal, assuming you are in the root folder of the repository:
```
        cd utils/
        python preprocess.py
```
The above will create the file **normalized_dataset.txt**, which will be automatically placed in the train_wgan/ folder. Depending on your system, the preprocessing can take around 1-2 minutes. Make sure this file is correctly created. Otherwise, you can download a precreated file from [here](https://drive.google.com/file/d/1es64N10Kh6P78icbVJjxD0S-T_sgoKGV/view?usp=sharing), and again place it in the train_wgan/ folder.

## Retargeting

Assuming you are in the root folder:

```
cd evaluation/
python retarget.py
```
The above function performs the following steps:
- Selects a random source character.
- Selects a random motion from the source character.
- Selects a random target character.
- Performs retargeting. 


The retargeting artefacts are placed in the folder *evaluation/results*, which contains the files:
- Bvh file of the source motion: **original.bhv**.
- Bvh file of the retargeting: **final.bvh**
- Groundtruth bvh file: **groundtruth.bvh**

The results(bvh files) can be visualized using [Blender](https://www.blender.org/) or other similar applications. Multiple runs overwrite the results.

### Note 1):
Due to the randomness, the retargeting can be either *intra-structure or cross-structure*. For more fine-grained control(intra or cross retargeting), please change/harcode the source character/motion-target character within the **retarget.py** file. 

### Note 2): 
For beautiful rendering of the results, follow the instructions from [here](https://github.com/DeepMotionEditing/deep-motion-editing) (Rendering section).




## Training
The network architecture is based on the Wasserstein GAN. The generator(retargeter) is an encoder-decoder architecture. The encoder is based on transformers, whereas the decoder is a plain FFN network(which is fixed). The discriminator is also based on transformers. The various hyperparameters of the network can be found on the **net_conf.txt** file located in the train_wgan folder. You can also add multiple values for the hyperparameters seperated using commas, which leads to multiple architectures. Training the network, suffices to(assuming your are in the root folder):
```
cd train_wgan
python train.py
```
The training process will create a *trials folder* within the train_wgan folder, containing the results of each trial. A successful training will create 3 files as shown in the picture below:

![alt text](https://github.com/vasilislasdas/motionretargeting/blob/main/images/training_result.png)

There are 3 artefacts that are being created:
- The network configuration(same as the **net_conf.txt**  when a single set of hyperparameter values is used).
- The weights of the trained retargeter network(**retargeter_final.zip**). This is the most important file. If you want to perform retargeting, copy it to the evaluation folder and rename it **retargeter.zip**
- The losses during training, which can be used later for plotting.

Specifying multiple hyperparameter values in the *net_conf.txt* file will lead to multiple folders created.

### Plotting losses

Copy-paste the file **all_losses.npz** that is generated when training the network, to the *evaluation/* folder. Then, assuming you are on the root folder: 
```
cd evaluation
python plot_experiments.py
```
There is already an existing losses file inside the evaluation folder for demonstration purposes.


## DEMO

![](https://github.com/vasilislasdas/motionretargeting/blob/main/demo/demo_gif.gif)

Download the full demo [here](https://drive.google.com/file/d/1B5MjEHqhWHNlWZXJgdSG5jjYkDchAX5J/view)




## Authors

**Vasilis Lasdas** 



