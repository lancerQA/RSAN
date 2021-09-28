# RSAN
## 1.Preparing Dataset and Model
Datasets can be download from [Xian et al. (CVPR2017)](https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip) and take them into dir data.
## 2.Prerequisites
To install all the dependency packages, please run:   
```
pip install -r requirements.txt
```
## 3.Runing
Before running commands, you can set the hyperparameters in config.py. Please run the following commands and testing RSAN on different datasets:   
```
$ bash run/cub.sh   
$ bash run/sun.sh   
$ bash run/flo.sh  
$ bash run/awa1.sh 
$ bash run/awa2.sh   
$ bash run/apy.sh
```
