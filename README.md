# FaceMap
Here we present the source codes for FaceMap. FaceMap is an effective unsupervised method on large-scale dataset for face clustering. The paper can be found [here](./ECCV2022-FaceMap.pdf). 

![FaceMap-framework](./FaceMap-framework.png)

## Requirements 
 - python=3.7
 - infomap=1.9.0
 - faiss


## Datasets 
```
cd FaceMap
```

Create a new folder for data:

```
mkdir data
```

To run the code, please download the refined MS1M dataset and partition it into 10 splits, then construct the data directory as follows:

```
|——data
   |——MS1M
       |——features
          |——part0_train.bin
          |——part1_test.bin
          |——...
          |——part9_test.bin
       |——labels
          |——part0_train.meta
          |——part1_test.meta
          |——...
          |——part9_test.meta
   |——CASIA
       |——features
          |——part0_train.npy
          |——part1_test.npy
       |——labels
          |——part0_train.txt
          |——part1_test.txt
   |——VGG
       |——features
          |——part0_train.npy
          |——part1_test.npy
       |——labels
          |——part0_train.txt
          |——part1_test.txt
```

We have used MS1M from: https://github.com/yl-1993/learn-to-cluster.

CASIA and VGG can be downloaded from [Google Drive]() or [BaiduYunPan]()[password: xxxx]


## Run 
Adjust the configuration in ```./configs/config.py```, then run the algorithm as follows:
```
cd FaceMap
python main.py
```

## Evaluation 
```
cd FaceMap
python main_eval.py
```
