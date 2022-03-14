# FaceMap
Here we present the source codes for FaceMap. FaceMap is an effective unsupervised method on large-scale dataset for face clustering. The paper can be found [here](./FaceMap.pdf). 

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
To test our code with the refined MS1M dataset, please follow the link https://github.com/yl-1993/learn-to-cluster to download the dataset and put it in the directory ```/data``` as follow:

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
```

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

## Pre-trained Face model
To reproduce our experiments for CASIA and VGGFace2, please use the pre-trained ArcFace model in https://github.com/deepinsight/insightface/tree/master/model_zoo to extract features.
