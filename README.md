It is easy for you to train your own dataset by this project.

```
Models factory: VGG, ResNet50_v1, ResNet50_v2, Inception_ResNet_v2, EfficientNetB4, EfficientNetB5, ResNeSt

Tricks in config.py: Pretrain, weight_decay, label_smoothing, cross entropy + l2 loss, Data augment tricks, Learning rate: cosine decay with warm up, Test 10 crop	
```

## Requirements:

Conda environment is highly recommended, or you need to install cuda and cudnn by yourself.

```python
python==3.7.9
tensorflow-gpu==2.3.0
opencv-python==4.5.1.48
matplotlib==3.3.4

# recommend
conda install cudatoolkit==10.2.89
conda install cudnn==7.6.5
```

## Prepare your dataset:

Make your dataset as followed format.

Copy your dataset to the folder.

```
--- Tensorflow_CNN_Models-master
   --- data
      --- *** your dataset name, such as GarbageData
         --- classify 1
         	1.jpg
         	...
         --- classify 2
         ...
   --- models
   --- result
   --- utils
   config.py
   resnest_train.py
   split_data.py
   test.py
   train.py
```

## Split your dataset & change config.py

Run split_data.py.

```python
root_path = "./data/GarbageData"  # change your dataset name.
train_rate = 0.9  # train : test = 9 : 1
test_rate = 1 - train_rate
```

You will get your json label, train label and test label as follows.

```
   --- data
      --- *** your dataset name, such as GarbageData
         --- classify 1
         	1.jpg
         	...
         --- classify 2
         ...
      --- MyData
         --- TestData
            ...
         --- TrainData
            ...
      label_to_content.json
      test_label.txt
      train_label.txt
```

Change config.py. Select your model and tricks. Your train_title will influence your result name.

## Train

Run train.py.(Using @tf.function)

The result will be save to ./result/log and the weight will be save to ./result/weight.

```
-----------------------------------------------------------------------------------------
Train epoch: 1/20, cross entropy loss: 2.65489, l2 loss: 0.33411, accuracy: 0.36311
Test epoch: 1/20, cross entropy loss: 4.05534, l2 loss: 0.33411, accuracy: 0.04850
-----------------------------------------------------------------------------------------
Train epoch: 2/20, cross entropy loss: 2.28954, l2 loss: 0.37730, accuracy: 0.45644
Test epoch: 2/20, cross entropy loss: 2.18764, l2 loss: 0.37730, accuracy: 0.40532
-----------------------------------------------------------------------------------------
Train epoch: 3/20, cross entropy loss: 1.99214, l2 loss: 0.40373, accuracy: 0.55919
Test epoch: 3/20, cross entropy loss: 1.80864, l2 loss: 0.40373, accuracy: 0.50166
-----------------------------------------------------------------------------------------
Train epoch: 4/20, cross entropy loss: 1.82606, l2 loss: 0.42428, accuracy: 0.61079
Test epoch: 4/20, cross entropy loss: 1.53145, l2 loss: 0.42428, accuracy: 0.57475
-----------------------------------------------------------------------------------------
Train epoch: 5/20, cross entropy loss: 1.71733, l2 loss: 0.44046, accuracy: 0.65329
Test epoch: 5/20, cross entropy loss: 1.25438, l2 loss: 0.44046, accuracy: 0.64850
-----------------------------------------------------------------------------------------
Train epoch: 6/20, cross entropy loss: 1.60470, l2 loss: 0.45118, accuracy: 0.68812
Test epoch: 6/20, cross entropy loss: 1.21690, l2 loss: 0.45118, accuracy: 0.65050
-----------------------------------------------------------------------------------------
Train epoch: 7/20, cross entropy loss: 1.52050, l2 loss: 0.45884, accuracy: 0.72044
Test epoch: 7/20, cross entropy loss: 1.12248, l2 loss: 0.45884, accuracy: 0.70299
-----------------------------------------------------------------------------------------
Train epoch: 8/20, cross entropy loss: 1.43504, l2 loss: 0.46350, accuracy: 0.75163
Test epoch: 8/20, cross entropy loss: 1.03309, l2 loss: 0.46350, accuracy: 0.70764
-----------------------------------------------------------------------------------------
Train epoch: 9/20, cross entropy loss: 1.36629, l2 loss: 0.46593, accuracy: 0.77569
Test epoch: 9/20, cross entropy loss: 0.98224, l2 loss: 0.46593, accuracy: 0.74286
-----------------------------------------------------------------------------------------
Train epoch: 10/20, cross entropy loss: 1.28774, l2 loss: 0.46577, accuracy: 0.80885
Test epoch: 10/20, cross entropy loss: 0.87786, l2 loss: 0.46577, accuracy: 0.75615
-----------------------------------------------------------------------------------------
Train epoch: 11/20, cross entropy loss: 1.21254, l2 loss: 0.46423, accuracy: 0.83488
Test epoch: 11/20, cross entropy loss: 0.82786, l2 loss: 0.46423, accuracy: 0.78140
-----------------------------------------------------------------------------------------
Train epoch: 12/20, cross entropy loss: 1.15379, l2 loss: 0.46184, accuracy: 0.85703
Test epoch: 12/20, cross entropy loss: 0.71040, l2 loss: 0.46184, accuracy: 0.80997
-----------------------------------------------------------------------------------------
Train epoch: 13/20, cross entropy loss: 1.10205, l2 loss: 0.45906, accuracy: 0.88010
Test epoch: 13/20, cross entropy loss: 0.71692, l2 loss: 0.45906, accuracy: 0.81130
-----------------------------------------------------------------------------------------
Train epoch: 14/20, cross entropy loss: 1.05009, l2 loss: 0.45617, accuracy: 0.90014
Test epoch: 14/20, cross entropy loss: 0.60689, l2 loss: 0.45617, accuracy: 0.83522
-----------------------------------------------------------------------------------------
Train epoch: 15/20, cross entropy loss: 1.00174, l2 loss: 0.45365, accuracy: 0.91713
Test epoch: 15/20, cross entropy loss: 0.60150, l2 loss: 0.45365, accuracy: 0.84385
-----------------------------------------------------------------------------------------
Train epoch: 16/20, cross entropy loss: 0.98002, l2 loss: 0.45166, accuracy: 0.92829
Test epoch: 16/20, cross entropy loss: 0.62462, l2 loss: 0.45166, accuracy: 0.84718
-----------------------------------------------------------------------------------------
Train epoch: 17/20, cross entropy loss: 0.94655, l2 loss: 0.45032, accuracy: 0.93960
Test epoch: 17/20, cross entropy loss: 0.60040, l2 loss: 0.45032, accuracy: 0.85648
-----------------------------------------------------------------------------------------
Train epoch: 18/20, cross entropy loss: 0.94355, l2 loss: 0.44958, accuracy: 0.94028
Test epoch: 18/20, cross entropy loss: 0.58464, l2 loss: 0.44958, accuracy: 0.85249
-----------------------------------------------------------------------------------------
Train epoch: 19/20, cross entropy loss: 0.92963, l2 loss: 0.44929, accuracy: 0.94347
Test epoch: 19/20, cross entropy loss: 0.58871, l2 loss: 0.44929, accuracy: 0.85781
-----------------------------------------------------------------------------------------
Train epoch: 20/20, cross entropy loss: 0.92727, l2 loss: 0.44924, accuracy: 0.94597
Test epoch: 20/20, cross entropy loss: 0.58803, l2 loss: 0.44924, accuracy: 0.85914
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
Test 10 crop: cross entropy loss: 19.11319, l2 loss: 0.44924, accuracy: 0.87442
-----------------------------------------------------------------------------------------
```

[![image]](https://github.com/BillyBai/Tensorflow_CNN_Models/blob/master/result/log/ResNet50_v1_Pretrain_Tricks_accuracy.jpg)

[![image]](https://github.com/BillyBai/Tensorflow_CNN_Models/blob/master/result/log/ResNet50_v1_Pretrain_Tricks_loss.jpg)

result/log/ResNet50_v1_Pretrain_Tricks_loss.jpg

You can also use model.fit as resnest_train.py

## Test

Run test.py.

Set your model path, image path and the model in main().

You can test one image to see the confidence, or you can test all of the test images at once.

You can use test 10 crops trick to improve accuracy by about 2%.

```
Loading model...
-----------------------------------------------------------------------------------------
Test one image:
image: ./data/MyData/TestData/01_Dry_Box/01_Dry_Box_14.jpg
classification: 01_Dry_Box
confidence: 0.8301
run time: 0.0
[0.8301 0.0032 0.0039 0.0041 0.0124 0.0041 0.007  0.004  0.0034 0.0032
 0.005  0.0048 0.0034 0.0038 0.0021 0.0031 0.0045 0.0046 0.016  0.0047
 0.0047 0.0032 0.0044 0.002  0.0036 0.0024 0.0049 0.0035 0.0027 0.0025
 0.0063 0.0025 0.0047 0.0031 0.0032 0.0041 0.004  0.0023 0.0046 0.004 ]
-----------------------------------------------------------------------------------------
-----------------------------------------------------------------------------------------
Test one image 10 crop:
image: ./data/MyData/TestData/01_Dry_Box/01_Dry_Box_14.jpg
classification: 01_Dry_Box
confidence: 0.8297
run time: 0.0
[0.8297 0.0036 0.0038 0.0039 0.0118 0.0038 0.0073 0.0039 0.0033 0.0031
 0.0051 0.0045 0.0034 0.0038 0.003  0.0032 0.0045 0.0042 0.0173 0.0051
 0.0043 0.0037 0.0041 0.002  0.0035 0.0023 0.0043 0.004  0.0027 0.0028
 0.0051 0.0026 0.0043 0.0035 0.003  0.0048 0.0039 0.0023 0.0047 0.0038]
-----------------------------------------------------------------------------------------
```

## If the project is helpful to you, please give me a star, thank you for your encouragement.
