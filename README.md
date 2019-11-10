Deep Learning in Python
===========================

## Contents
1. [Artificial Neural Networks](#artificial-neural-networks)
2. [Convolutional Neural Networks](#convolutional-neural-networks)
3. [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)
4. [How to run the Python program](#how-to-run-the-python-program)

## Artificial Neural Networks

a.  [ann.py](https://github.com/ramonfigueiredopessoa/deep_learning_in_python/blob/master/1_artificial_neural_networks/ann.py)

* Importing the dataset ([Churn_Modelling.csv](https://github.com/ramonfigueiredopessoa/deep_learning_in_python/blob/master/1_artificial_neural_networks/Churn_Modelling.csv))
* Encoding categorical data
* Splitting the dataset into the Training set and Test set
* Feature Scaling
* Creating the Artificial Neural Networks (ANN) using [Keras](https://keras.io/)
	* Initialising the ANN
	* Adding the input layer and the first hidden layer
	* Adding the second hidden layer
	* Adding the output layer
	* Compiling the ANN
	* Fitting the ANN to the Training set
* Predicting the Test set results
* Predicting a single new observation
* Creating the Confusion Matrix
* Calculating metrics using the confusion matrix

### Training the ANN with Stochastic Gradient Descent

**Step 1.** Randomly initialise the weights to small numbers close to 0 (but not 0).

**Step 2.** Input the first observation of your dataset in the input layer, each feature in one input node.

**Step 3.** Forward-Propagation: from left to right, the neurons are activated in a way that the impact of each neuron's activation is limited by the weights. Propagate the activations until getting the predicted results y.

**Step 4.** Compare the predicted results to the actual result. Measure the generated error.

**Step 5.** Back-Propagation: fron right to left, the error is back-propagated. Update the weights according to how much they are responsible for the error. The learning rate decides by how much we update the weights.

**Step 6.** Repeat Steps 1 to 5 and update the weights after each observation (Reinforcement Learning). Or: Repeat Steps 1 to 5 but update the weights only after a batch of observation (Batch Learning).

**Step 7.** When the whole training set passed through the ANN, that makes an epoch. Redo more epochs.

See [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)

### ANN algorithm output using Keras and TensorFlow (GPU)

#### Computer settings

* Ubuntu 18.04.3 LTS 64-bit
* GPU NVIDIA GeForce GTX 1080 Ti
* MacBook Pro (15-inch, 2017)
* Intel Core i7-7700 CPU @ 3.60GHz × 8
* Memory 32 GB

TensorFlow-GPU
* tensorflow-2.0.0
* cuda 10.0
* cuDNN 7.5.0

```
Using TensorFlow backend.

Epoch 1/100
 260/8000 [..............................] - ETA: 8s - loss: 0.6892 - accuracy: 0.7923  2019-11-09 19:54:30.292135: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
8000/8000 [==============================] - 2s 209us/step - loss: 0.4847 - accuracy: 0.7960
Epoch 2/100
8000/8000 [==============================] - 2s 191us/step - loss: 0.4272 - accuracy: 0.7960
Epoch 3/100
8000/8000 [==============================] - 1s 183us/step - loss: 0.4207 - accuracy: 0.8112
Epoch 4/100
8000/8000 [==============================] - 1s 184us/step - loss: 0.4144 - accuracy: 0.8292
Epoch 5/100
8000/8000 [==============================] - 1s 176us/step - loss: 0.4100 - accuracy: 0.8349
Epoch 6/100
8000/8000 [==============================] - 1s 183us/step - loss: 0.4070 - accuracy: 0.8344
Epoch 7/100
8000/8000 [==============================] - 2s 189us/step - loss: 0.4046 - accuracy: 0.8356
Epoch 8/100
8000/8000 [==============================] - 1s 182us/step - loss: 0.4033 - accuracy: 0.8335
Epoch 9/100
8000/8000 [==============================] - 1s 187us/step - loss: 0.4027 - accuracy: 0.8350
Epoch 10/100
8000/8000 [==============================] - 1s 161us/step - loss: 0.4013 - accuracy: 0.8364
Epoch 11/100
8000/8000 [==============================] - 1s 155us/step - loss: 0.4006 - accuracy: 0.8345
Epoch 12/100
8000/8000 [==============================] - 2s 192us/step - loss: 0.4000 - accuracy: 0.8341
Epoch 13/100
8000/8000 [==============================] - 1s 173us/step - loss: 0.3995 - accuracy: 0.8339
Epoch 14/100
8000/8000 [==============================] - 1s 179us/step - loss: 0.3990 - accuracy: 0.8338
Epoch 15/100
8000/8000 [==============================] - 1s 177us/step - loss: 0.3984 - accuracy: 0.8356
Epoch 16/100
8000/8000 [==============================] - 2s 192us/step - loss: 0.3981 - accuracy: 0.8350
Epoch 17/100
8000/8000 [==============================] - 2s 199us/step - loss: 0.3978 - accuracy: 0.8346
Epoch 18/100
8000/8000 [==============================] - 2s 189us/step - loss: 0.3974 - accuracy: 0.8354
Epoch 19/100
8000/8000 [==============================] - 2s 201us/step - loss: 0.3976 - accuracy: 0.8356
Epoch 20/100
8000/8000 [==============================] - 1s 179us/step - loss: 0.3971 - accuracy: 0.8363
Epoch 21/100
8000/8000 [==============================] - 2s 189us/step - loss: 0.3972 - accuracy: 0.8365
Epoch 22/100
8000/8000 [==============================] - 2s 188us/step - loss: 0.3961 - accuracy: 0.8347
Epoch 23/100
8000/8000 [==============================] - 2s 190us/step - loss: 0.3967 - accuracy: 0.8342
Epoch 24/100
8000/8000 [==============================] - 2s 192us/step - loss: 0.3964 - accuracy: 0.8360
Epoch 25/100
8000/8000 [==============================] - 2s 193us/step - loss: 0.3963 - accuracy: 0.8363
Epoch 26/100
8000/8000 [==============================] - 2s 192us/step - loss: 0.3958 - accuracy: 0.8361
Epoch 27/100
8000/8000 [==============================] - 2s 191us/step - loss: 0.3954 - accuracy: 0.8355
Epoch 28/100
8000/8000 [==============================] - 2s 193us/step - loss: 0.3954 - accuracy: 0.8360
Epoch 29/100
8000/8000 [==============================] - 2s 202us/step - loss: 0.3954 - accuracy: 0.8347
Epoch 30/100
8000/8000 [==============================] - 1s 185us/step - loss: 0.3955 - accuracy: 0.8349
Epoch 31/100
8000/8000 [==============================] - 2s 215us/step - loss: 0.3955 - accuracy: 0.8370
Epoch 32/100
8000/8000 [==============================] - 2s 203us/step - loss: 0.3956 - accuracy: 0.8375
Epoch 33/100
8000/8000 [==============================] - 2s 198us/step - loss: 0.3950 - accuracy: 0.8389
Epoch 34/100
8000/8000 [==============================] - 2s 217us/step - loss: 0.3946 - accuracy: 0.8353
Epoch 35/100
8000/8000 [==============================] - 2s 208us/step - loss: 0.3948 - accuracy: 0.8357
Epoch 36/100
8000/8000 [==============================] - 1s 147us/step - loss: 0.3941 - accuracy: 0.8391
Epoch 37/100
8000/8000 [==============================] - 1s 156us/step - loss: 0.3948 - accuracy: 0.8379
Epoch 38/100
8000/8000 [==============================] - 1s 153us/step - loss: 0.3940 - accuracy: 0.8370
Epoch 39/100
8000/8000 [==============================] - 1s 159us/step - loss: 0.3941 - accuracy: 0.8350
Epoch 40/100
8000/8000 [==============================] - 1s 178us/step - loss: 0.3940 - accuracy: 0.8356
Epoch 41/100
8000/8000 [==============================] - 1s 185us/step - loss: 0.3935 - accuracy: 0.8376
Epoch 42/100
8000/8000 [==============================] - 1s 185us/step - loss: 0.3927 - accuracy: 0.8382
Epoch 43/100
8000/8000 [==============================] - 1s 183us/step - loss: 0.3930 - accuracy: 0.8374
Epoch 44/100
8000/8000 [==============================] - 2s 213us/step - loss: 0.3922 - accuracy: 0.8367
Epoch 45/100
8000/8000 [==============================] - 1s 183us/step - loss: 0.3905 - accuracy: 0.8376
Epoch 46/100
8000/8000 [==============================] - 1s 178us/step - loss: 0.3893 - accuracy: 0.8396
Epoch 47/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3860 - accuracy: 0.8403
Epoch 48/100
8000/8000 [==============================] - 1s 176us/step - loss: 0.3804 - accuracy: 0.8416
Epoch 49/100
8000/8000 [==============================] - 1s 178us/step - loss: 0.3738 - accuracy: 0.8445
Epoch 50/100
8000/8000 [==============================] - 1s 165us/step - loss: 0.3685 - accuracy: 0.8476
Epoch 51/100
8000/8000 [==============================] - 1s 177us/step - loss: 0.3657 - accuracy: 0.8503
Epoch 52/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3630 - accuracy: 0.8505
Epoch 53/100
8000/8000 [==============================] - 1s 178us/step - loss: 0.3628 - accuracy: 0.8520
Epoch 54/100
8000/8000 [==============================] - 1s 185us/step - loss: 0.3604 - accuracy: 0.8539
Epoch 55/100
8000/8000 [==============================] - 1s 175us/step - loss: 0.3593 - accuracy: 0.8559
Epoch 56/100
8000/8000 [==============================] - 1s 172us/step - loss: 0.3571 - accuracy: 0.8576
Epoch 57/100
8000/8000 [==============================] - 1s 171us/step - loss: 0.3564 - accuracy: 0.8575
Epoch 58/100
8000/8000 [==============================] - 1s 171us/step - loss: 0.3552 - accuracy: 0.8580
Epoch 59/100
8000/8000 [==============================] - 1s 172us/step - loss: 0.3546 - accuracy: 0.8596
Epoch 60/100
8000/8000 [==============================] - 1s 169us/step - loss: 0.3539 - accuracy: 0.8580
Epoch 61/100
8000/8000 [==============================] - 1s 178us/step - loss: 0.3530 - accuracy: 0.8600
Epoch 62/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3516 - accuracy: 0.8595
Epoch 63/100
8000/8000 [==============================] - 1s 180us/step - loss: 0.3522 - accuracy: 0.8620
Epoch 64/100
8000/8000 [==============================] - 1s 177us/step - loss: 0.3515 - accuracy: 0.8618
Epoch 65/100
8000/8000 [==============================] - 1s 165us/step - loss: 0.3508 - accuracy: 0.8614
Epoch 66/100
8000/8000 [==============================] - 1s 168us/step - loss: 0.3510 - accuracy: 0.8620
Epoch 67/100
8000/8000 [==============================] - 1s 175us/step - loss: 0.3503 - accuracy: 0.8609
Epoch 68/100
8000/8000 [==============================] - 1s 166us/step - loss: 0.3499 - accuracy: 0.8609
Epoch 69/100
8000/8000 [==============================] - 1s 182us/step - loss: 0.3490 - accuracy: 0.8619
Epoch 70/100
8000/8000 [==============================] - 1s 167us/step - loss: 0.3490 - accuracy: 0.8611
Epoch 71/100
8000/8000 [==============================] - 1s 179us/step - loss: 0.3495 - accuracy: 0.8604
Epoch 72/100
8000/8000 [==============================] - 1s 178us/step - loss: 0.3472 - accuracy: 0.8605
Epoch 73/100
8000/8000 [==============================] - 1s 167us/step - loss: 0.3453 - accuracy: 0.8609
Epoch 74/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3446 - accuracy: 0.8611
Epoch 75/100
8000/8000 [==============================] - 1s 170us/step - loss: 0.3440 - accuracy: 0.8612
Epoch 76/100
8000/8000 [==============================] - 1s 172us/step - loss: 0.3443 - accuracy: 0.8599
Epoch 77/100
8000/8000 [==============================] - 2s 201us/step - loss: 0.3440 - accuracy: 0.8580
Epoch 78/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3422 - accuracy: 0.8591
Epoch 79/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3425 - accuracy: 0.8593
Epoch 80/100
8000/8000 [==============================] - 1s 171us/step - loss: 0.3430 - accuracy: 0.8608
Epoch 81/100
8000/8000 [==============================] - 1s 172us/step - loss: 0.3426 - accuracy: 0.8594
Epoch 82/100
8000/8000 [==============================] - 1s 173us/step - loss: 0.3424 - accuracy: 0.8606
Epoch 83/100
8000/8000 [==============================] - 1s 185us/step - loss: 0.3422 - accuracy: 0.8616
Epoch 84/100
8000/8000 [==============================] - 1s 173us/step - loss: 0.3425 - accuracy: 0.8610
Epoch 85/100
8000/8000 [==============================] - 1s 171us/step - loss: 0.3420 - accuracy: 0.8610
Epoch 86/100
8000/8000 [==============================] - 2s 188us/step - loss: 0.3422 - accuracy: 0.8604
Epoch 87/100
8000/8000 [==============================] - 1s 175us/step - loss: 0.3422 - accuracy: 0.8596
Epoch 88/100
8000/8000 [==============================] - 1s 176us/step - loss: 0.3423 - accuracy: 0.8600
Epoch 89/100
8000/8000 [==============================] - 2s 198us/step - loss: 0.3418 - accuracy: 0.8600
Epoch 90/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3421 - accuracy: 0.8580
Epoch 91/100
8000/8000 [==============================] - 1s 148us/step - loss: 0.3424 - accuracy: 0.8631
Epoch 92/100
8000/8000 [==============================] - 1s 162us/step - loss: 0.3418 - accuracy: 0.8602
Epoch 93/100
8000/8000 [==============================] - 1s 153us/step - loss: 0.3421 - accuracy: 0.8604
Epoch 94/100
8000/8000 [==============================] - 1s 148us/step - loss: 0.3421 - accuracy: 0.8590
Epoch 95/100
8000/8000 [==============================] - 1s 152us/step - loss: 0.3419 - accuracy: 0.8609
Epoch 96/100
8000/8000 [==============================] - 1s 148us/step - loss: 0.3411 - accuracy: 0.8602
Epoch 97/100
8000/8000 [==============================] - 1s 166us/step - loss: 0.3410 - accuracy: 0.8621
Epoch 98/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3413 - accuracy: 0.8609
Epoch 99/100
8000/8000 [==============================] - 1s 168us/step - loss: 0.3415 - accuracy: 0.8611
Epoch 100/100
8000/8000 [==============================] - 1s 174us/step - loss: 0.3410 - accuracy: 0.8615


Predicting the Test set results
 [0 1 0 ... 0 0 0]


Predicting [0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]
 [[False]]


Confusion Matrix
 [[1518   77]
 [ 200  205]]


True Positive (TP): 1518
False Negative (FN): 77
True Negative (TN): 200
False Positive (FP): 205


Accuracy = (TP + TN) / (TP + TN + FP + FN): 85.90 %
Recall = TP / (TP + FN): 95.17 %
Precision = TP / (TP + FP): 88.10 %
Fmeasure = (2 * recall * precision) / (recall + precision): 91.50 %

```

Go to [Contents](#contents)

## Convolutional Neural Networks

a.  [cnn.py](https://github.com/ramonfigueiredopessoa/deep_learning_in_python/blob/master/2_convolutional_neural_networks/cnn.py)

* Using a dataset with 10000 images of cats and dogs ([cats and dogs dataset](https://github.com/ramonfigueiredopessoa/deep_learning_in_python/blob/master/2_convolutional_neural_networks/dataset.txt))
	* Training set: 8000 (4000 cat images + 4000 dogs images)
	* Test set: 2000 (1000 cat images + 1000 dogs images)
* Creating the Convolutional Neural Network using [Keras](https://keras.io/)
	* Initialising the CNN
	* Convolution
	* Pooling
	* Adding a second convolutional layer
	* Flattening
	* Full connection
	* Compiling the CNN
	* Fitting the CNN to the images

### Training the CNN

**Step 1.** Convolution

**Step 2.** Max Pooling

**Step 3.** Flattening

**Step 4.** Full connection

See [Metrics using the Confusion Matrix](#metrics-using-the-confusion-matrix)

### CNN algorithm output using Keras and TensorFlow (CPU)

* **Note:** 
	* I executed  this code using tensorflow (CPU). Execute this code using CPU takes lots of time. 
	* If you have GPU you can use tensorflow-gpu. 
	* The following GPU-enabled devices are supported: NVIDIA(R) GPU card with CUDA(R) Compute Capability 3.5 or higher. See the list of [CUDA-enabled GPU cards](https://developer.nvidia.com/cuda-gpus).
	* The following NVIDIA(R) software must be installed on your system:
		* [NVIDIA(R) GPU drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us) — CUDA 10.0 requires 410.x or higher.
		* [CUDA(R) Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) — TensorFlow supports CUDA 10.0 (TensorFlow >= 1.13.0)
		* [CUPTI](https://docs.nvidia.com/cuda/cupti/) ships with the CUDA Toolkit.
		* [cuDNN SDK](https://developer.nvidia.com/cudnn) (>= 7.4.1)
		* (Optional) [TensorRT 5.0](https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html) to improve latency and throughput for inference on some models.

##### Computer settings

* Windows 10 Professional (x64)
* Processor Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz
* Memory 32 GB

```
Using TensorFlow backend.

Found 8000 images belonging to 2 classes.
Found 2000 images belonging to 2 classes.
Epoch 1/25
8000/8000 [==============================] - 887s 111ms/step - loss: 0.3579 - accuracy: 0.8324 - val_loss: 0.5902 - val_accuracy: 0.8086
Epoch 2/25
8000/8000 [==============================] - 759s 95ms/step - loss: 0.1079 - accuracy: 0.9588 - val_loss: 0.5788 - val_accuracy: 0.8000
Epoch 3/25
8000/8000 [==============================] - 716s 90ms/step - loss: 0.0539 - accuracy: 0.9813 - val_loss: 0.6424 - val_accuracy: 0.7990
Epoch 4/25
8000/8000 [==============================] - 717s 90ms/step - loss: 0.0386 - accuracy: 0.9867 - val_loss: 0.9461 - val_accuracy: 0.8047
Epoch 5/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0308 - accuracy: 0.9896 - val_loss: 1.3553 - val_accuracy: 0.7848
Epoch 6/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0259 - accuracy: 0.9913 - val_loss: 1.3581 - val_accuracy: 0.7889
Epoch 7/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0224 - accuracy: 0.9927 - val_loss: 0.9129 - val_accuracy: 0.8069
Epoch 8/25
8000/8000 [==============================] - 726s 91ms/step - loss: 0.0189 - accuracy: 0.9939 - val_loss: 1.2980 - val_accuracy: 0.7935
Epoch 9/25
8000/8000 [==============================] - 736s 92ms/step - loss: 0.0178 - accuracy: 0.9943 - val_loss: 1.9009 - val_accuracy: 0.7885
Epoch 10/25
8000/8000 [==============================] - 716s 90ms/step - loss: 0.0163 - accuracy: 0.9949 - val_loss: 1.4097 - val_accuracy: 0.7889
Epoch 11/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0138 - accuracy: 0.9957 - val_loss: 0.7039 - val_accuracy: 0.7976
Epoch 12/25
8000/8000 [==============================] - 716s 89ms/step - loss: 0.0130 - accuracy: 0.9959 - val_loss: 1.4262 - val_accuracy: 0.7914
Epoch 13/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0123 - accuracy: 0.9963 - val_loss: 0.7608 - val_accuracy: 0.7976
Epoch 14/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0122 - accuracy: 0.9963 - val_loss: 2.7076 - val_accuracy: 0.8005
Epoch 15/25
8000/8000 [==============================] - 715s 89ms/step - loss: 0.0108 - accuracy: 0.9967 - val_loss: 3.5931 - val_accuracy: 0.7930
Epoch 16/25
8000/8000 [==============================] - 713s 89ms/step - loss: 0.0104 - accuracy: 0.9969 - val_loss: 0.6374 - val_accuracy: 0.7970
Epoch 17/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0100 - accuracy: 0.9969 - val_loss: 1.3442 - val_accuracy: 0.8016
Epoch 18/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0091 - accuracy: 0.9973 - val_loss: 2.7414 - val_accuracy: 0.8020
Epoch 19/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0091 - accuracy: 0.9973 - val_loss: 1.3481 - val_accuracy: 0.7944
Epoch 20/25
8000/8000 [==============================] - 712s 89ms/step - loss: 0.0089 - accuracy: 0.9974 - val_loss: 4.1220 - val_accuracy: 0.7976
Epoch 21/25
8000/8000 [==============================] - 714s 89ms/step - loss: 0.0086 - accuracy: 0.9975 - val_loss: 0.8613 - val_accuracy: 0.7923
Epoch 22/25
8000/8000 [==============================] - 716s 90ms/step - loss: 0.0088 - accuracy: 0.9976 - val_loss: 3.9867 - val_accuracy: 0.7960
Epoch 23/25
8000/8000 [==============================] - 731s 91ms/step - loss: 0.0076 - accuracy: 0.9977 - val_loss: 1.3609 - val_accuracy: 0.7892
Epoch 24/25
8000/8000 [==============================] - 749s 94ms/step - loss: 0.0074 - accuracy: 0.9978 - val_loss: 2.1906 - val_accuracy: 0.7942
Epoch 25/25
8000/8000 [==============================] - 718s 90ms/step - loss: 0.0067 - accuracy: 0.9979 - val_loss: 1.2555 - val_accuracy: 0.8042
```

## Metrics using the Confusion Matrix 

### Confusion Matrix (Binary Classification)

![Confusion Matrix: Binary Classification](https://github.com/ramonfigueiredopessoa/deep_learning_in_python/blob/master/confusion_matrix-binary_classification.png)

### True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP)

* **True Positive (TP):** Observation is positive, and is predicted to be positive.
* **False Negative (FN):** Observation is positive, but is predicted negative.
* **True Negative (TN):** Observation is negative, and is predicted to be negative.
* **False Positive (FP):** Observation is negative, but is predicted positive.

### Classification Rate / Accuracy

Classification Rate or Accuracy is given by the relation:

Accuracy = (TP + TN) / (TP + TN + FP + FN)

However, there are problems with accuracy.  It assumes equal costs for both kinds of errors. A 99% accuracy can be excellent, good, mediocre, poor or terrible depending upon the problem.

### Recall

Recall can be defined as the ratio of the total number of correctly classified positive examples divide to the total number of positive examples. High Recall indicates the class is correctly recognized (small number of FN).

Recall is given by the relation:

Recall = TP / (TP + FN)

### Precision

To get the value of precision we divide the total number of correctly classified positive examples by the total number of predicted positive examples. High Precision indicates an example labeled as positive is indeed positive (small number of FP).

Precision is given by the relation:

Precision = TP / (TP + FP)

High recall, low precision: 
This means that most of the positive examples are correctly recognized (low FN) but there are a lot of false positives.

Low recall, high precision: 
This shows that we miss a lot of positive examples (high FN) but those we predict as positive are indeed positive (low FP)

### F1-Score

Since we have two measures (Precision and Recall) it helps to have a measurement that represents both of them. We calculate an F1-Score (F-measure) which uses Harmonic Mean in place of Arithmetic Mean as it punishes the extreme values more.

The F1-Score will always be nearer to the smaller value of Precision or Recall.

F1-Score = (2 * Recall * Precision) / (Recall + Presision)

### Confusion Matrix (Multi-Class Classification)

![Confusion Matrix: Multi-Class Classification - TP, TN, FP, FN](https://github.com/ramonfigueiredopessoa/deep_learning_in_python/blob/master/confusion_matrix-multi-class_classification-TP_TN_FP_FN.jpg)

### True Positive (TP), False Negative (FN), True Negative (TN), False Positive (FP)

* **True Positive (TP):** Observation is positive, and is predicted to be positive.
* **False Negative (FN):** Observation is positive, but is predicted negative.
* **True Negative (TN):** Observation is negative, and is predicted to be negative.
* **False Positive (FP):** Observation is negative, but is predicted positive.

### Classification Rate / Accuracy

Accuracy = (TP + TN) / (TP + TN + FP + FN)

### Recall

Recall = TP / (TP + FN)

### Precision

Precision = TP / (TP + FP)

### F1-Score

F1-Score = (2 * Recall * Precision) / (Recall + Presision)

### Example of metrics calculation using a multi-class confusion matrix

![Confusion Matrix: Multi-Class Classification](https://github.com/ramonfigueiredopessoa/deep_learning_in_python/blob/master/confusion_matrix-multi-class_classification.png)

* True Positive (TP) of class 1: 14
* True Positive (TP) of class 2: 15
* True Positive (TP) of class 3: 6

### ACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 1

**Accuracy (class 1)** = TP (class 1) + cm[1][1] + cm[1][2] + cm[2][1] + cm[2][2] / sum_matrix_values

= 14 + (15 + 0 + 0 + 6) / (14 + 0 + 0 + 1 + 15 + 0 + 0 + 0 + 6) = 35/36 = 0.9722222222 (97.22 %)

**Precision (class 1)** = TP (class 1) / (cm[0][0] + cm[1][0] + cm[2][0])

= 14 / (14 + 1 + 0) = 14/15 = 0.9333333333 (93.33 %)

**Recall (class 1)** = TP (class 1) / (cm[0][0] + cm[0][1] + cm[0][2])

= 14 / (14 + 0 + 0) = 14/14 = 1.0 (100 %)

**F1-Score (class 1)** = (2 * recall_class1 * precision_class1) / (recall_class1 + precision_class1)

= (2 * 1.0 * 0.9333333333) / (1.0 + 0.9333333333) = 1.8666666666/1.9333333333 = 0.9655172414 (96.55 %)

### ACCURACY, PRECISION, RECALL, F1-SCORE FOR CLASS 2

**Accuracy (class 2)** = TP (class 2) + cm[0][0] + cm[0][2] + cm[2][0] + cm[2][2] / sum_matrix_values: 97.22 %

**Precision (class 2)** = TP (class 2) / (cm[0][1] + cm[1][1] + cm[2][1]): 100.00 %

**Recall (class 2)** = TP (class 2) / (cm[1][0] + cm[1][1] + cm[1][2]): 93.75 %

**F1-Score (class 2)** = (2 * recall_class2 * precision_class2) / (recall_class2 + precision_class2): 96.77 %

### PRECISION, RECALL, F1-SCORE FOR CLASS 3

**Accuracy (class 3)** = TP (class 3) + cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1] / sum_matrix_values: 100.00 %

**Precision (class 3)** = TP (class 3) / (cm[0][2] + cm[1][2] + cm[2][2]): 100.00 %

**Recall (class 3)** = TP (class 3) / (cm[2][0] + cm[2][1] + cm[2][2]): 100.00 %

**F1-Score (class 3)** = (2 * recall_class3 * precision_class3) / (recall_class3 + precision_class3): 100.00 %

Go to [Contents](#contents)

## How to run the Python program?

1. Install [virtualenv](https://virtualenv.pypa.io/en/latest/)
	* To activate the virtualenv on Linux or MacOS: ```source venv/bin/activate```
	* To activate the virtualenv on Windows: ```\venv\Script\activate.bat```

2. Run the program

```sh
cd <folder_name>/

virtualenv venv

source venv/bin/activate

pip install -r requirements.txt

python <name_of_python_program>.py
```

**Note**: To desactivate the virtual environment

```sh
deactivate
```

Go to [Contents](#contents)