Forward-Forward
===============

The Forward-Forward algorithm [1,2] evaluated on the MNIST handwritten digit recognition task; Python implementation following [3]. Two versions are implemented: one with [numpy](https://numpy.org/) and one with [jax](https://github.com/google/jax).

Network: 5-layers; input layer with 784 (28x28) states, 3 hidden layers with 1000 states each and an output layer with 10 states corresponding to the 10 digits.

Data partitioning: Train on 50k samples, validate on 10k, test on 10k.

Error rate (energy / softmax):
* Hinton's matlab code [2]: 1.44% / 1.47%  
* This repo (numpy): 1.33% / 1.38%       
* This repo (jax) :1.30% / 1.45%

See the [Deep Boltzmann Machines](https://github.com/jesper-olsen/rbm-py) repo for another result on the same task.

References:
-----------
[1] [The Forward-Forward Algorithm: Some Preliminary Investigations, Geoffrey Hinton, NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf) <br/>
[2] [Hinton's NIPS'22 Talk](https://nips.cc/virtual/2022/invited-talk/55869) <br/>
[3] [Hinton's matlab code](https://www.cs.toronto.edu/~hinton/ffcode.zip) <br/>
[4] [Hinton's preprocessed MNIST db](https://www.cs.toronto.edu/~hinton/mnistdata.mat) <br/>
[5] [LeCun's raw MNIST db](http://yann.lecun.com/exdb/mnist/)

Run:
----

Download MNIST - either [5] or [4]; Edit mnist.py if [4]:   
```
% mkdir -p MNIST/raw
% cd MNIST
% wget https://www.cs.toronto.edu/~hinton/mnistdata.mat
% cd raw
% wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
% wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
% wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
% wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
% gunzip *.gz
% cd ../..
```

Train a model - run either main.py (numpy version) or main_jax.py.

```
% time python main.py
Batchsize: 100 Input-dim: 784 #training batches: 500
states per layer:  [784, 1000, 1000, 1000, 10]
ep   0 gain 1.000 trainlogcost 61.8054 PairwiseErrs: 5921, 5906, 6414
ep   1 gain 1.000 trainlogcost 28.5926 PairwiseErrs: 2760, 2562, 3282
ep   2 gain 1.000 trainlogcost 20.6733 PairwiseErrs: 2203, 1879, 2324
ep   3 gain 1.000 trainlogcost 16.5810 PairwiseErrs: 1866, 1595, 1991
ep   4 gain 1.000 trainlogcost 13.0995 PairwiseErrs: 1682, 1321, 1681
Energy-based errs: Train 428/10000 Valid 438/10000
Softmax-based errs: Valid 370/10000
rms:  0.0403, 0.0361, 0.0409
suprms:  0.0403, 0.0218
ep   5 gain 1.000 trainlogcost 11.1300 PairwiseErrs: 1497, 1138, 1413
ep   6 gain 1.000 trainlogcost 9.4637 PairwiseErrs: 1321, 971, 1301
ep   7 gain 1.000 trainlogcost 8.7353 PairwiseErrs: 1206, 902, 1187
ep   8 gain 1.000 trainlogcost 7.9537 PairwiseErrs: 1179, 784, 1096
ep   9 gain 1.000 trainlogcost 6.9619 PairwiseErrs: 1143, 750, 957
Energy-based errs: Train 242/10000 Valid 343/10000
Softmax-based errs: Valid 285/10000
rms:  0.0418, 0.0363, 0.0408
suprms:  0.0321, 0.0195
ep  10 gain 1.000 trainlogcost 6.0931 PairwiseErrs: 1044, 633, 882
ep  11 gain 1.000 trainlogcost 5.3677 PairwiseErrs: 960, 624, 827
ep  12 gain 1.000 trainlogcost 4.9876 PairwiseErrs: 894, 592, 836
ep  13 gain 1.000 trainlogcost 4.7329 PairwiseErrs: 862, 525, 752
ep  14 gain 1.000 trainlogcost 4.3135 PairwiseErrs: 821, 492, 694
Energy-based errs: Train 174/10000 Valid 270/10000
Softmax-based errs: Valid 249/10000
rms:  0.0424, 0.0363, 0.0404
suprms:  0.0271, 0.0197
ep  15 gain 1.000 trainlogcost 3.8409 PairwiseErrs: 789, 447, 638
ep  16 gain 1.000 trainlogcost 3.4593 PairwiseErrs: 755, 401, 607
ep  17 gain 1.000 trainlogcost 3.4946 PairwiseErrs: 706, 407, 562
ep  18 gain 1.000 trainlogcost 3.3885 PairwiseErrs: 679, 365, 572
ep  19 gain 1.000 trainlogcost 2.9400 PairwiseErrs: 623, 349, 512
Energy-based errs: Train 101/10000 Valid 229/10000
Softmax-based errs: Valid 215/10000
rms:  0.0429, 0.0358, 0.0398
suprms:  0.0238, 0.0196
ep  20 gain 1.000 trainlogcost 2.4612 PairwiseErrs: 586, 286, 430
ep  21 gain 1.000 trainlogcost 2.6846 PairwiseErrs: 566, 310, 416
ep  22 gain 1.000 trainlogcost 2.4294 PairwiseErrs: 570, 265, 441
ep  23 gain 1.000 trainlogcost 2.0932 PairwiseErrs: 524, 247, 404
ep  24 gain 1.000 trainlogcost 2.0991 PairwiseErrs: 539, 230, 372
Energy-based errs: Train 113/10000 Valid 228/10000
Softmax-based errs: Valid 237/10000
rms:  0.0432, 0.0352, 0.0391
suprms:  0.0217, 0.0193
ep  25 gain 1.000 trainlogcost 1.9902 PairwiseErrs: 508, 229, 392
ep  26 gain 1.000 trainlogcost 1.8399 PairwiseErrs: 473, 188, 338
ep  27 gain 1.000 trainlogcost 1.8218 PairwiseErrs: 497, 197, 327
ep  28 gain 1.000 trainlogcost 1.6203 PairwiseErrs: 439, 184, 292
ep  29 gain 1.000 trainlogcost 1.6777 PairwiseErrs: 439, 168, 332
Energy-based errs: Train 71/10000 Valid 178/10000
Softmax-based errs: Valid 194/10000
rms:  0.0433, 0.0347, 0.0384
suprms:  0.0202, 0.0191
ep  30 gain 1.000 trainlogcost 1.5999 PairwiseErrs: 438, 164, 253
ep  31 gain 1.000 trainlogcost 1.6148 PairwiseErrs: 434, 170, 278
ep  32 gain 1.000 trainlogcost 1.3281 PairwiseErrs: 417, 147, 280
ep  33 gain 1.000 trainlogcost 1.2226 PairwiseErrs: 373, 141, 285
ep  34 gain 1.000 trainlogcost 1.3062 PairwiseErrs: 397, 158, 251
Energy-based errs: Train 70/10000 Valid 205/10000
Softmax-based errs: Valid 209/10000
rms:  0.0433, 0.0341, 0.0376
suprms:  0.0190, 0.0186
ep  35 gain 1.000 trainlogcost 1.3162 PairwiseErrs: 362, 131, 250
ep  36 gain 1.000 trainlogcost 1.1939 PairwiseErrs: 398, 132, 265
ep  37 gain 1.000 trainlogcost 1.2284 PairwiseErrs: 357, 106, 221
ep  38 gain 1.000 trainlogcost 1.2620 PairwiseErrs: 401, 105, 207
ep  39 gain 1.000 trainlogcost 1.0853 PairwiseErrs: 353, 109, 199
Energy-based errs: Train 48/10000 Valid 196/10000
Softmax-based errs: Valid 177/10000
rms:  0.0435, 0.0333, 0.0368
suprms:  0.0181, 0.0184
ep  40 gain 1.000 trainlogcost 0.9844 PairwiseErrs: 348, 106, 206
ep  41 gain 1.000 trainlogcost 1.0729 PairwiseErrs: 332, 110, 205
ep  42 gain 1.000 trainlogcost 1.1798 PairwiseErrs: 366, 120, 191
ep  43 gain 1.000 trainlogcost 1.1198 PairwiseErrs: 370, 109, 200
ep  44 gain 1.000 trainlogcost 1.0763 PairwiseErrs: 348, 104, 205
Energy-based errs: Train 59/10000 Valid 182/10000
Softmax-based errs: Valid 184/10000
rms:  0.0440, 0.0326, 0.0361
suprms:  0.0175, 0.0181
ep  45 gain 1.000 trainlogcost 0.9911 PairwiseErrs: 308, 82, 165
ep  46 gain 1.000 trainlogcost 1.0948 PairwiseErrs: 335, 76, 195
ep  47 gain 1.000 trainlogcost 1.0943 PairwiseErrs: 334, 98, 200
ep  48 gain 1.000 trainlogcost 0.9817 PairwiseErrs: 322, 96, 178
ep  49 gain 1.000 trainlogcost 1.0245 PairwiseErrs: 358, 100, 193
Energy-based errs: Train 35/10000 Valid 191/10000
Softmax-based errs: Valid 189/10000
rms:  0.0444, 0.0319, 0.0354
suprms:  0.0172, 0.0177
ep  50 gain 1.000 trainlogcost 0.9921 PairwiseErrs: 358, 82, 172
ep  51 gain 1.000 trainlogcost 0.8527 PairwiseErrs: 295, 61, 156
ep  52 gain 1.000 trainlogcost 0.8655 PairwiseErrs: 319, 89, 167
ep  53 gain 1.000 trainlogcost 1.0242 PairwiseErrs: 323, 86, 151
ep  54 gain 1.000 trainlogcost 0.9642 PairwiseErrs: 306, 90, 156
Energy-based errs: Train 34/10000 Valid 175/10000
Softmax-based errs: Valid 172/10000
rms:  0.0446, 0.0312, 0.0347
suprms:  0.0170, 0.0176
ep  55 gain 1.000 trainlogcost 0.9050 PairwiseErrs: 298, 77, 160
ep  56 gain 1.000 trainlogcost 0.9185 PairwiseErrs: 306, 78, 180
ep  57 gain 1.000 trainlogcost 0.7065 PairwiseErrs: 277, 50, 142
ep  58 gain 1.000 trainlogcost 0.8138 PairwiseErrs: 287, 47, 165
ep  59 gain 1.000 trainlogcost 0.7789 PairwiseErrs: 249, 49, 136
Energy-based errs: Train 31/10000 Valid 179/10000
Softmax-based errs: Valid 179/10000
rms:  0.0445, 0.0304, 0.0341
suprms:  0.0166, 0.0173
ep  60 gain 1.000 trainlogcost 0.7297 PairwiseErrs: 240, 42, 123
ep  61 gain 1.000 trainlogcost 0.7902 PairwiseErrs: 250, 68, 139
ep  62 gain 1.000 trainlogcost 0.7796 PairwiseErrs: 252, 51, 110
ep  63 gain 1.000 trainlogcost 0.7372 PairwiseErrs: 219, 51, 130
ep  64 gain 0.984 trainlogcost 0.6993 PairwiseErrs: 247, 52, 111
Energy-based errs: Train 33/10000 Valid 183/10000
Softmax-based errs: Valid 182/10000
rms:  0.0444, 0.0297, 0.0334
suprms:  0.0162, 0.0171
ep  65 gain 0.968 trainlogcost 0.7500 PairwiseErrs: 247, 58, 134
ep  66 gain 0.952 trainlogcost 0.6869 PairwiseErrs: 211, 61, 130
ep  67 gain 0.936 trainlogcost 0.6493 PairwiseErrs: 227, 44, 100
ep  68 gain 0.920 trainlogcost 0.6234 PairwiseErrs: 211, 39, 105
ep  69 gain 0.904 trainlogcost 0.5912 PairwiseErrs: 228, 35, 93
Energy-based errs: Train 14/10000 Valid 177/10000
Softmax-based errs: Valid 172/10000
rms:  0.0443, 0.0290, 0.0327
suprms:  0.0157, 0.0168
ep  70 gain 0.888 trainlogcost 0.5819 PairwiseErrs: 215, 41, 92
ep  71 gain 0.872 trainlogcost 0.6130 PairwiseErrs: 215, 34, 97
ep  72 gain 0.856 trainlogcost 0.5644 PairwiseErrs: 191, 39, 94
ep  73 gain 0.840 trainlogcost 0.4894 PairwiseErrs: 198, 32, 73
ep  74 gain 0.824 trainlogcost 0.5247 PairwiseErrs: 194, 28, 68
Energy-based errs: Train 5/10000 Valid 161/10000
Softmax-based errs: Valid 173/10000
rms:  0.0441, 0.0282, 0.0319
suprms:  0.0152, 0.0164
ep  75 gain 0.808 trainlogcost 0.4815 PairwiseErrs: 141, 16, 47
ep  76 gain 0.792 trainlogcost 0.4650 PairwiseErrs: 146, 12, 51
ep  77 gain 0.776 trainlogcost 0.4523 PairwiseErrs: 150, 18, 50
ep  78 gain 0.760 trainlogcost 0.4682 PairwiseErrs: 154, 13, 42
ep  79 gain 0.744 trainlogcost 0.4587 PairwiseErrs: 166, 14, 44
Energy-based errs: Train 7/10000 Valid 173/10000
Softmax-based errs: Valid 165/10000
rms:  0.0435, 0.0274, 0.0311
suprms:  0.0148, 0.0162
ep  80 gain 0.728 trainlogcost 0.4260 PairwiseErrs: 138, 2, 35
ep  81 gain 0.712 trainlogcost 0.3953 PairwiseErrs: 119, 6, 28
ep  82 gain 0.696 trainlogcost 0.4259 PairwiseErrs: 117, 7, 24
ep  83 gain 0.680 trainlogcost 0.4074 PairwiseErrs: 113, 2, 31
ep  84 gain 0.664 trainlogcost 0.4053 PairwiseErrs: 113, 8, 37
Energy-based errs: Train 2/10000 Valid 165/10000
Softmax-based errs: Valid 163/10000
rms:  0.0427, 0.0266, 0.0303
suprms:  0.0145, 0.0159
ep  85 gain 0.648 trainlogcost 0.4272 PairwiseErrs: 85, 3, 22
ep  86 gain 0.632 trainlogcost 0.3929 PairwiseErrs: 99, 5, 21
ep  87 gain 0.616 trainlogcost 0.4033 PairwiseErrs: 89, 5, 17
ep  88 gain 0.600 trainlogcost 0.4061 PairwiseErrs: 90, 1, 12
ep  89 gain 0.584 trainlogcost 0.4070 PairwiseErrs: 96, 2, 16
Energy-based errs: Train 1/10000 Valid 151/10000
Softmax-based errs: Valid 168/10000
rms:  0.0418, 0.0259, 0.0295
suprms:  0.0146, 0.0161
ep  90 gain 0.568 trainlogcost 0.4083 PairwiseErrs: 77, 2, 16
ep  91 gain 0.552 trainlogcost 0.3981 PairwiseErrs: 66, 3, 14
ep  92 gain 0.536 trainlogcost 0.4081 PairwiseErrs: 81, 2, 11
ep  93 gain 0.520 trainlogcost 0.3977 PairwiseErrs: 68, 1, 7
ep  94 gain 0.504 trainlogcost 0.3725 PairwiseErrs: 77, 2, 7
Energy-based errs: Train 2/10000 Valid 156/10000
Softmax-based errs: Valid 161/10000
rms:  0.0411, 0.0253, 0.0289
suprms:  0.0146, 0.0163
ep  95 gain 0.488 trainlogcost 0.3916 PairwiseErrs: 71, 1, 5
ep  96 gain 0.472 trainlogcost 0.3894 PairwiseErrs: 61, 1, 6
ep  97 gain 0.456 trainlogcost 0.3923 PairwiseErrs: 75, 0, 4
ep  98 gain 0.440 trainlogcost 0.4035 PairwiseErrs: 61, 0, 5
ep  99 gain 0.424 trainlogcost 0.3927 PairwiseErrs: 58, 0, 2
Energy-based errs: Train 1/10000 Valid 146/10000
Softmax-based errs: Valid 162/10000
rms:  0.0404, 0.0247, 0.0283
suprms:  0.0147, 0.0165
ep 100 gain 0.408 trainlogcost 0.3968 PairwiseErrs: 60, 0, 2
ep 101 gain 0.392 trainlogcost 0.3965 PairwiseErrs: 36, 0, 2
ep 102 gain 0.376 trainlogcost 0.3948 PairwiseErrs: 57, 0, 4
ep 103 gain 0.360 trainlogcost 0.3887 PairwiseErrs: 50, 0, 2
ep 104 gain 0.344 trainlogcost 0.3997 PairwiseErrs: 47, 0, 4
Energy-based errs: Train 0/10000 Valid 148/10000
Softmax-based errs: Valid 170/10000
rms:  0.0397, 0.0243, 0.0278
suprms:  0.0147, 0.0166
ep 105 gain 0.328 trainlogcost 0.4037 PairwiseErrs: 36, 0, 0
ep 106 gain 0.312 trainlogcost 0.4162 PairwiseErrs: 46, 0, 0
ep 107 gain 0.296 trainlogcost 0.4211 PairwiseErrs: 39, 0, 2
ep 108 gain 0.280 trainlogcost 0.4224 PairwiseErrs: 40, 0, 0
ep 109 gain 0.264 trainlogcost 0.4185 PairwiseErrs: 35, 0, 0
Energy-based errs: Train 0/10000 Valid 138/10000
Softmax-based errs: Valid 161/10000
rms:  0.0392, 0.0240, 0.0274
suprms:  0.0149, 0.0169
ep 110 gain 0.248 trainlogcost 0.4191 PairwiseErrs: 34, 0, 4
ep 111 gain 0.232 trainlogcost 0.4227 PairwiseErrs: 35, 0, 0
ep 112 gain 0.216 trainlogcost 0.4165 PairwiseErrs: 32, 0, 1
ep 113 gain 0.200 trainlogcost 0.4290 PairwiseErrs: 29, 0, 0
ep 114 gain 0.184 trainlogcost 0.4280 PairwiseErrs: 34, 0, 0
Energy-based errs: Train 0/10000 Valid 145/10000
Softmax-based errs: Valid 163/10000
rms:  0.0389, 0.0237, 0.0271
suprms:  0.0151, 0.0170
ep 115 gain 0.168 trainlogcost 0.4329 PairwiseErrs: 34, 0, 0
ep 116 gain 0.152 trainlogcost 0.4330 PairwiseErrs: 28, 0, 0
ep 117 gain 0.136 trainlogcost 0.4374 PairwiseErrs: 23, 0, 0
ep 118 gain 0.120 trainlogcost 0.4405 PairwiseErrs: 23, 0, 1
ep 119 gain 0.104 trainlogcost 0.4478 PairwiseErrs: 24, 0, 0
Energy-based errs: Train 0/10000 Valid 150/10000
Softmax-based errs: Valid 163/10000
rms:  0.0386, 0.0236, 0.0270
suprms:  0.0152, 0.0172
ep 120 gain 0.088 trainlogcost 0.4468 PairwiseErrs: 13, 0, 0
ep 121 gain 0.072 trainlogcost 0.4476 PairwiseErrs: 21, 0, 0
ep 122 gain 0.056 trainlogcost 0.4467 PairwiseErrs: 14, 0, 0
ep 123 gain 0.040 trainlogcost 0.4450 PairwiseErrs: 23, 0, 0
ep 124 gain 0.024 trainlogcost 0.4454 PairwiseErrs: 19, 0, 0
Energy-based errs: Train 0/10000 Valid 150/10000
Softmax-based errs: Valid 166/10000
rms:  0.0385, 0.0235, 0.0269
suprms:  0.0152, 0.0172
Energy-based errs: Train 0/10000 Test 133/10000
Softmax-based errs: Train 0/10000 Test 138/10000

real	42m1.433s
user	30m12.164s
sys	5m10.400s

% sysctl -a | grep machdep.cpu
machdep.cpu.core_count: 8
machdep.cpu.brand_string: Apple M1
```
