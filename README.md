Forward-Forward
===============

The Forward-Forward algorithm [1,2] evaluated on the MNIST handwritten digit recognition task; Python+Numpy implementation following [3].

Error rate (energy / softmax) - train on 50k samples, validate on 10k, test on 10k:
* Hinton's matlab code [2]: 1.44% / 1.47%  
* This repo: 1.33% / 1.38%       

Network: 5-layers; input layer with 784 (28x28) states, 3 hidden layers with 1000 states each and an output layer with 10 states corresponding to the 10 digits.

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

Train a model:

```
% time python main.py
Batchsize: 100 Input-dim: 784 #training batches: 500
states per layer:  [784, 1000, 1000, 1000, 10]
ep   0 gain 1.000 trainlogcost 60.5095 PairwiseErrs: 5751, 5943, 6686
ep   1 gain 1.000 trainlogcost 27.5784 PairwiseErrs: 2747, 2603, 3159
ep   2 gain 1.000 trainlogcost 20.2401 PairwiseErrs: 2274, 1934, 2350
ep   3 gain 1.000 trainlogcost 15.6189 PairwiseErrs: 1863, 1533, 1833
ep   4 gain 1.000 trainlogcost 12.6109 PairwiseErrs: 1638, 1279, 1729
Energy-based errs: Train 477/10000 Valid 494/10000
Softmax-based errs: Valid 355/10000
rms:  0.0402, 0.0361, 0.0411
suprms:  0.0392, 0.0224
ep   5 gain 1.000 trainlogcost 11.0091 PairwiseErrs: 1466, 1137, 1441
ep   6 gain 1.000 trainlogcost 9.7446 PairwiseErrs: 1322, 991, 1243
ep   7 gain 1.000 trainlogcost 7.9477 PairwiseErrs: 1200, 850, 1096
ep   8 gain 1.000 trainlogcost 7.3157 PairwiseErrs: 1160, 815, 1117
ep   9 gain 1.000 trainlogcost 6.5421 PairwiseErrs: 1065, 703, 913
Energy-based errs: Train 225/10000 Valid 303/10000
Softmax-based errs: Valid 254/10000
rms:  0.0414, 0.0364, 0.0411
suprms:  0.0312, 0.0193
ep  10 gain 1.000 trainlogcost 6.5949 PairwiseErrs: 1077, 646, 974
ep  11 gain 1.000 trainlogcost 5.5925 PairwiseErrs: 980, 621, 843
ep  12 gain 1.000 trainlogcost 4.7399 PairwiseErrs: 896, 523, 748
ep  13 gain 1.000 trainlogcost 4.7420 PairwiseErrs: 827, 538, 729
ep  14 gain 1.000 trainlogcost 3.9755 PairwiseErrs: 782, 467, 635
Energy-based errs: Train 181/10000 Valid 278/10000
Softmax-based errs: Valid 265/10000
rms:  0.0423, 0.0364, 0.0409
suprms:  0.0264, 0.0193
ep  15 gain 1.000 trainlogcost 3.8626 PairwiseErrs: 769, 470, 655
ep  16 gain 1.000 trainlogcost 3.5369 PairwiseErrs: 710, 445, 636
ep  17 gain 1.000 trainlogcost 3.2951 PairwiseErrs: 696, 373, 618
ep  18 gain 1.000 trainlogcost 3.0723 PairwiseErrs: 636, 350, 487
ep  19 gain 1.000 trainlogcost 2.8362 PairwiseErrs: 636, 336, 472
Energy-based errs: Train 125/10000 Valid 245/10000
Softmax-based errs: Valid 224/10000
rms:  0.0427, 0.0360, 0.0403
suprms:  0.0231, 0.0192
ep  20 gain 1.000 trainlogcost 2.7015 PairwiseErrs: 637, 330, 507
ep  21 gain 1.000 trainlogcost 2.5557 PairwiseErrs: 586, 266, 458
ep  22 gain 1.000 trainlogcost 2.3118 PairwiseErrs: 543, 264, 427
ep  23 gain 1.000 trainlogcost 2.3662 PairwiseErrs: 531, 255, 435
ep  24 gain 1.000 trainlogcost 2.1204 PairwiseErrs: 564, 240, 420
Energy-based errs: Train 86/10000 Valid 206/10000
Softmax-based errs: Valid 200/10000
rms:  0.0430, 0.0354, 0.0397
suprms:  0.0213, 0.0193
ep  25 gain 1.000 trainlogcost 2.1646 PairwiseErrs: 541, 210, 361
ep  26 gain 1.000 trainlogcost 1.8379 PairwiseErrs: 487, 205, 372
ep  27 gain 1.000 trainlogcost 1.8013 PairwiseErrs: 476, 172, 351
ep  28 gain 1.000 trainlogcost 1.6856 PairwiseErrs: 451, 197, 314
ep  29 gain 1.000 trainlogcost 1.7014 PairwiseErrs: 480, 183, 330
Energy-based errs: Train 79/10000 Valid 240/10000
Softmax-based errs: Valid 199/10000
rms:  0.0433, 0.0348, 0.0390
suprms:  0.0198, 0.0190
ep  30 gain 1.000 trainlogcost 1.7938 PairwiseErrs: 462, 197, 319
ep  31 gain 1.000 trainlogcost 1.6134 PairwiseErrs: 478, 171, 309
ep  32 gain 1.000 trainlogcost 1.6390 PairwiseErrs: 443, 194, 296
ep  33 gain 1.000 trainlogcost 1.4127 PairwiseErrs: 436, 150, 283
ep  34 gain 1.000 trainlogcost 1.3936 PairwiseErrs: 425, 148, 256
Energy-based errs: Train 56/10000 Valid 190/10000
Softmax-based errs: Valid 178/10000
rms:  0.0437, 0.0342, 0.0381
suprms:  0.0188, 0.0187
ep  35 gain 1.000 trainlogcost 1.3840 PairwiseErrs: 406, 152, 253
ep  36 gain 1.000 trainlogcost 1.2820 PairwiseErrs: 399, 133, 232
ep  37 gain 1.000 trainlogcost 1.1741 PairwiseErrs: 384, 129, 211
ep  38 gain 1.000 trainlogcost 1.1226 PairwiseErrs: 337, 93, 201
ep  39 gain 1.000 trainlogcost 1.1488 PairwiseErrs: 334, 112, 221
Energy-based errs: Train 44/10000 Valid 193/10000
Softmax-based errs: Valid 181/10000
rms:  0.0436, 0.0335, 0.0373
suprms:  0.0180, 0.0184
ep  40 gain 1.000 trainlogcost 1.0721 PairwiseErrs: 349, 109, 194
ep  41 gain 1.000 trainlogcost 1.0098 PairwiseErrs: 323, 88, 191
ep  42 gain 1.000 trainlogcost 1.0324 PairwiseErrs: 363, 103, 191
ep  43 gain 1.000 trainlogcost 0.9932 PairwiseErrs: 335, 80, 177
ep  44 gain 1.000 trainlogcost 0.9876 PairwiseErrs: 315, 88, 189
Energy-based errs: Train 32/10000 Valid 202/10000
Softmax-based errs: Valid 188/10000
rms:  0.0439, 0.0326, 0.0364
suprms:  0.0173, 0.0180
ep  45 gain 1.000 trainlogcost 0.9672 PairwiseErrs: 315, 85, 151
ep  46 gain 1.000 trainlogcost 0.9009 PairwiseErrs: 327, 94, 172
ep  47 gain 1.000 trainlogcost 0.9328 PairwiseErrs: 317, 86, 165
ep  48 gain 1.000 trainlogcost 0.9956 PairwiseErrs: 323, 86, 165
ep  49 gain 1.000 trainlogcost 0.9351 PairwiseErrs: 334, 76, 182
Energy-based errs: Train 38/10000 Valid 190/10000
Softmax-based errs: Valid 199/10000
rms:  0.0443, 0.0318, 0.0356
suprms:  0.0168, 0.0177
ep  50 gain 1.010 trainlogcost 1.0094 PairwiseErrs: 324, 79, 164
ep  51 gain 0.990 trainlogcost 0.9677 PairwiseErrs: 311, 94, 160
ep  52 gain 0.970 trainlogcost 0.9396 PairwiseErrs: 297, 76, 140
ep  53 gain 0.950 trainlogcost 0.8401 PairwiseErrs: 291, 66, 152
ep  54 gain 0.930 trainlogcost 0.7494 PairwiseErrs: 281, 63, 140
Energy-based errs: Train 26/10000 Valid 176/10000
Softmax-based errs: Valid 187/10000
rms:  0.0443, 0.0311, 0.0349
suprms:  0.0165, 0.0174
ep  55 gain 0.910 trainlogcost 0.7487 PairwiseErrs: 266, 53, 125
ep  56 gain 0.890 trainlogcost 0.7583 PairwiseErrs: 250, 60, 100
ep  57 gain 0.870 trainlogcost 0.7688 PairwiseErrs: 271, 47, 123
ep  58 gain 0.850 trainlogcost 0.6808 PairwiseErrs: 255, 46, 105
ep  59 gain 0.830 trainlogcost 0.6411 PairwiseErrs: 242, 37, 88
Energy-based errs: Train 18/10000 Valid 184/10000
Softmax-based errs: Valid 196/10000
rms:  0.0443, 0.0303, 0.0341
suprms:  0.0160, 0.0170
ep  60 gain 0.810 trainlogcost 0.6846 PairwiseErrs: 244, 57, 94
ep  61 gain 0.790 trainlogcost 0.5892 PairwiseErrs: 222, 27, 91
ep  62 gain 0.770 trainlogcost 0.5244 PairwiseErrs: 196, 10, 70
ep  63 gain 0.750 trainlogcost 0.5238 PairwiseErrs: 192, 18, 63
ep  64 gain 0.730 trainlogcost 0.5124 PairwiseErrs: 178, 24, 59
Energy-based errs: Train 8/10000 Valid 170/10000
Softmax-based errs: Valid 166/10000
rms:  0.0437, 0.0295, 0.0332
suprms:  0.0153, 0.0165
ep  65 gain 0.710 trainlogcost 0.4724 PairwiseErrs: 169, 18, 42
ep  66 gain 0.690 trainlogcost 0.4768 PairwiseErrs: 159, 11, 34
ep  67 gain 0.670 trainlogcost 0.4781 PairwiseErrs: 159, 9, 32
ep  68 gain 0.650 trainlogcost 0.4652 PairwiseErrs: 160, 13, 40
ep  69 gain 0.630 trainlogcost 0.4662 PairwiseErrs: 135, 6, 27
Energy-based errs: Train 8/10000 Valid 178/10000
Softmax-based errs: Valid 164/10000
rms:  0.0430, 0.0286, 0.0323
suprms:  0.0150, 0.0163
ep  70 gain 0.610 trainlogcost 0.4366 PairwiseErrs: 142, 3, 27
ep  71 gain 0.590 trainlogcost 0.4239 PairwiseErrs: 130, 3, 26
ep  72 gain 0.570 trainlogcost 0.4434 PairwiseErrs: 138, 5, 29
ep  73 gain 0.550 trainlogcost 0.4032 PairwiseErrs: 111, 2, 12
ep  74 gain 0.530 trainlogcost 0.4190 PairwiseErrs: 126, 3, 17
Energy-based errs: Train 2/10000 Valid 172/10000
Softmax-based errs: Valid 172/10000
rms:  0.0421, 0.0279, 0.0316
suprms:  0.0148, 0.0163
ep  75 gain 0.510 trainlogcost 0.4234 PairwiseErrs: 98, 2, 18
ep  76 gain 0.490 trainlogcost 0.4077 PairwiseErrs: 116, 1, 10
ep  77 gain 0.470 trainlogcost 0.4052 PairwiseErrs: 94, 1, 8
ep  78 gain 0.450 trainlogcost 0.4024 PairwiseErrs: 104, 0, 11
ep  79 gain 0.430 trainlogcost 0.4168 PairwiseErrs: 99, 0, 9
Energy-based errs: Train 0/10000 Valid 156/10000
Softmax-based errs: Valid 161/10000
rms:  0.0414, 0.0273, 0.0309
suprms:  0.0148, 0.0163
ep  80 gain 0.410 trainlogcost 0.4147 PairwiseErrs: 84, 1, 10
ep  81 gain 0.390 trainlogcost 0.3921 PairwiseErrs: 80, 0, 6
ep  82 gain 0.370 trainlogcost 0.4050 PairwiseErrs: 77, 0, 5
ep  83 gain 0.350 trainlogcost 0.4082 PairwiseErrs: 83, 0, 3
ep  84 gain 0.330 trainlogcost 0.4115 PairwiseErrs: 69, 0, 4
Energy-based errs: Train 0/10000 Valid 145/10000
Softmax-based errs: Valid 158/10000
rms:  0.0408, 0.0268, 0.0304
suprms:  0.0149, 0.0164
ep  85 gain 0.310 trainlogcost 0.4290 PairwiseErrs: 66, 0, 3
ep  86 gain 0.290 trainlogcost 0.4268 PairwiseErrs: 63, 0, 0
ep  87 gain 0.270 trainlogcost 0.4270 PairwiseErrs: 62, 0, 3
ep  88 gain 0.250 trainlogcost 0.4237 PairwiseErrs: 55, 0, 3
ep  89 gain 0.230 trainlogcost 0.4360 PairwiseErrs: 52, 0, 2
Energy-based errs: Train 0/10000 Valid 159/10000
Softmax-based errs: Valid 161/10000
rms:  0.0403, 0.0265, 0.0300
suprms:  0.0151, 0.0167
ep  90 gain 0.210 trainlogcost 0.4278 PairwiseErrs: 53, 0, 1
ep  91 gain 0.190 trainlogcost 0.4400 PairwiseErrs: 55, 0, 1
ep  92 gain 0.170 trainlogcost 0.4450 PairwiseErrs: 51, 0, 0
ep  93 gain 0.150 trainlogcost 0.4442 PairwiseErrs: 44, 0, 0
ep  94 gain 0.130 trainlogcost 0.4497 PairwiseErrs: 44, 0, 0
Energy-based errs: Train 0/10000 Valid 155/10000
Softmax-based errs: Valid 160/10000
rms:  0.0400, 0.0263, 0.0298
suprms:  0.0152, 0.0168
ep  95 gain 0.110 trainlogcost 0.4453 PairwiseErrs: 45, 0, 1
ep  96 gain 0.090 trainlogcost 0.4451 PairwiseErrs: 49, 0, 0
ep  97 gain 0.070 trainlogcost 0.4485 PairwiseErrs: 46, 0, 0
ep  98 gain 0.050 trainlogcost 0.4512 PairwiseErrs: 47, 0, 1
ep  99 gain 0.030 trainlogcost 0.4492 PairwiseErrs: 38, 0, 0
Energy-based errs: Train 0/10000 Valid 155/10000
Softmax-based errs: Valid 161/10000
rms:  0.0399, 0.0262, 0.0297
suprms:  0.0153, 0.0169
Energy-based errs: Train 0/10000 Test 133/10000
Softmax-based errs: Train 0/10000 Test 138/10000

real	30m14.151s
user	24m15.381s
sys	4m37.789s

% sysctl -a | grep machdep.cpu
machdep.cpu.core_count: 8
machdep.cpu.brand_string: Apple M1
```
