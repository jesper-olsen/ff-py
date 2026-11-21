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
mkdir MNIST
cd MNIST
wget https://www.cs.toronto.edu/~hinton/mnistdata.mat
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

Install dependencies
```
% uv venv
% uv add numpy jax matplotlib
```

Train a model - run either main.py (numpy version) or main_jax.py.

```
time uv run main.py
Batchsize: 100 Input-dim: 784 #training batches: 500
states per layer:  [784, 1000, 1000, 1000, 10]
ep:   0 gain: 1.000 trainlogcost: 53.2679 PairwiseErrs: 5891, 5899, 6401
ep:   1 gain: 1.000 trainlogcost: 23.0378 PairwiseErrs: 3052, 2686, 3159
ep:   2 gain: 1.000 trainlogcost: 16.5152 PairwiseErrs: 2443, 2012, 2237
ep:   3 gain: 1.000 trainlogcost: 12.2872 PairwiseErrs: 2132, 1582, 1726
ep:   4 gain: 1.000 trainlogcost: 10.4937 PairwiseErrs: 1917, 1393, 1474
Energy-based errs; Train: 1449/50000 Valid: 321/10000
Softmax-based errs; Train: 1230/50000 Valid: 325/10000
rms:  0.0394, 0.0368, 0.0375
suprms:  0.0357, 0.0278
ep:   5 gain: 1.000 trainlogcost: 8.8690 PairwiseErrs: 1710, 1177, 1239
ep:   6 gain: 1.000 trainlogcost: 7.7001 PairwiseErrs: 1587, 1012, 1020
ep:   7 gain: 1.000 trainlogcost: 6.6266 PairwiseErrs: 1437, 899, 918
ep:   8 gain: 1.000 trainlogcost: 5.6745 PairwiseErrs: 1308, 805, 906
ep:   9 gain: 1.000 trainlogcost: 5.2133 PairwiseErrs: 1261, 738, 787
Energy-based errs; Train: 890/50000 Valid: 282/10000
Softmax-based errs; Train: 795/50000 Valid: 270/10000
rms:  0.0410, 0.0380, 0.0377
suprms:  0.0289, 0.0257
ep:  10 gain: 1.000 trainlogcost: 5.0586 PairwiseErrs: 1142, 668, 762
ep:  11 gain: 1.000 trainlogcost: 4.3251 PairwiseErrs: 1041, 575, 641
ep:  12 gain: 1.000 trainlogcost: 3.8009 PairwiseErrs: 963, 497, 532
ep:  13 gain: 1.000 trainlogcost: 3.5291 PairwiseErrs: 935, 459, 516
ep:  14 gain: 1.000 trainlogcost: 3.4239 PairwiseErrs: 905, 427, 502
Energy-based errs; Train: 515/50000 Valid: 227/10000
Softmax-based errs; Train: 596/50000 Valid: 252/10000
rms:  0.0420, 0.0386, 0.0376
suprms:  0.0252, 0.0247
ep:  15 gain: 1.000 trainlogcost: 3.3401 PairwiseErrs: 890, 437, 472
ep:  16 gain: 1.000 trainlogcost: 2.8632 PairwiseErrs: 871, 424, 444
ep:  17 gain: 1.000 trainlogcost: 2.7993 PairwiseErrs: 852, 367, 400
ep:  18 gain: 1.000 trainlogcost: 2.7726 PairwiseErrs: 819, 362, 370
ep:  19 gain: 1.000 trainlogcost: 2.4355 PairwiseErrs: 737, 310, 349
Energy-based errs; Train: 349/50000 Valid: 204/10000
Softmax-based errs; Train: 311/50000 Valid: 231/10000
rms:  0.0431, 0.0386, 0.0372
suprms:  0.0229, 0.0239
ep:  20 gain: 1.000 trainlogcost: 2.3134 PairwiseErrs: 723, 300, 313
ep:  21 gain: 1.000 trainlogcost: 2.1164 PairwiseErrs: 688, 297, 275
ep:  22 gain: 1.000 trainlogcost: 1.9044 PairwiseErrs: 678, 241, 247
ep:  23 gain: 1.000 trainlogcost: 1.8344 PairwiseErrs: 635, 249, 268
ep:  24 gain: 1.000 trainlogcost: 1.7432 PairwiseErrs: 611, 190, 252
Energy-based errs; Train: 265/50000 Valid: 196/10000
Softmax-based errs; Train: 244/50000 Valid: 215/10000
rms:  0.0436, 0.0384, 0.0365
suprms:  0.0211, 0.0227
ep:  25 gain: 1.000 trainlogcost: 1.6532 PairwiseErrs: 587, 210, 216
ep:  26 gain: 1.000 trainlogcost: 1.6263 PairwiseErrs: 550, 198, 223
ep:  27 gain: 1.000 trainlogcost: 1.5325 PairwiseErrs: 577, 186, 225
ep:  28 gain: 1.000 trainlogcost: 1.5462 PairwiseErrs: 537, 173, 193
ep:  29 gain: 1.000 trainlogcost: 1.3086 PairwiseErrs: 496, 166, 167
Energy-based errs; Train: 210/50000 Valid: 192/10000
Softmax-based errs; Train: 160/50000 Valid: 201/10000
rms:  0.0442, 0.0383, 0.0359
suprms:  0.0203, 0.0217
ep:  30 gain: 1.000 trainlogcost: 1.3262 PairwiseErrs: 512, 158, 168
ep:  31 gain: 1.000 trainlogcost: 1.4036 PairwiseErrs: 520, 165, 181
ep:  32 gain: 1.000 trainlogcost: 1.2195 PairwiseErrs: 494, 129, 157
ep:  33 gain: 1.000 trainlogcost: 1.2288 PairwiseErrs: 487, 127, 131
ep:  34 gain: 1.000 trainlogcost: 1.3302 PairwiseErrs: 482, 127, 130
Energy-based errs; Train: 168/50000 Valid: 193/10000
Softmax-based errs; Train: 109/50000 Valid: 201/10000
rms:  0.0450, 0.0378, 0.0351
suprms:  0.0196, 0.0210
ep:  35 gain: 1.000 trainlogcost: 1.2322 PairwiseErrs: 495, 120, 154
ep:  36 gain: 1.000 trainlogcost: 1.1427 PairwiseErrs: 499, 106, 127
ep:  37 gain: 1.000 trainlogcost: 1.1418 PairwiseErrs: 438, 111, 105
ep:  38 gain: 1.000 trainlogcost: 1.0288 PairwiseErrs: 433, 94, 98
ep:  39 gain: 1.000 trainlogcost: 1.0744 PairwiseErrs: 432, 112, 122
Energy-based errs; Train: 134/50000 Valid: 203/10000
Softmax-based errs; Train: 111/50000 Valid: 193/10000
rms:  0.0458, 0.0371, 0.0343
suprms:  0.0191, 0.0205
ep:  40 gain: 1.000 trainlogcost: 0.9785 PairwiseErrs: 414, 97, 102
ep:  41 gain: 1.000 trainlogcost: 0.9694 PairwiseErrs: 402, 83, 79
ep:  42 gain: 1.000 trainlogcost: 0.9574 PairwiseErrs: 389, 78, 99
ep:  43 gain: 1.000 trainlogcost: 0.9923 PairwiseErrs: 393, 70, 88
ep:  44 gain: 1.000 trainlogcost: 1.0723 PairwiseErrs: 401, 82, 103
Energy-based errs; Train: 131/50000 Valid: 200/10000
Softmax-based errs; Train: 184/50000 Valid: 219/10000
rms:  0.0467, 0.0365, 0.0335
suprms:  0.0188, 0.0200
ep:  45 gain: 1.000 trainlogcost: 1.0039 PairwiseErrs: 397, 73, 85
ep:  46 gain: 1.000 trainlogcost: 0.9430 PairwiseErrs: 389, 67, 78
ep:  47 gain: 1.000 trainlogcost: 0.9605 PairwiseErrs: 380, 62, 85
ep:  48 gain: 1.000 trainlogcost: 0.9579 PairwiseErrs: 343, 57, 74
ep:  49 gain: 1.000 trainlogcost: 0.9140 PairwiseErrs: 347, 55, 86
Energy-based errs; Train: 96/50000 Valid: 194/10000
Softmax-based errs; Train: 76/50000 Valid: 182/10000
rms:  0.0473, 0.0357, 0.0327
suprms:  0.0186, 0.0197
ep:  50 gain: 1.000 trainlogcost: 0.8773 PairwiseErrs: 310, 53, 80
ep:  51 gain: 1.000 trainlogcost: 0.9449 PairwiseErrs: 350, 72, 70
ep:  52 gain: 1.000 trainlogcost: 0.9167 PairwiseErrs: 341, 70, 84
ep:  53 gain: 1.000 trainlogcost: 0.9840 PairwiseErrs: 312, 89, 96
ep:  54 gain: 1.000 trainlogcost: 0.8792 PairwiseErrs: 306, 57, 84
Energy-based errs; Train: 68/50000 Valid: 188/10000
Softmax-based errs; Train: 71/50000 Valid: 191/10000
rms:  0.0478, 0.0353, 0.0320
suprms:  0.0187, 0.0194
ep:  55 gain: 1.000 trainlogcost: 0.8352 PairwiseErrs: 291, 42, 64
ep:  56 gain: 1.000 trainlogcost: 0.8711 PairwiseErrs: 282, 46, 69
ep:  57 gain: 1.000 trainlogcost: 0.7977 PairwiseErrs: 262, 46, 55
ep:  58 gain: 1.000 trainlogcost: 0.8360 PairwiseErrs: 277, 39, 60
ep:  59 gain: 1.000 trainlogcost: 0.7900 PairwiseErrs: 254, 36, 56
Energy-based errs; Train: 51/50000 Valid: 169/10000
Softmax-based errs; Train: 40/50000 Valid: 194/10000
rms:  0.0475, 0.0343, 0.0312
suprms:  0.0186, 0.0190
ep:  60 gain: 1.000 trainlogcost: 0.7731 PairwiseErrs: 278, 45, 59
ep:  61 gain: 1.000 trainlogcost: 0.7630 PairwiseErrs: 245, 44, 54
ep:  62 gain: 1.000 trainlogcost: 0.7158 PairwiseErrs: 252, 29, 60
ep:  63 gain: 1.000 trainlogcost: 0.7074 PairwiseErrs: 236, 27, 39
ep:  64 gain: 0.984 trainlogcost: 0.7384 PairwiseErrs: 237, 22, 49
Energy-based errs; Train: 71/50000 Valid: 187/10000
Softmax-based errs; Train: 45/50000 Valid: 206/10000
rms:  0.0476, 0.0335, 0.0305
suprms:  0.0182, 0.0186
ep:  65 gain: 0.968 trainlogcost: 0.7048 PairwiseErrs: 242, 30, 53
ep:  66 gain: 0.952 trainlogcost: 0.6494 PairwiseErrs: 267, 29, 27
ep:  67 gain: 0.936 trainlogcost: 0.6541 PairwiseErrs: 235, 21, 39
ep:  68 gain: 0.920 trainlogcost: 0.6557 PairwiseErrs: 246, 24, 50
ep:  69 gain: 0.904 trainlogcost: 0.7053 PairwiseErrs: 228, 31, 27
Energy-based errs; Train: 31/50000 Valid: 170/10000
Softmax-based errs; Train: 26/50000 Valid: 195/10000
rms:  0.0478, 0.0327, 0.0296
suprms:  0.0177, 0.0183
ep:  70 gain: 0.888 trainlogcost: 0.6877 PairwiseErrs: 239, 42, 45
ep:  71 gain: 0.872 trainlogcost: 0.6675 PairwiseErrs: 249, 31, 29
ep:  72 gain: 0.856 trainlogcost: 0.5723 PairwiseErrs: 219, 23, 24
ep:  73 gain: 0.840 trainlogcost: 0.5935 PairwiseErrs: 182, 11, 12
ep:  74 gain: 0.824 trainlogcost: 0.5850 PairwiseErrs: 187, 13, 17
Energy-based errs; Train: 11/50000 Valid: 156/10000
Softmax-based errs; Train: 11/50000 Valid: 173/10000
rms:  0.0477, 0.0318, 0.0288
suprms:  0.0175, 0.0179
ep:  75 gain: 0.808 trainlogcost: 0.5661 PairwiseErrs: 193, 12, 21
ep:  76 gain: 0.792 trainlogcost: 0.5261 PairwiseErrs: 167, 3, 16
ep:  77 gain: 0.776 trainlogcost: 0.5305 PairwiseErrs: 169, 5, 14
ep:  78 gain: 0.760 trainlogcost: 0.5069 PairwiseErrs: 153, 5, 13
ep:  79 gain: 0.744 trainlogcost: 0.5144 PairwiseErrs: 163, 7, 13
Energy-based errs; Train: 7/50000 Valid: 173/10000
Softmax-based errs; Train: 7/50000 Valid: 178/10000
rms:  0.0468, 0.0307, 0.0279
suprms:  0.0173, 0.0175
ep:  80 gain: 0.728 trainlogcost: 0.5001 PairwiseErrs: 165, 3, 7
ep:  81 gain: 0.712 trainlogcost: 0.4961 PairwiseErrs: 141, 5, 5
ep:  82 gain: 0.696 trainlogcost: 0.4867 PairwiseErrs: 132, 5, 8
ep:  83 gain: 0.680 trainlogcost: 0.4646 PairwiseErrs: 142, 2, 6
ep:  84 gain: 0.664 trainlogcost: 0.4753 PairwiseErrs: 139, 3, 7
Energy-based errs; Train: 11/50000 Valid: 155/10000
Softmax-based errs; Train: 15/50000 Valid: 178/10000
rms:  0.0461, 0.0297, 0.0271
suprms:  0.0169, 0.0172
ep:  85 gain: 0.648 trainlogcost: 0.4659 PairwiseErrs: 121, 4, 10
ep:  86 gain: 0.632 trainlogcost: 0.4771 PairwiseErrs: 112, 3, 3
ep:  87 gain: 0.616 trainlogcost: 0.4617 PairwiseErrs: 100, 2, 5
ep:  88 gain: 0.600 trainlogcost: 0.4553 PairwiseErrs: 94, 1, 2
ep:  89 gain: 0.584 trainlogcost: 0.4345 PairwiseErrs: 89, 0, 2
Energy-based errs; Train: 1/50000 Valid: 163/10000
Softmax-based errs; Train: 5/50000 Valid: 173/10000
rms:  0.0452, 0.0288, 0.0264
suprms:  0.0168, 0.0170
ep:  90 gain: 0.568 trainlogcost: 0.4350 PairwiseErrs: 91, 1, 7
ep:  91 gain: 0.552 trainlogcost: 0.4264 PairwiseErrs: 81, 0, 1
ep:  92 gain: 0.536 trainlogcost: 0.4316 PairwiseErrs: 93, 1, 1
ep:  93 gain: 0.520 trainlogcost: 0.4349 PairwiseErrs: 82, 0, 3
ep:  94 gain: 0.504 trainlogcost: 0.4251 PairwiseErrs: 75, 2, 1
Energy-based errs; Train: 1/50000 Valid: 153/10000
Softmax-based errs; Train: 4/50000 Valid: 168/10000
rms:  0.0444, 0.0280, 0.0257
suprms:  0.0166, 0.0168
ep:  95 gain: 0.488 trainlogcost: 0.4152 PairwiseErrs: 71, 1, 1
ep:  96 gain: 0.472 trainlogcost: 0.4160 PairwiseErrs: 79, 0, 1
ep:  97 gain: 0.456 trainlogcost: 0.4310 PairwiseErrs: 77, 1, 2
ep:  98 gain: 0.440 trainlogcost: 0.4138 PairwiseErrs: 62, 1, 0
ep:  99 gain: 0.424 trainlogcost: 0.4078 PairwiseErrs: 62, 0, 0
Energy-based errs; Train: 1/50000 Valid: 159/10000
Softmax-based errs; Train: 2/50000 Valid: 176/10000
rms:  0.0435, 0.0273, 0.0251
suprms:  0.0165, 0.0167
ep: 100 gain: 0.408 trainlogcost: 0.4130 PairwiseErrs: 57, 0, 1
ep: 101 gain: 0.392 trainlogcost: 0.4058 PairwiseErrs: 48, 0, 0
ep: 102 gain: 0.376 trainlogcost: 0.4049 PairwiseErrs: 60, 0, 0
ep: 103 gain: 0.360 trainlogcost: 0.4019 PairwiseErrs: 52, 0, 0
ep: 104 gain: 0.344 trainlogcost: 0.4115 PairwiseErrs: 49, 0, 0
Energy-based errs; Train: 0/50000 Valid: 152/10000
Softmax-based errs; Train: 2/50000 Valid: 171/10000
rms:  0.0428, 0.0267, 0.0247
suprms:  0.0164, 0.0166
ep: 105 gain: 0.328 trainlogcost: 0.4073 PairwiseErrs: 48, 0, 0
ep: 106 gain: 0.312 trainlogcost: 0.4086 PairwiseErrs: 58, 0, 0
ep: 107 gain: 0.296 trainlogcost: 0.4067 PairwiseErrs: 44, 1, 1
ep: 108 gain: 0.280 trainlogcost: 0.4022 PairwiseErrs: 42, 0, 0
ep: 109 gain: 0.264 trainlogcost: 0.4040 PairwiseErrs: 44, 0, 0
Energy-based errs; Train: 0/50000 Valid: 149/10000
Softmax-based errs; Train: 1/50000 Valid: 173/10000
rms:  0.0422, 0.0263, 0.0243
suprms:  0.0163, 0.0166
ep: 110 gain: 0.248 trainlogcost: 0.4036 PairwiseErrs: 33, 0, 0
ep: 111 gain: 0.232 trainlogcost: 0.4024 PairwiseErrs: 39, 0, 0
ep: 112 gain: 0.216 trainlogcost: 0.4032 PairwiseErrs: 34, 0, 0
ep: 113 gain: 0.200 trainlogcost: 0.4028 PairwiseErrs: 30, 0, 0
ep: 114 gain: 0.184 trainlogcost: 0.4023 PairwiseErrs: 34, 0, 0
Energy-based errs; Train: 1/50000 Valid: 148/10000
Softmax-based errs; Train: 1/50000 Valid: 173/10000
rms:  0.0418, 0.0260, 0.0240
suprms:  0.0163, 0.0166
ep: 115 gain: 0.168 trainlogcost: 0.4048 PairwiseErrs: 27, 0, 1
ep: 116 gain: 0.152 trainlogcost: 0.4044 PairwiseErrs: 25, 0, 0
ep: 117 gain: 0.136 trainlogcost: 0.4033 PairwiseErrs: 26, 0, 0
ep: 118 gain: 0.120 trainlogcost: 0.4026 PairwiseErrs: 31, 0, 0
ep: 119 gain: 0.104 trainlogcost: 0.4025 PairwiseErrs: 28, 0, 0
Energy-based errs; Train: 0/50000 Valid: 154/10000
Softmax-based errs; Train: 1/50000 Valid: 171/10000
rms:  0.0416, 0.0258, 0.0239
suprms:  0.0163, 0.0165
ep: 120 gain: 0.088 trainlogcost: 0.4041 PairwiseErrs: 15, 0, 0
ep: 121 gain: 0.072 trainlogcost: 0.4017 PairwiseErrs: 21, 0, 0
ep: 122 gain: 0.056 trainlogcost: 0.4014 PairwiseErrs: 21, 0, 0
ep: 123 gain: 0.040 trainlogcost: 0.4001 PairwiseErrs: 25, 0, 0
ep: 124 gain: 0.024 trainlogcost: 0.3999 PairwiseErrs: 16, 0, 0
Energy-based errs; Train: 0/50000 Valid: 154/10000
Softmax-based errs; Train: 1/50000 Valid: 168/10000
rms:  0.0414, 0.0257, 0.0238
suprms:  0.0163, 0.0165
Energy-based errs: Train 0/50000 Test 144/10000
Softmax-based errs: Train 1/50000 Test 153/10000
1390.03s user 10.61s system 98% cpu 23:36.96 total
```

```
% sysctl -a | grep machdep.cpu
machdep.cpu.cores_per_package: 10
machdep.cpu.core_count: 10
machdep.cpu.logical_per_package: 10
machdep.cpu.thread_count: 10
machdep.cpu.brand_string: Apple M5
```
