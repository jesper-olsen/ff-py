Forward-Forward
===============

The Forward-Forward algorithm [1,2] evaluated on the MNIST handwritten digit recognition task; Python implementation following [3]. Two versions are implemented: one with [numpy](https://numpy.org/) and one with [jax](https://github.com/google/jax).

Network: 5-layers; input layer with 784 (28x28) states, 3 hidden layers with 1000 states each and an output layer with 10 states corresponding to the 10 digits.

Data partitioning: Train on 50k samples, validate on 10k, test on 10k.

Error rate (energy / softmax):
* Hinton's matlab code [2]: 1.44% / 1.47%  
* This repo (numpy): 1.42% / 1.44%       
* This repo (jax) :1.44% / 1.49%

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
Energy-based errs; Train: 306/10000 Valid: 323/10000
Softmax-based errs; Train: 265/10000 Valid: 325/10000
rms:  0.0394, 0.0368, 0.0375
suprms:  0.0357, 0.0278
ep:   5 gain: 1.000 trainlogcost: 8.8690 PairwiseErrs: 1710, 1177, 1239
ep:   6 gain: 1.000 trainlogcost: 7.7001 PairwiseErrs: 1587, 1012, 1020
ep:   7 gain: 1.000 trainlogcost: 6.6266 PairwiseErrs: 1437, 899, 918
ep:   8 gain: 1.000 trainlogcost: 5.6745 PairwiseErrs: 1308, 805, 906
ep:   9 gain: 1.000 trainlogcost: 5.2133 PairwiseErrs: 1261, 738, 787
Energy-based errs; Train: 199/10000 Valid: 285/10000
Softmax-based errs; Train: 171/10000 Valid: 270/10000
rms:  0.0410, 0.0380, 0.0377
suprms:  0.0289, 0.0257
ep:  10 gain: 1.000 trainlogcost: 5.0586 PairwiseErrs: 1142, 668, 762
ep:  11 gain: 1.000 trainlogcost: 4.3251 PairwiseErrs: 1041, 575, 641
ep:  12 gain: 1.000 trainlogcost: 3.8009 PairwiseErrs: 963, 497, 532
ep:  13 gain: 1.000 trainlogcost: 3.5291 PairwiseErrs: 935, 459, 516
ep:  14 gain: 1.000 trainlogcost: 3.4239 PairwiseErrs: 905, 427, 502
Energy-based errs; Train: 114/10000 Valid: 228/10000
Softmax-based errs; Train: 133/10000 Valid: 252/10000
rms:  0.0420, 0.0386, 0.0376
suprms:  0.0252, 0.0247
ep:  15 gain: 1.000 trainlogcost: 3.3401 PairwiseErrs: 890, 437, 472
ep:  16 gain: 1.000 trainlogcost: 2.8632 PairwiseErrs: 871, 424, 444
ep:  17 gain: 1.000 trainlogcost: 2.7993 PairwiseErrs: 852, 367, 400
ep:  18 gain: 1.000 trainlogcost: 2.7726 PairwiseErrs: 819, 362, 370
ep:  19 gain: 1.000 trainlogcost: 2.4355 PairwiseErrs: 737, 310, 349
Energy-based errs; Train: 74/10000 Valid: 203/10000
Softmax-based errs; Train: 60/10000 Valid: 231/10000
rms:  0.0431, 0.0386, 0.0372
suprms:  0.0229, 0.0239
ep:  20 gain: 1.000 trainlogcost: 2.3134 PairwiseErrs: 723, 300, 313
ep:  21 gain: 1.000 trainlogcost: 2.1164 PairwiseErrs: 688, 297, 275
ep:  22 gain: 1.000 trainlogcost: 1.9044 PairwiseErrs: 678, 241, 247
ep:  23 gain: 1.000 trainlogcost: 1.8344 PairwiseErrs: 635, 249, 268
ep:  24 gain: 1.000 trainlogcost: 1.7432 PairwiseErrs: 611, 190, 252
Energy-based errs; Train: 65/10000 Valid: 195/10000
Softmax-based errs; Train: 47/10000 Valid: 215/10000
rms:  0.0436, 0.0384, 0.0365
suprms:  0.0211, 0.0227
ep:  25 gain: 1.000 trainlogcost: 1.6532 PairwiseErrs: 587, 210, 216
ep:  26 gain: 1.000 trainlogcost: 1.6263 PairwiseErrs: 550, 198, 223
ep:  27 gain: 1.000 trainlogcost: 1.5325 PairwiseErrs: 577, 186, 225
ep:  28 gain: 1.000 trainlogcost: 1.5462 PairwiseErrs: 537, 173, 193
ep:  29 gain: 1.000 trainlogcost: 1.3086 PairwiseErrs: 496, 166, 167
Energy-based errs; Train: 50/10000 Valid: 193/10000
Softmax-based errs; Train: 38/10000 Valid: 201/10000
rms:  0.0442, 0.0383, 0.0359
suprms:  0.0203, 0.0217
ep:  30 gain: 1.000 trainlogcost: 1.3262 PairwiseErrs: 512, 158, 168
ep:  31 gain: 1.000 trainlogcost: 1.4036 PairwiseErrs: 520, 165, 181
ep:  32 gain: 1.000 trainlogcost: 1.2195 PairwiseErrs: 494, 129, 157
ep:  33 gain: 1.000 trainlogcost: 1.2288 PairwiseErrs: 487, 127, 131
ep:  34 gain: 1.000 trainlogcost: 1.3302 PairwiseErrs: 482, 127, 130
Energy-based errs; Train: 33/10000 Valid: 191/10000
Softmax-based errs; Train: 18/10000 Valid: 201/10000
rms:  0.0450, 0.0378, 0.0351
suprms:  0.0196, 0.0210
ep:  35 gain: 1.000 trainlogcost: 1.2322 PairwiseErrs: 495, 120, 154
ep:  36 gain: 1.000 trainlogcost: 1.1427 PairwiseErrs: 499, 106, 127
ep:  37 gain: 1.000 trainlogcost: 1.1418 PairwiseErrs: 438, 111, 105
ep:  38 gain: 1.000 trainlogcost: 1.0288 PairwiseErrs: 433, 94, 98
ep:  39 gain: 1.000 trainlogcost: 1.0744 PairwiseErrs: 432, 112, 122
Energy-based errs; Train: 25/10000 Valid: 203/10000
Softmax-based errs; Train: 25/10000 Valid: 193/10000
rms:  0.0458, 0.0371, 0.0343
suprms:  0.0191, 0.0205
ep:  40 gain: 1.000 trainlogcost: 0.9785 PairwiseErrs: 414, 97, 102
ep:  41 gain: 1.000 trainlogcost: 0.9694 PairwiseErrs: 402, 83, 79
ep:  42 gain: 1.000 trainlogcost: 0.9574 PairwiseErrs: 389, 78, 99
ep:  43 gain: 1.000 trainlogcost: 0.9923 PairwiseErrs: 393, 70, 88
ep:  44 gain: 1.000 trainlogcost: 1.0723 PairwiseErrs: 401, 82, 103
Energy-based errs; Train: 27/10000 Valid: 200/10000
Softmax-based errs; Train: 41/10000 Valid: 219/10000
rms:  0.0467, 0.0365, 0.0335
suprms:  0.0188, 0.0200
ep:  45 gain: 1.000 trainlogcost: 1.0039 PairwiseErrs: 397, 73, 85
ep:  46 gain: 1.000 trainlogcost: 0.9430 PairwiseErrs: 389, 67, 78
ep:  47 gain: 1.000 trainlogcost: 0.9605 PairwiseErrs: 380, 62, 85
ep:  48 gain: 1.000 trainlogcost: 0.9579 PairwiseErrs: 343, 57, 74
ep:  49 gain: 1.000 trainlogcost: 0.9140 PairwiseErrs: 347, 55, 86
Energy-based errs; Train: 15/10000 Valid: 195/10000
Softmax-based errs; Train: 15/10000 Valid: 182/10000
rms:  0.0473, 0.0357, 0.0327
suprms:  0.0186, 0.0197
ep:  50 gain: 1.010 trainlogcost: 0.9164 PairwiseErrs: 326, 46, 75
ep:  51 gain: 0.990 trainlogcost: 0.8952 PairwiseErrs: 328, 70, 75
ep:  52 gain: 0.970 trainlogcost: 0.8882 PairwiseErrs: 344, 46, 79
ep:  53 gain: 0.950 trainlogcost: 0.9288 PairwiseErrs: 270, 56, 83
ep:  54 gain: 0.930 trainlogcost: 0.8421 PairwiseErrs: 326, 51, 67
Energy-based errs; Train: 19/10000 Valid: 178/10000
Softmax-based errs; Train: 14/10000 Valid: 183/10000
rms:  0.0475, 0.0350, 0.0319
suprms:  0.0185, 0.0194
ep:  55 gain: 0.910 trainlogcost: 0.7765 PairwiseErrs: 285, 35, 52
ep:  56 gain: 0.890 trainlogcost: 0.7214 PairwiseErrs: 292, 33, 32
ep:  57 gain: 0.870 trainlogcost: 0.6578 PairwiseErrs: 236, 26, 25
ep:  58 gain: 0.850 trainlogcost: 0.6438 PairwiseErrs: 247, 24, 28
ep:  59 gain: 0.830 trainlogcost: 0.6196 PairwiseErrs: 237, 26, 24
Energy-based errs; Train: 10/10000 Valid: 157/10000
Softmax-based errs; Train: 5/10000 Valid: 190/10000
rms:  0.0469, 0.0339, 0.0309
suprms:  0.0178, 0.0185
ep:  60 gain: 0.810 trainlogcost: 0.6020 PairwiseErrs: 233, 13, 38
ep:  61 gain: 0.790 trainlogcost: 0.5828 PairwiseErrs: 215, 13, 24
ep:  62 gain: 0.770 trainlogcost: 0.5686 PairwiseErrs: 187, 8, 17
ep:  63 gain: 0.750 trainlogcost: 0.5707 PairwiseErrs: 185, 12, 13
ep:  64 gain: 0.730 trainlogcost: 0.5251 PairwiseErrs: 199, 4, 21
Energy-based errs; Train: 2/10000 Valid: 159/10000
Softmax-based errs; Train: 1/10000 Valid: 183/10000
rms:  0.0463, 0.0327, 0.0300
suprms:  0.0173, 0.0179
ep:  65 gain: 0.710 trainlogcost: 0.5369 PairwiseErrs: 182, 4, 8
ep:  66 gain: 0.690 trainlogcost: 0.5152 PairwiseErrs: 161, 6, 14
ep:  67 gain: 0.670 trainlogcost: 0.5104 PairwiseErrs: 144, 2, 8
ep:  68 gain: 0.650 trainlogcost: 0.5064 PairwiseErrs: 177, 2, 6
ep:  69 gain: 0.630 trainlogcost: 0.4941 PairwiseErrs: 158, 2, 9
Energy-based errs; Train: 1/10000 Valid: 164/10000
Softmax-based errs; Train: 4/10000 Valid: 176/10000
rms:  0.0456, 0.0317, 0.0291
suprms:  0.0170, 0.0176
ep:  70 gain: 0.610 trainlogcost: 0.4777 PairwiseErrs: 152, 2, 2
ep:  71 gain: 0.590 trainlogcost: 0.4963 PairwiseErrs: 171, 4, 5
ep:  72 gain: 0.570 trainlogcost: 0.4907 PairwiseErrs: 165, 4, 2
ep:  73 gain: 0.550 trainlogcost: 0.4690 PairwiseErrs: 146, 0, 2
ep:  74 gain: 0.530 trainlogcost: 0.4541 PairwiseErrs: 148, 2, 5
Energy-based errs; Train: 0/10000 Valid: 161/10000
Softmax-based errs; Train: 2/10000 Valid: 184/10000
rms:  0.0449, 0.0307, 0.0283
suprms:  0.0167, 0.0173
ep:  75 gain: 0.510 trainlogcost: 0.4493 PairwiseErrs: 123, 0, 1
ep:  76 gain: 0.490 trainlogcost: 0.4515 PairwiseErrs: 125, 1, 5
ep:  77 gain: 0.470 trainlogcost: 0.4368 PairwiseErrs: 97, 0, 1
ep:  78 gain: 0.450 trainlogcost: 0.4484 PairwiseErrs: 95, 0, 2
ep:  79 gain: 0.430 trainlogcost: 0.4425 PairwiseErrs: 100, 0, 0
Energy-based errs; Train: 0/10000 Valid: 165/10000
Softmax-based errs; Train: 1/10000 Valid: 178/10000
rms:  0.0441, 0.0300, 0.0277
suprms:  0.0166, 0.0171
ep:  80 gain: 0.410 trainlogcost: 0.4410 PairwiseErrs: 91, 0, 0
ep:  81 gain: 0.390 trainlogcost: 0.4353 PairwiseErrs: 94, 0, 0
ep:  82 gain: 0.370 trainlogcost: 0.4339 PairwiseErrs: 80, 0, 0
ep:  83 gain: 0.350 trainlogcost: 0.4249 PairwiseErrs: 84, 0, 0
ep:  84 gain: 0.330 trainlogcost: 0.4289 PairwiseErrs: 88, 0, 0
Energy-based errs; Train: 0/10000 Valid: 164/10000
Softmax-based errs; Train: 1/10000 Valid: 185/10000
rms:  0.0434, 0.0294, 0.0272
suprms:  0.0165, 0.0170
ep:  85 gain: 0.310 trainlogcost: 0.4277 PairwiseErrs: 60, 0, 0
ep:  86 gain: 0.290 trainlogcost: 0.4222 PairwiseErrs: 72, 1, 1
ep:  87 gain: 0.270 trainlogcost: 0.4264 PairwiseErrs: 68, 0, 0
ep:  88 gain: 0.250 trainlogcost: 0.4255 PairwiseErrs: 71, 0, 0
ep:  89 gain: 0.230 trainlogcost: 0.4221 PairwiseErrs: 64, 0, 0
Energy-based errs; Train: 0/10000 Valid: 166/10000
Softmax-based errs; Train: 1/10000 Valid: 176/10000
rms:  0.0429, 0.0290, 0.0268
suprms:  0.0164, 0.0170
ep:  90 gain: 0.210 trainlogcost: 0.4231 PairwiseErrs: 61, 0, 0
ep:  91 gain: 0.190 trainlogcost: 0.4176 PairwiseErrs: 61, 0, 0
ep:  92 gain: 0.170 trainlogcost: 0.4217 PairwiseErrs: 65, 0, 0
ep:  93 gain: 0.150 trainlogcost: 0.4226 PairwiseErrs: 58, 0, 0
ep:  94 gain: 0.130 trainlogcost: 0.4212 PairwiseErrs: 62, 1, 0
Energy-based errs; Train: 0/10000 Valid: 164/10000
Softmax-based errs; Train: 1/10000 Valid: 177/10000
rms:  0.0426, 0.0287, 0.0266
suprms:  0.0164, 0.0169
ep:  95 gain: 0.110 trainlogcost: 0.4201 PairwiseErrs: 58, 0, 0
ep:  96 gain: 0.090 trainlogcost: 0.4200 PairwiseErrs: 63, 0, 0
ep:  97 gain: 0.070 trainlogcost: 0.4202 PairwiseErrs: 55, 0, 0
ep:  98 gain: 0.050 trainlogcost: 0.4203 PairwiseErrs: 47, 0, 0
ep:  99 gain: 0.030 trainlogcost: 0.4195 PairwiseErrs: 56, 0, 0
Energy-based errs; Train: 0/10000 Valid: 160/10000
Softmax-based errs; Train: 1/10000 Valid: 175/10000
rms:  0.0424, 0.0286, 0.0265
suprms:  0.0164, 0.0169
Energy-based errs: Train 0/50000 Test 142/10000
Softmax-based errs: Train 1/50000 Test 144/10000
830.67s user 4.85s system 99% cpu 14:01.05 total
```

```
% sysctl -a | grep machdep.cpu
machdep.cpu.cores_per_package: 10
machdep.cpu.core_count: 10
machdep.cpu.logical_per_package: 10
machdep.cpu.thread_count: 10
machdep.cpu.brand_string: Apple M5
```
