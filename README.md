Forward-Forward
===============

The Forward-Forward algorithm [1,5] evaluated on the MNIST handwritten digit recognition task; Python+Numpy implementation following [2].

Accuracy (energy / softmax) - train on 50k samples, validate on 10k, test on 10k:
* Hinton's matlab code [2]: 1.44% / 1.47%  
* The repo: 1.35% / 1.41%       

Network: 5-layers; input layer with 784 (28x28) states, 3 hidden layers with 1000 states each and an output layer with 10 states corresponding to the 10 digits.

References:
-----------
[1] [The Forward-Forward Algorithm: Some Preliminary Investigations, Geoffrey Hinton, NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf) <br/>
[2] [Hinton's matlab code](https://www.cs.toronto.edu/~hinton/ffcode.zip) <br/>
[3] [Hinton's preprocessed MNIST db](https://www.cs.toronto.edu/~hinton/mnistdata.mat) <br/>
[4] [LeCun's raw MNIST db](http://yann.lecun.com/exdb/mnist/)
[5] [Hinton's NIPS'22 Talk](https://nips.cc/virtual/2022/invited-talk/55869)

Run:
----

Download MNIST - either [4] or [3]; Edit mnist.py if [3]:   
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
% python main.py
Batchsize: 100 Input-dim: 784 #training batches: 500
states per layer:  [784, 1000, 1000, 1000, 10]
ep   0 gain 1.000 trainlogcost 71.0193 PairwiseErrs: [5770, 6033, 6835]
ep   1 gain 1.000 trainlogcost 31.5669 PairwiseErrs: [2816, 2763, 3405]
ep   2 gain 1.000 trainlogcost 21.5774 PairwiseErrs: [2226, 1939, 2541]
ep   3 gain 1.000 trainlogcost 16.3414 PairwiseErrs: [1869, 1515, 1932]
ep   4 gain 1.000 trainlogcost 14.5586 PairwiseErrs: [1689, 1421, 1784]
Energy-based errs: Train 405/10000 Valid 417/10000
Softmax-based errs: Valid 353/10000
rms:  [0.04103313177492559, 0.03802084684058512, 0.0388355843523133]
suprms:  [0.03821720460656551, 0.02653125402807066]
ep   5 gain 1.000 trainlogcost 11.5735 PairwiseErrs: [1420, 1140, 1475]
ep   6 gain 1.000 trainlogcost 9.8649 PairwiseErrs: [1300, 1013, 1229]
ep   7 gain 1.000 trainlogcost 8.5639 PairwiseErrs: [1206, 849, 1121]
ep   8 gain 1.000 trainlogcost 7.8229 PairwiseErrs: [1126, 794, 1122]
ep   9 gain 1.000 trainlogcost 6.8767 PairwiseErrs: [1060, 772, 864]
Energy-based errs: Train 249/10000 Valid 317/10000
Softmax-based errs: Valid 286/10000
rms:  [0.043160542627129005, 0.038768993492288524, 0.039859212897436276]
suprms:  [0.03051844495004876, 0.021642455940532816]
ep  10 gain 1.000 trainlogcost 6.0323 PairwiseErrs: [966, 641, 894]
ep  11 gain 1.000 trainlogcost 5.8639 PairwiseErrs: [982, 619, 941]
ep  12 gain 1.000 trainlogcost 5.3601 PairwiseErrs: [918, 596, 875]
ep  13 gain 1.000 trainlogcost 4.5976 PairwiseErrs: [793, 494, 770]
ep  14 gain 1.000 trainlogcost 4.4191 PairwiseErrs: [752, 490, 787]
Energy-based errs: Train 227/10000 Valid 328/10000
Softmax-based errs: Valid 262/10000
rms:  [0.044602099732984304, 0.03928634788334539, 0.04065815806121992]
suprms:  [0.025717125532100236, 0.020897613097256892]
ep  15 gain 1.000 trainlogcost 3.8498 PairwiseErrs: [687, 418, 657]
ep  16 gain 1.000 trainlogcost 3.9697 PairwiseErrs: [704, 400, 620]
ep  17 gain 1.000 trainlogcost 3.4528 PairwiseErrs: [687, 393, 612]
ep  18 gain 1.000 trainlogcost 3.1833 PairwiseErrs: [643, 373, 523]
ep  19 gain 1.000 trainlogcost 2.9159 PairwiseErrs: [589, 336, 546]
Energy-based errs: Train 83/10000 Valid 218/10000
Softmax-based errs: Valid 223/10000
rms:  [0.04580820308336519, 0.03944470040919049, 0.040883648456736046]
suprms:  [0.022894827649744317, 0.02052012424786559]
ep  20 gain 1.000 trainlogcost 2.5784 PairwiseErrs: [585, 266, 489]
ep  21 gain 1.000 trainlogcost 2.5366 PairwiseErrs: [545, 272, 489]
ep  22 gain 1.000 trainlogcost 2.3529 PairwiseErrs: [534, 260, 488]
ep  23 gain 1.000 trainlogcost 2.5341 PairwiseErrs: [537, 256, 444]
ep  24 gain 1.000 trainlogcost 2.5583 PairwiseErrs: [562, 260, 462]
Energy-based errs: Train 80/10000 Valid 219/10000
Softmax-based errs: Valid 228/10000
rms:  [0.047279971036122075, 0.03939289937805696, 0.04106858818093196]
suprms:  [0.021380465238887765, 0.02035880845396274]
ep  25 gain 1.000 trainlogcost 2.7297 PairwiseErrs: [578, 276, 455]
ep  26 gain 1.000 trainlogcost 2.1602 PairwiseErrs: [518, 232, 411]
ep  27 gain 1.000 trainlogcost 2.0054 PairwiseErrs: [459, 209, 385]
ep  28 gain 1.000 trainlogcost 1.8128 PairwiseErrs: [436, 212, 439]
ep  29 gain 1.000 trainlogcost 1.6609 PairwiseErrs: [410, 175, 359]
Energy-based errs: Train 66/10000 Valid 228/10000
Softmax-based errs: Valid 219/10000
rms:  [0.04837829696802028, 0.03936317731703073, 0.041142123598290656]
suprms:  [0.019940684804950818, 0.019556054867374675]
ep  30 gain 1.000 trainlogcost 1.6088 PairwiseErrs: [410, 150, 328]
ep  31 gain 1.000 trainlogcost 1.6015 PairwiseErrs: [384, 165, 305]
ep  32 gain 1.000 trainlogcost 1.4414 PairwiseErrs: [399, 137, 316]
ep  33 gain 1.000 trainlogcost 1.3744 PairwiseErrs: [393, 143, 288]
ep  34 gain 1.000 trainlogcost 1.3059 PairwiseErrs: [338, 118, 299]
Energy-based errs: Train 58/10000 Valid 224/10000
Softmax-based errs: Valid 207/10000
rms:  [0.049131767639609886, 0.03914461678502913, 0.041091883439555464]
suprms:  [0.018947571482192532, 0.019046527644219883]
ep  35 gain 1.000 trainlogcost 1.2886 PairwiseErrs: [357, 110, 254]
ep  36 gain 1.000 trainlogcost 1.1864 PairwiseErrs: [341, 110, 260]
ep  37 gain 1.000 trainlogcost 1.1675 PairwiseErrs: [347, 100, 272]
ep  38 gain 1.000 trainlogcost 1.2273 PairwiseErrs: [367, 113, 231]
ep  39 gain 1.000 trainlogcost 1.1792 PairwiseErrs: [342, 109, 227]
Energy-based errs: Train 52/10000 Valid 191/10000
Softmax-based errs: Valid 197/10000
rms:  [0.049981047972049955, 0.03889493512434857, 0.04091964243375563]
suprms:  [0.01821596121443444, 0.018651744805479528]
ep  40 gain 1.000 trainlogcost 1.0965 PairwiseErrs: [300, 74, 238]
ep  41 gain 1.000 trainlogcost 1.1472 PairwiseErrs: [296, 86, 220]
ep  42 gain 1.000 trainlogcost 0.9927 PairwiseErrs: [276, 86, 222]
ep  43 gain 1.000 trainlogcost 1.0290 PairwiseErrs: [283, 89, 197]
ep  44 gain 1.000 trainlogcost 0.9352 PairwiseErrs: [262, 91, 207]
Energy-based errs: Train 26/10000 Valid 190/10000
Softmax-based errs: Valid 183/10000
rms:  [0.05042136080751076, 0.038563331083758416, 0.040731665367979786]
suprms:  [0.017710886724232322, 0.018112964758600393]
ep  45 gain 1.000 trainlogcost 0.9998 PairwiseErrs: [257, 76, 203]
ep  46 gain 1.000 trainlogcost 0.9462 PairwiseErrs: [281, 80, 179]
ep  47 gain 1.000 trainlogcost 0.9597 PairwiseErrs: [269, 84, 157]
ep  48 gain 1.000 trainlogcost 0.9359 PairwiseErrs: [233, 66, 180]
ep  49 gain 1.000 trainlogcost 0.9181 PairwiseErrs: [256, 65, 161]
Energy-based errs: Train 32/10000 Valid 188/10000
Softmax-based errs: Valid 192/10000
rms:  [0.05090683633929016, 0.03816425158733312, 0.04056720929192193]
suprms:  [0.01748615312975319, 0.017891676270819213]
ep  50 gain 1.010 trainlogcost 0.8550 PairwiseErrs: [223, 53, 125]
ep  51 gain 0.990 trainlogcost 0.7836 PairwiseErrs: [218, 64, 152]
ep  52 gain 0.970 trainlogcost 0.7873 PairwiseErrs: [242, 55, 139]
ep  53 gain 0.950 trainlogcost 0.7709 PairwiseErrs: [206, 44, 127]
ep  54 gain 0.930 trainlogcost 0.7316 PairwiseErrs: [207, 41, 113]
Energy-based errs: Train 20/10000 Valid 180/10000
Softmax-based errs: Valid 197/10000
rms:  [0.05125470337028791, 0.037687951003756125, 0.04020169142408348]
suprms:  [0.016783574663682306, 0.0175539603741885]
ep  55 gain 0.910 trainlogcost 0.7184 PairwiseErrs: [216, 47, 110]
ep  56 gain 0.890 trainlogcost 0.6490 PairwiseErrs: [202, 39, 98]
ep  57 gain 0.870 trainlogcost 0.5675 PairwiseErrs: [196, 34, 102]
ep  58 gain 0.850 trainlogcost 0.5538 PairwiseErrs: [169, 19, 84]
ep  59 gain 0.830 trainlogcost 0.5381 PairwiseErrs: [173, 21, 82]
Energy-based errs: Train 7/10000 Valid 183/10000
Softmax-based errs: Valid 172/10000
rms:  [0.05121283079294069, 0.03713964992355416, 0.039735934247099154]
suprms:  [0.015943639500312353, 0.01673726908300535]
ep  60 gain 0.810 trainlogcost 0.5319 PairwiseErrs: [153, 14, 66]
ep  61 gain 0.790 trainlogcost 0.5358 PairwiseErrs: [146, 9, 59]
ep  62 gain 0.770 trainlogcost 0.5076 PairwiseErrs: [125, 11, 47]
ep  63 gain 0.750 trainlogcost 0.4954 PairwiseErrs: [140, 10, 48]
ep  64 gain 0.730 trainlogcost 0.4901 PairwiseErrs: [105, 11, 43]
Energy-based errs: Train 7/10000 Valid 147/10000
Softmax-based errs: Valid 164/10000
rms:  [0.05081161712453423, 0.03655931572496365, 0.039224772174533316]
suprms:  [0.015686016127476162, 0.016553836231720794]
ep  65 gain 0.710 trainlogcost 0.4744 PairwiseErrs: [123, 9, 49]
ep  66 gain 0.690 trainlogcost 0.4656 PairwiseErrs: [129, 5, 43]
ep  67 gain 0.670 trainlogcost 0.4822 PairwiseErrs: [93, 2, 34]
ep  68 gain 0.650 trainlogcost 0.4557 PairwiseErrs: [83, 3, 31]
ep  69 gain 0.630 trainlogcost 0.4454 PairwiseErrs: [78, 1, 28]
Energy-based errs: Train 4/10000 Valid 161/10000
Softmax-based errs: Valid 169/10000
rms:  [0.0503192010083405, 0.03602189051652008, 0.038704250757661866]
suprms:  [0.015524481367188413, 0.016490380986233257]
ep  70 gain 0.610 trainlogcost 0.4394 PairwiseErrs: [81, 1, 32]
ep  71 gain 0.590 trainlogcost 0.4142 PairwiseErrs: [82, 0, 32]
ep  72 gain 0.570 trainlogcost 0.4334 PairwiseErrs: [81, 2, 22]
ep  73 gain 0.550 trainlogcost 0.4497 PairwiseErrs: [67, 0, 8]
ep  74 gain 0.530 trainlogcost 0.4295 PairwiseErrs: [59, 1, 20]
Energy-based errs: Train 0/10000 Valid 163/10000
Softmax-based errs: Valid 170/10000
rms:  [0.049819302830087876, 0.035542898441373656, 0.03824054177232341]
suprms:  [0.015393703048367737, 0.01643448881694772]
ep  75 gain 0.510 trainlogcost 0.4161 PairwiseErrs: [62, 1, 11]
ep  76 gain 0.490 trainlogcost 0.4089 PairwiseErrs: [56, 0, 6]
ep  77 gain 0.470 trainlogcost 0.4304 PairwiseErrs: [57, 1, 7]
ep  78 gain 0.450 trainlogcost 0.4111 PairwiseErrs: [45, 1, 4]
ep  79 gain 0.430 trainlogcost 0.4139 PairwiseErrs: [44, 0, 2]
Energy-based errs: Train 0/10000 Valid 161/10000
Softmax-based errs: Valid 172/10000
rms:  [0.049327275936471605, 0.03514178211346526, 0.03783678756779345]
suprms:  [0.015350313800394828, 0.016454004754985783]
ep  80 gain 0.410 trainlogcost 0.4145 PairwiseErrs: [43, 1, 5]
ep  81 gain 0.390 trainlogcost 0.3979 PairwiseErrs: [45, 0, 7]
ep  82 gain 0.370 trainlogcost 0.4030 PairwiseErrs: [37, 0, 11]
ep  83 gain 0.350 trainlogcost 0.4063 PairwiseErrs: [39, 0, 5]
ep  84 gain 0.330 trainlogcost 0.4051 PairwiseErrs: [37, 0, 3]
Energy-based errs: Train 0/10000 Valid 158/10000
Softmax-based errs: Valid 163/10000
rms:  [0.04892916395288167, 0.03482791746899944, 0.037514803616468334]
suprms:  [0.015302915111987576, 0.016455893014901624]
ep  85 gain 0.310 trainlogcost 0.4107 PairwiseErrs: [36, 0, 3]
ep  86 gain 0.290 trainlogcost 0.4148 PairwiseErrs: [26, 0, 1]
ep  87 gain 0.270 trainlogcost 0.4134 PairwiseErrs: [30, 0, 5]
ep  88 gain 0.250 trainlogcost 0.4067 PairwiseErrs: [22, 0, 4]
ep  89 gain 0.230 trainlogcost 0.4105 PairwiseErrs: [18, 0, 3]
Energy-based errs: Train 0/10000 Valid 156/10000
Softmax-based errs: Valid 167/10000
rms:  [0.048617508018234494, 0.034598073150538776, 0.03726713026932858]
suprms:  [0.015342668042629959, 0.016547064181376744]
ep  90 gain 0.210 trainlogcost 0.4134 PairwiseErrs: [20, 0, 2]
ep  91 gain 0.190 trainlogcost 0.4091 PairwiseErrs: [16, 0, 3]
ep  92 gain 0.170 trainlogcost 0.4104 PairwiseErrs: [22, 0, 3]
ep  93 gain 0.150 trainlogcost 0.4093 PairwiseErrs: [12, 0, 2]
ep  94 gain 0.130 trainlogcost 0.4083 PairwiseErrs: [18, 0, 3]
Energy-based errs: Train 0/10000 Valid 153/10000
Softmax-based errs: Valid 161/10000
rms:  [0.048423627531245635, 0.034453361563510476, 0.03711420391814717]
suprms:  [0.015364237023916769, 0.016590003901247645]
ep  95 gain 0.110 trainlogcost 0.4077 PairwiseErrs: [15, 0, 1]
ep  96 gain 0.090 trainlogcost 0.4062 PairwiseErrs: [17, 0, 2]
ep  97 gain 0.070 trainlogcost 0.4063 PairwiseErrs: [16, 0, 2]
ep  98 gain 0.050 trainlogcost 0.4058 PairwiseErrs: [13, 0, 2]
ep  99 gain 0.030 trainlogcost 0.4067 PairwiseErrs: [11, 0, 1]
Energy-based errs: Train 0/10000 Valid 154/10000
Softmax-based errs: Valid 161/10000
rms:  [0.0483405628424256, 0.034393293139508375, 0.037049979998455565]
suprms:  [0.015369385049001117, 0.016602931047217672]
Energy-based errs: Train 0/10000 Test 135/10000
Softmax-based errs: Train 0/10000 Test 141/10000

real	102m42.328s
user	713m17.996s
sys	25m22.540s

% sysctl -a | grep machdep.cpu
machdep.cpu.core_count: 8
machdep.cpu.brand_string: Apple M1
```
