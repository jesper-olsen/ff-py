Forward-Forward
===============

The Forward-Forward algorithm [1] evaluated on the MNIST handwritten digit recognition task; Python+Numpy implementation based on [2].


References:
-----------
[1] [The Forward-Forward Algorithm: Some Preliminary Investigations, Geoffrey Hinton, NeurIPS 2022](https://www.cs.toronto.edu/~hinton/FFA13.pdf) <br/>
[2] [Hinton's matlab implementation](https://www.cs.toronto.edu/~hinton/ffcode.zip) <br/>
[3] [mnistdata.mat](https://www.cs.toronto.edu/~hinton/mnistdata.mat) <br/>

Run:
----

```
% wget https://www.cs.toronto.edu/~hinton/mnistdata.mat
% time python main.py
Batchsize: 100 Input-dim: 784 #training batches: 500
nums per layer:  [784, 1000, 1000, 1000, 10]
ep   0 gain 1.000 trainlogcost 49.1148 PairwiseErrs: 5493 5494 6022
ep   1 gain 1.000 trainlogcost 19.7043 PairwiseErrs: 2628 2278 2654
ep   2 gain 1.000 trainlogcost 14.9785 PairwiseErrs: 2127 1723 1960
ep   3 gain 1.000 trainlogcost 12.0462 PairwiseErrs: 1816 1435 1723
ep   4 gain 1.000 trainlogcost 10.4088 PairwiseErrs: 1575 1203 1493
Energy-based errs: Train 379/10000 Valid 429/10000
Softmax-based errs: Valid 375/10000
rms:  [0.04030901318891625, 0.03645233158132287, 0.03817873565218584]
suprms:  [0.03310189209736562, 0.023474943171433536]
ep   5 gain 1.000 trainlogcost 8.7328 PairwiseErrs: 1428 1033 1266
ep   6 gain 1.000 trainlogcost 7.5871 PairwiseErrs: 1311  905 1120
ep   7 gain 1.000 trainlogcost 6.7644 PairwiseErrs: 1125  806 1047
ep   8 gain 1.000 trainlogcost 5.9325 PairwiseErrs: 1087  710  929
ep   9 gain 1.000 trainlogcost 5.2349 PairwiseErrs: 1033  628  852
Energy-based errs: Train 171/10000 Valid 264/10000
Softmax-based errs: Valid 255/10000
rms:  [0.04184359050875991, 0.036991606303647104, 0.03859976067610819]
suprms:  [0.02726366316360332, 0.021250599986890863]
ep  10 gain 1.000 trainlogcost 4.8720 PairwiseErrs:  906  595  838
ep  11 gain 1.000 trainlogcost 4.5404 PairwiseErrs:  857  531  736
ep  12 gain 1.000 trainlogcost 4.0313 PairwiseErrs:  848  487  716
ep  13 gain 1.000 trainlogcost 3.8634 PairwiseErrs:  805  439  652
ep  14 gain 1.000 trainlogcost 3.5067 PairwiseErrs:  740  413  615
Energy-based errs: Train 115/10000 Valid 226/10000
Softmax-based errs: Valid 222/10000
rms:  [0.043046032248313604, 0.03733313115876664, 0.03896115937752122]
suprms:  [0.023639827564772738, 0.0204581491256486]
ep  15 gain 1.000 trainlogcost 3.3617 PairwiseErrs:  675  391  588
ep  16 gain 1.000 trainlogcost 3.0291 PairwiseErrs:  658  373  535
ep  17 gain 1.000 trainlogcost 3.1683 PairwiseErrs:  657  354  576
ep  18 gain 1.000 trainlogcost 2.6714 PairwiseErrs:  625  284  479
ep  19 gain 1.000 trainlogcost 2.6889 PairwiseErrs:  615  316  490
Energy-based errs: Train 144/10000 Valid 257/10000
Softmax-based errs: Valid 218/10000
rms:  [0.04412958119239091, 0.037498130809682235, 0.03916722668392432]
suprms:  [0.021436095681983396, 0.020374821756173618]
ep  20 gain 1.000 trainlogcost 2.5296 PairwiseErrs:  595  287  425
ep  21 gain 1.000 trainlogcost 2.2457 PairwiseErrs:  554  248  400
ep  22 gain 1.000 trainlogcost 2.1549 PairwiseErrs:  512  234  383
ep  23 gain 1.000 trainlogcost 2.0360 PairwiseErrs:  511  236  415
ep  24 gain 1.000 trainlogcost 1.8495 PairwiseErrs:  468  201  392
Energy-based errs: Train 64/10000 Valid 203/10000
Softmax-based errs: Valid 195/10000
rms:  [0.04483595074173552, 0.03746274097096787, 0.03921710623129999]
suprms:  [0.01990696573133344, 0.019818994132419464]
ep  25 gain 1.000 trainlogcost 1.6841 PairwiseErrs:  430  198  339
ep  26 gain 1.000 trainlogcost 1.6276 PairwiseErrs:  428  183  316
ep  27 gain 1.000 trainlogcost 1.5493 PairwiseErrs:  395  154  303
ep  28 gain 1.000 trainlogcost 1.5815 PairwiseErrs:  410  166  313
ep  29 gain 1.000 trainlogcost 1.5596 PairwiseErrs:  403  155  325
Energy-based errs: Train 67/10000 Valid 212/10000
Softmax-based errs: Valid 213/10000
rms:  [0.04569281275309137, 0.03732592259680052, 0.03919954782458951]
suprms:  [0.018723521785358622, 0.019571952995294063]
ep  30 gain 1.000 trainlogcost 1.4281 PairwiseErrs:  387  140  300
ep  31 gain 1.000 trainlogcost 1.3724 PairwiseErrs:  373  133  290
ep  32 gain 1.000 trainlogcost 1.3504 PairwiseErrs:  355  122  284
ep  33 gain 1.000 trainlogcost 1.3686 PairwiseErrs:  373  142  239
ep  34 gain 1.000 trainlogcost 1.2648 PairwiseErrs:  351  105  234
Energy-based errs: Train 63/10000 Valid 209/10000
Softmax-based errs: Valid 217/10000
rms:  [0.04640562887443283, 0.03712808769227475, 0.03912947928900606]
suprms:  [0.01812412757796979, 0.019338680061801655]
ep  35 gain 1.000 trainlogcost 1.2001 PairwiseErrs:  330  118  267
ep  36 gain 1.000 trainlogcost 1.0695 PairwiseErrs:  320   94  207
ep  37 gain 1.000 trainlogcost 1.0301 PairwiseErrs:  296   87  188
ep  38 gain 1.000 trainlogcost 0.9789 PairwiseErrs:  303   81  211
ep  39 gain 1.000 trainlogcost 0.9145 PairwiseErrs:  275   73  189
Energy-based errs: Train 36/10000 Valid 191/10000
Softmax-based errs: Valid 196/10000
rms:  [0.0468818898884325, 0.03680643315004454, 0.03898480936451221]
suprms:  [0.01704962653263666, 0.018581254409660537]
ep  40 gain 1.000 trainlogcost 0.9507 PairwiseErrs:  280   91  191
ep  41 gain 1.000 trainlogcost 0.9637 PairwiseErrs:  288   70  203
ep  42 gain 1.000 trainlogcost 0.8587 PairwiseErrs:  272   66  158
ep  43 gain 1.000 trainlogcost 0.8043 PairwiseErrs:  248   68  164
ep  44 gain 1.000 trainlogcost 0.8272 PairwiseErrs:  245   54  146
Energy-based errs: Train 33/10000 Valid 185/10000
Softmax-based errs: Valid 202/10000
rms:  [0.04734033658235443, 0.03644265641418183, 0.03874192682982914]
suprms:  [0.016612477306533847, 0.018320899286703204]
ep  45 gain 1.000 trainlogcost 0.8349 PairwiseErrs:  245   53  150
ep  46 gain 1.000 trainlogcost 0.7766 PairwiseErrs:  236   62  142
ep  47 gain 1.000 trainlogcost 0.7409 PairwiseErrs:  232   48  137
ep  48 gain 1.000 trainlogcost 0.6962 PairwiseErrs:  212   41  128
ep  49 gain 1.000 trainlogcost 0.7412 PairwiseErrs:  207   58  125
Energy-based errs: Train 17/10000 Valid 190/10000
Softmax-based errs: Valid 184/10000
rms:  [0.04751493048205362, 0.03605351458778264, 0.03847518787358369]
suprms:  [0.016000331890704895, 0.017911711577136336]
ep  50 gain 1.010 trainlogcost 0.6499 PairwiseErrs:  216   43  131
ep  51 gain 0.990 trainlogcost 0.6706 PairwiseErrs:  188   51  122
ep  52 gain 0.970 trainlogcost 0.6416 PairwiseErrs:  200   39  131
ep  53 gain 0.950 trainlogcost 0.6406 PairwiseErrs:  209   36  122
ep  54 gain 0.930 trainlogcost 0.6280 PairwiseErrs:  186   29   87
Energy-based errs: Train 21/10000 Valid 185/10000
Softmax-based errs: Valid 185/10000
rms:  [0.047741502781104034, 0.035603994435331125, 0.03813178913053258]
suprms:  [0.015524166552061439, 0.017520130107314447]
ep  55 gain 0.910 trainlogcost 0.6247 PairwiseErrs:  201   28  106
ep  56 gain 0.890 trainlogcost 0.5961 PairwiseErrs:  183   31   81
ep  57 gain 0.870 trainlogcost 0.5410 PairwiseErrs:  173   16   74
ep  58 gain 0.850 trainlogcost 0.5742 PairwiseErrs:  135   16   63
ep  59 gain 0.830 trainlogcost 0.5101 PairwiseErrs:  145   14   62
Energy-based errs: Train 14/10000 Valid 172/10000
Softmax-based errs: Valid 182/10000
rms:  [0.047713853832884436, 0.03511106455111999, 0.03771844430370491]
suprms:  [0.015055211581840913, 0.017127339370517133]
ep  60 gain 0.810 trainlogcost 0.4899 PairwiseErrs:  146   23   69
ep  61 gain 0.790 trainlogcost 0.4917 PairwiseErrs:  136   20   49
ep  62 gain 0.770 trainlogcost 0.4432 PairwiseErrs:  125    6   47
ep  63 gain 0.750 trainlogcost 0.4544 PairwiseErrs:  117    7   44
ep  64 gain 0.730 trainlogcost 0.4498 PairwiseErrs:  117    7   35
Energy-based errs: Train 4/10000 Valid 168/10000
Softmax-based errs: Valid 160/10000
rms:  [0.04736045076334161, 0.034567763308098284, 0.03725706640377816]
suprms:  [0.014683252157294862, 0.016857766888810122]
ep  65 gain 0.710 trainlogcost 0.4308 PairwiseErrs:   99    4   32
ep  66 gain 0.690 trainlogcost 0.4198 PairwiseErrs:   93    6   26
ep  67 gain 0.670 trainlogcost 0.3968 PairwiseErrs:   85    2   33
ep  68 gain 0.650 trainlogcost 0.3918 PairwiseErrs:   78    3   21
ep  69 gain 0.630 trainlogcost 0.3961 PairwiseErrs:   74    3   22
Energy-based errs: Train 1/10000 Valid 152/10000
Softmax-based errs: Valid 163/10000
rms:  [0.04686158253502207, 0.03405864339141108, 0.03677689626524941]
suprms:  [0.014380362852144807, 0.016592597303916837]
ep  70 gain 0.610 trainlogcost 0.3653 PairwiseErrs:   66    2   22
ep  71 gain 0.590 trainlogcost 0.3786 PairwiseErrs:   63    0   11
ep  72 gain 0.570 trainlogcost 0.3646 PairwiseErrs:   65    0   20
ep  73 gain 0.550 trainlogcost 0.3730 PairwiseErrs:   59    1   16
ep  74 gain 0.530 trainlogcost 0.3671 PairwiseErrs:   53    1   17
Energy-based errs: Train 0/10000 Valid 155/10000
Softmax-based errs: Valid 164/10000
rms:  [0.04634611863584721, 0.03360615731683351, 0.036335228393359564]
suprms:  [0.01418342140305515, 0.01644756139066215]
ep  75 gain 0.510 trainlogcost 0.3562 PairwiseErrs:   45    0   14
ep  76 gain 0.490 trainlogcost 0.3578 PairwiseErrs:   48    1   12
ep  77 gain 0.470 trainlogcost 0.3613 PairwiseErrs:   42    0   12
ep  78 gain 0.450 trainlogcost 0.3600 PairwiseErrs:   46    0    8
ep  79 gain 0.430 trainlogcost 0.3521 PairwiseErrs:   38    0    5
Energy-based errs: Train 0/10000 Valid 147/10000
Softmax-based errs: Valid 151/10000
rms:  [0.0459073493910027, 0.03322998282241794, 0.03595802322940816]
suprms:  [0.014063632369063161, 0.016414247120251867]
ep  80 gain 0.410 trainlogcost 0.3602 PairwiseErrs:   28    0    3
ep  81 gain 0.390 trainlogcost 0.3682 PairwiseErrs:   42    0    6
ep  82 gain 0.370 trainlogcost 0.3632 PairwiseErrs:   32    0    4
ep  83 gain 0.350 trainlogcost 0.3568 PairwiseErrs:   36    0    4
ep  84 gain 0.330 trainlogcost 0.3516 PairwiseErrs:   25    0    2
Energy-based errs: Train 0/10000 Valid 141/10000
Softmax-based errs: Valid 146/10000
rms:  [0.045532662298736884, 0.03293570790841308, 0.03565015498637654]
suprms:  [0.014077792974775604, 0.01647963753817391]
ep  85 gain 0.310 trainlogcost 0.3574 PairwiseErrs:   25    0    5
ep  86 gain 0.290 trainlogcost 0.3523 PairwiseErrs:   20    0    2
ep  87 gain 0.270 trainlogcost 0.3581 PairwiseErrs:   17    0    1
ep  88 gain 0.250 trainlogcost 0.3594 PairwiseErrs:   18    0    0
ep  89 gain 0.230 trainlogcost 0.3620 PairwiseErrs:   17    0    1
Energy-based errs: Train 0/10000 Valid 144/10000
Softmax-based errs: Valid 147/10000
rms:  [0.04524737009592482, 0.03271745520115143, 0.035420291638530534]
suprms:  [0.0140842547034289, 0.016536269094641398]
ep  90 gain 0.210 trainlogcost 0.3590 PairwiseErrs:   14    0    3
ep  91 gain 0.190 trainlogcost 0.3577 PairwiseErrs:   15    0    3
ep  92 gain 0.170 trainlogcost 0.3646 PairwiseErrs:   19    0    1
ep  93 gain 0.150 trainlogcost 0.3645 PairwiseErrs:    9    0    0
ep  94 gain 0.130 trainlogcost 0.3622 PairwiseErrs:   12    0    1
Energy-based errs: Train 0/10000 Valid 142/10000
Softmax-based errs: Valid 148/10000
rms:  [0.04506201775920842, 0.03258018037580766, 0.035278420495746185]
suprms:  [0.014100911118054999, 0.01658898158462811]
ep  95 gain 0.110 trainlogcost 0.3615 PairwiseErrs:   16    0    1
ep  96 gain 0.090 trainlogcost 0.3609 PairwiseErrs:    9    0    1
ep  97 gain 0.070 trainlogcost 0.3600 PairwiseErrs:   14    0    2
ep  98 gain 0.050 trainlogcost 0.3584 PairwiseErrs:   12    0    1
ep  99 gain 0.030 trainlogcost 0.3581 PairwiseErrs:    9    0    1
Energy-based errs: Train 0/10000 Valid 145/10000
Softmax-based errs: Valid 147/10000
rms:  [0.04498485614340065, 0.03252363614861234, 0.03521701336978582]
suprms:  [0.01410813484921783, 0.016608840118146413]
Energy-based errs: Train 0/10000 Test 134/10000
Softmax-based errs: Train 0/10000 Test 142/10000

real	111m49.315s
user	752m57.285s
sys	48m21.001s
```
