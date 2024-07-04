import collections
import numpy as np
from scipy.special import expit as logistic # 1/(1+exp(-x))
import mnist

tiny = 1e-20  # for preventing divisions by zero
dtype=np.float32
NUMLAB = 10

def ffnormrows(a):
    # Makes every 'a' have a sum of squared activities that averages 1 per neuron.
    num_comp = a.shape[1]
    return ( a / (tiny + np.sqrt(np.mean(a**2, axis=1, keepdims=True))) * np.ones((num_comp), dtype=dtype))

def choosefrom(probs):
    """vectorized - probablistically choose neg examples that are more like the targets"""
    batch_size, nlab = probs.shape
    random_values = np.random.rand(batch_size, 1)
    cumulative_probs = np.cumsum(probs, axis=1)
    chosen_labels = (random_values < cumulative_probs).argmax(axis=1)
    postchoiceprobs = np.zeros_like(probs, dtype=dtype)
    postchoiceprobs[np.arange(batch_size), chosen_labels] = 1.0
    return postchoiceprobs

def softmax(scores):
    exp_scores = np.exp(scores)
    return  exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def equicols(matrix):
    norms = np.linalg.norm(matrix, axis=0)  # Calculate L2 norms of each column
    return matrix / norms  # Normalize each column by its L2 norm

def rms(x):
    # Assumes x is a matrix, but should work for vectors and scalars too
    return np.sqrt(np.sum(x**2) / (x.size))

def layer_io(vin, weights, biases):
    states = np.maximum(0, vin @ weights + biases)
    normstates = ffnormrows(states)
    return states,normstates

def ffenergytest(data):
    actsumsq = np.zeros((len(data), NUMLAB), dtype=dtype)
    for lab in range(NUMLAB):
        data[:, :NUMLAB] = np.zeros((len(data), NUMLAB), dtype=dtype)
        data[:, lab] = labelstrength * np.ones(len(data), dtype=dtype)
        normstates_lm1 = ffnormrows(data)
        for l in range(1, NLAYERS - 1):
            states,normstates_lm1 = layer_io(normstates_lm1, weights[l], biases[l])
            if l >= minlevelenergy:
                actsumsq[:, lab] += np.sum(states**2, axis=1)
    return np.argmax(actsumsq, axis=1) #guesses

def ffsoftmaxtest(data):
    data[:, :NUMLAB] = labelstrength * np.ones((len(data), NUMLAB),dtype=dtype) / NUMLAB
    normstates = {0: ffnormrows(data)}
    for l in range(1, NLAYERS - 1):
        _, normstates[l] = layer_io(normstates[l-1], weights[l], biases[l])
    labin = np.tile(biases[NLAYERS-1], (len(data), 1))
    for l in range(minlevelsup, NLAYERS - 1):
        labin += normstates[l] @ supweightsfrom[l]
    labin -= np.max(labin, axis=1, keepdims=True)
    unnormlabprobs = np.exp(labin)
    testpredictions = unnormlabprobs / np.sum(unnormlabprobs, axis=1, keepdims=True)
    # logcost += -np.sum(targets * np.log(tiny + testpredictions)) / numbatches
    return  np.argmax(testpredictions, axis=1) # guesses

def fftest(f_batch, batchdata, batchtargets):
    errors = tests = 0
    for batch in range(len(batchdata)):
        data = batchdata[:, :, batch]
        targets = batchtargets[:, :, batch]
        targetindices = np.argmax(targets, axis=1)
        guesses = f_batch(data)
        errors += np.sum(guesses != targetindices)
        tests += len(guesses)
    return errors, tests

np.random.seed(1234)

mnist_data=mnist.make_batches("MNIST")
batch_size, idim, numbatches = mnist_data["batchdata"].shape
print(f"Batchsize: {batch_size} Input-dim: {idim} #training batches: {numbatches}")

LAYERS = [idim, 1000, 1000, 1000, NUMLAB] 
NLAYERS = len(LAYERS)

lambdamean = 0.03
# Peer normalization: we regress the mean activity of each neuron towards the average mean for its layer.
# This prevents dead or hysterical units. We pretend there is a gradient even when hidden units are off.
# Choose strength of regression (lambdamean) so that average activities are similar but not too similar.

temp = 1  # rescales the logits used for deciding fake vs real
labelstrength = 1.0  # scaling up the activity of the label pixel doesn't seem to help much.
minlevelsup = 2  # used in training softmax predictor. Does not use hidden layers lower than this.
minlevelenergy = 2  # used in computing goodness at test time. Does not use hidden layers lower than this.
wc = 0.002 # weightcost on forward weights.
supwc = 0.003  # weightcost on label prediction weights.
epsilon = 0.01  # learning rate for forward weights.
epsilonsup = 0.1  # learning rate for linear softmax weights.
delay = 0.9  #  used for smoothing the gradient over minibatches. 0.9 = 1 - 0.1

normstates = [None] * NLAYERS
posprobs = [None] * NLAYERS  # column vector of probs that positive cases are positive.
negprobs = [None] * NLAYERS  # column vector of probs that negative cases are POSITIVE.

# meanstates - running average of the mean activity of a hidden unit.
meanstates = {l: 0.5 * np.ones(LAYERS[l], dtype=dtype) for l in range(1,NLAYERS-1)}

# the forward weights - scaled by sqrt(fanin). weights[2] is incoming weights to layer 2.
weights = {l: (1/np.sqrt(LAYERS[l-1]))*np.random.randn(LAYERS[l-1], LAYERS[l])  for l in range(1,NLAYERS)}
biases = {l: 0.0 * np.ones(LAYERS[l],dtype=dtype) for l in range(1,NLAYERS)}

#gradients are smoothed over minibatches

# gradients of probability of correct real/fake decision w.r.t. weights & biases
posdCbydweights = {l: np.zeros((LAYERS[l - 1], LAYERS[l]), dtype=dtype) for l in range(1,NLAYERS-1)}
negdCbydweights = {l: np.zeros((LAYERS[l - 1], LAYERS[l]), dtype=dtype) for l in range(1,NLAYERS-1)}
posdCbydbiases = {l: np.zeros((1, LAYERS[l]), dtype=dtype) for l in range(1,NLAYERS-1)}
negdCbydbiases = {l: np.zeros((1, LAYERS[l]), dtype=dtype) for l in range(1,NLAYERS-1)}
weightsgrad = {l: np.zeros((LAYERS[l - 1], LAYERS[l]), dtype=dtype) for l in range(1,NLAYERS-1)}
biasesgrad = {l: np.zeros((1, LAYERS[l]), dtype=dtype) for l in range(1,NLAYERS-1)}

# the weights used for predicting the label from the higher hidden layer activities.
supweightsfrom = {l: np.zeros((LAYERS[l], LAYERS[-1]), dtype=dtype) for l in range(1,NLAYERS-1)}
supweightsfromgrad = {l: np.zeros((LAYERS[l], LAYERS[-1]), dtype=dtype) for l in range(1,NLAYERS-1)}

print("states per layer: ", LAYERS)
MAXEPOCH = 100
for epoch in range(0, MAXEPOCH):
    # number of times a negative example has higher goodness than the positive example
    pairsumerrs = collections.defaultdict(int)
    trainlogcost = 0.0
    # multiplier on all weight changes - decays linearly to zero after MAXEPOCH/2
    epsgain = 1.0 if epoch<MAXEPOCH/2 else (1.0 + 2.0 * (MAXEPOCH - epoch)) / MAXEPOCH 

    np.set_printoptions(threshold=np.inf)
    for batch in range(numbatches):
        data = mnist_data["batchdata"][:, :, batch]  # 100x784
        targets = mnist_data["batchtargets"][:, :, batch]
        data[:, :NUMLAB] = labelstrength * targets
        normstates[0] = ffnormrows(data)

        for l in range(1, NLAYERS - 1):
            states,normstates[l]=layer_io(normstates[l-1], weights[l], biases[l])
            posprobs[l] = logistic( (np.sum(states**2, axis=1, keepdims=True) - LAYERS[l]) / temp )

            replicated_states = np.repeat(1-posprobs[l], LAYERS[l]).reshape(-1, LAYERS[l])
            # gradients of goodness w.r.t. total input to a hidden unit.
            dCbydin = replicated_states * states  # Element-wise multiplication
            # wrong sign: rate at which it gets BETTER not worse. Think of C as goodness.

            meanstates[l] = 0.9 * meanstates[l] + 0.1 * np.mean(states[l])  # Element-wise operations
            mean_meanstates = np.mean(meanstates[l])
            dCbydin = dCbydin + lambdamean * (mean_meanstates - meanstates[l])  
            # This is a regularizer that encourages the average activity of a unit to match that for
            # all the units in the layer. Notice that we do not gate by (states>0) for this extra term.
            # This allows the extra term to revive units that are always off.  May not be needed.

            posdCbydweights[l] = normstates[l - 1].T @ dCbydin
            posdCbydbiases[l] = np.sum(dCbydin, axis=0)

        # NOW WE GET THE HIDDEN STATES WHEN THE LABEL IS NEUTRAL AND USE THE NORMALIZED HIDDEN STATES AS
        # INPUTS TO A SOFTMAX.  THIS SOFTMAX IS USED TO PICK HARD NEGATIVE LABELS

        ones_array = np.ones((batch_size, LAYERS[-1]), dtype=dtype) # dummy label in 1st 10 columns
        data[:, :NUMLAB] = labelstrength * ones_array[:, :NUMLAB] / LAYERS[-1]
        normstates[0] = ffnormrows(data)
        for l in range(1, NLAYERS - 1):
            states, normstates[l] = layer_io(normstates[l-1], weights[l], biases[l])
        labin = np.tile(biases[NLAYERS-1], (batch_size, 1))
        for l in range(minlevelsup, NLAYERS - 1):
            labin += normstates[l] @ supweightsfrom[l]
            # normstates seems to work better than states for predicting the label

        max_labin = np.max(labin, axis=1, keepdims=True)
        labin -= np.tile(max_labin, (1, LAYERS[-1]))
        unnormlabprobs = np.exp(labin)
        sum_unnormlabprobs = np.sum(unnormlabprobs, axis=1, keepdims=True)
        trainpredictions = unnormlabprobs / np.tile( sum_unnormlabprobs, (1, LAYERS[-1]) )
        correctprobs = np.sum(trainpredictions * targets, axis=1)
        trainlogcost += np.sum(-np.log(tiny+correctprobs))/numbatches # per batch (not per case)

        trainguesses = np.argmax(trainpredictions, axis=1)
        targetindices = np.argmax(targets, axis=1)
        trainerrors = np.sum(trainguesses != targetindices)

        dCbydin = targets - trainpredictions
        # dCbydbiases[-1] = np.sum(dCbydin[-1], axis=0)

        for l in range(minlevelsup, NLAYERS - 1):
            dCbydsupweightsfrom = normstates[l].T @ dCbydin
            supweightsfromgrad[l] = \
                delay * supweightsfromgrad[l] + (1 - delay) * dCbydsupweightsfrom / batch_size
            supweightsfrom[l] = supweightsfrom[l] + epsgain * epsilonsup * \
                (supweightsfromgrad[l] - supwc * supweightsfrom[l])
        # HACK: it works better without predicting the label from the first hidden layer.

        # NOW WE MAKE NEGDATA
        negdata = data
        labinothers = (labin - 1000 * targets)  # big negative logits for the targets so we do not choose them
        chosen_labels = choosefrom( softmax(labinothers) )
        negdata[:, :NUMLAB] = labelstrength * chosen_labels
        normstates[0] = ffnormrows(negdata)
        for l in range(1, NLAYERS - 1):
            states, normstates[l] = layer_io(normstates[l-1], weights[l], biases[l])
            sum_states_l_squared = np.sum(states**2, axis=1, keepdims=True)
            # negprobs - probability of saying a negative case is POSITIVE.
            negprobs[l] = logistic( (sum_states_l_squared - LAYERS[l]) / temp)  
            dCbydin = -np.tile(negprobs[l], (1, LAYERS[l])) * states
            negdCbydweights[l] = normstates[l - 1].T @ dCbydin
            negdCbydbiases[l] = np.sum(dCbydin, axis=0)
            pairsumerrs[l] += np.sum(negprobs[l] > posprobs[l])

        for l in range(1, NLAYERS - 1):
            weightsgrad[l] = \
                delay * weightsgrad[l] + (1 - delay) * (posdCbydweights[l] + negdCbydweights[l]) / batch_size
            biasesgrad[l] = \
                delay * biasesgrad[l] + (1 - delay) * (posdCbydbiases[l] + negdCbydbiases[l]) / batch_size
            biases[l] = biases[l] + epsgain * epsilon * biasesgrad[l]
            weights[l] = weights[l] + epsgain * epsilon * (weightsgrad[l] - wc * weights[l])

            # Optionally, apply equicols(weights{l}) if equicols function is available
            # equicols makes the incoming weight vectors have the same L2 norm for all units in a layer.
            # weights[l] = equicols(weights[l])

    if True:
        print(f"ep {epoch:3} gain {epsgain:.3f} trainlogcost {trainlogcost:.4f} PairwiseErrs:",
            ", ".join([f"{pairsumerrs[l]}" for l in range(1,NLAYERS-1)]))

    if (epoch + 1) % 5 == 0:
        tr_errors, tr_tests = fftest(ffenergytest, mnist_data["batchdata"][:100], mnist_data["batchtargets"])
        verrors, vtests = fftest(ffenergytest, mnist_data["validbatchdata"][:100], mnist_data["validbatchtargets"])
        print(f"Energy-based errs: Train {tr_errors}/{tr_tests} Valid {verrors}/{vtests}")
        verrors, vtests = fftest(ffsoftmaxtest, mnist_data["validbatchdata"][:100], mnist_data["validbatchtargets"])
        print(f"Softmax-based errs: Valid {verrors}/{vtests}")

        print("rms: ", ", ".join([f"{rms(weights[l]):.4f}" for l in range(1, NLAYERS - 1)]))
        #print("rms: ", [rms(weights[l]) for l in range(1, NLAYERS - 1)])
        print("suprms: ", ", ".join([f"{rms(supweightsfrom[l]):.4f}" for l in range(minlevelsup, NLAYERS - 1)]))
        # the magnitudes of the sup weights show how much each hidden layer contributes to the softmax.

tr_errors, tr_tests = fftest(ffenergytest, mnist_data["batchdata"][:100], mnist_data["batchtargets"])
te_errors, te_tests = fftest(ffenergytest, mnist_data["testbatchdata"][:100], mnist_data["testbatchtargets"])
print(f"Energy-based errs: Train {tr_errors}/{tr_tests} Test {te_errors}/{te_tests}")

te_errors, te_tests = fftest(ffsoftmaxtest, mnist_data["testbatchdata"][:100], mnist_data["testbatchtargets"])
print(f"Softmax-based errs: Train {tr_errors}/{tr_tests} Test {te_errors}/{te_tests}")
