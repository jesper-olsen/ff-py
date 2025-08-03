import collections
import numpy as np
from scipy.special import expit as logistic # 1/(1+exp(-x))
import mnist

TINY = 1e-20  # for preventing divisions by zero
DTYPE=np.float32
NUMLAB = 10

LAMBDAMEAN = 0.03
# Peer normalization: we regress the mean activity of each neuron towards the average mean for its layer.
# This prevents dead or hysterical units. We pretend there is a gradient even when hidden units are off.
# Choose strength of regression (LAMBDAMEAN) so that average activities are similar but not too similar.

TEMP = 1  # rescales the logits used for deciding fake vs real
LABELSTRENGTH = 1.0  # scaling up the activity of the label pixel doesn't seem to help much.
MINLEVELSUP = 2  # used in training softmax predictor. Does not use hidden layers lower than this.
MINLEVELENERGY = 2  # used in computing goodness at test time. Does not use hidden layers lower than this.
WC = 0.002 # weightcost on forward weights.
SUPWC = 0.003  # weightcost on label prediction weights.
EPSILON = 0.01  # learning rate for forward weights.
EPSILONSUP = 0.1  # learning rate for linear softmax weights.
DELAY = 0.9  #  used for smoothing the gradient over minibatches. 0.9 = 1 - 0.1

def ffnormrows(a):
    # Makes every 'a' have a sum of squared activities that averages 1 per neuron.
    num_comp = a.shape[1]
    return a / (TINY + np.sqrt(np.mean(a**2, axis=1, keepdims=True)))

def choosefrom(probs):
    """vectorized - probabilistically choose neg examples that are more like the targets"""
    batch_size, _ = probs.shape   # batch_size x nlabels
    random_values = np.random.rand(batch_size, 1)
    cumulative_probs = np.cumsum(probs, axis=1)
    chosen_labels = (random_values < cumulative_probs).argmax(axis=1)
    postchoiceprobs = np.zeros_like(probs, dtype=DTYPE)
    postchoiceprobs[np.arange(batch_size), chosen_labels] = 1.0
    return postchoiceprobs

def softmax(scores):
    exp_scores = np.exp(scores)
    return  exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

def equicols(matrix):
    norms = np.linalg.norm(matrix, axis=0)  # Calculate L2 norms of each column
    return matrix / norms  # Normalize each column by its L2 norm

def rms(x) -> float:
    # Assumes x is a matrix, but should work for vectors and scalars too
    return np.sqrt(np.sum(x**2) / (x.size))

def layer_io(vin, lmodel):
    states = np.maximum(0, vin @ lmodel['weights'] + lmodel['biases'])
    normstates = ffnormrows(states)
    return states,normstates

def ffenergytest(data, model):
    actsumsq = np.zeros((len(data), NUMLAB), dtype=DTYPE)
    for lab in range(NUMLAB):
        data[:, :NUMLAB] = np.zeros((len(data), NUMLAB), dtype=DTYPE)
        data[:, lab] = LABELSTRENGTH * np.ones(len(data), dtype=DTYPE)
        normstates_lm1 = ffnormrows(data)
        for l in range(1,len(model)):
        #for l in range(1, NLAYERS - 1):
            states,normstates_lm1 = layer_io(normstates_lm1, model[l])
            if l >= MINLEVELENERGY:
                actsumsq[:, lab] += np.sum(states**2, axis=1)
    return np.argmax(actsumsq, axis=1) #guesses

def ffsoftmaxtest(data, model):
    data[:, :NUMLAB] = LABELSTRENGTH * np.ones((len(data), NUMLAB),dtype=DTYPE) / NUMLAB
    normstates = {0: ffnormrows(data)}
    for l in range(1,len(model)):
        _, normstates[l] = layer_io(normstates[l-1], model[l])
    labin = np.tile(model[-1]['biases'], (len(data), 1))
    for l in range(MINLEVELSUP, len(model)):
        labin += normstates[l] @ model[l]['supweights']
    labin -= np.max(labin, axis=1, keepdims=True)
    unnormlabprobs = np.exp(labin)
    testpredictions = unnormlabprobs / np.sum(unnormlabprobs, axis=1, keepdims=True)
    # logcost += -np.sum(targets * np.log(TINY + testpredictions)) / numbatches
    return  np.argmax(testpredictions, axis=1) # guesses

def fftest(f_batch, batchdata, batchtargets, model):
    errors = tests = 0
    for batch in range(batchdata.shape[2]):
        data = batchdata[:, :, batch]
        targets = batchtargets[:, :, batch]
        targetindices = np.argmax(targets, axis=1)
        guesses = f_batch(data, model)
        errors += np.sum(guesses != targetindices)
        tests += len(guesses)
    return errors, tests

def train(mnist_data):
    batch_size, idim, numbatches = mnist_data["batchdata"].shape
    print(f"Batchsize: {batch_size} Input-dim: {idim} #training batches: {numbatches}")

    LAYERS = [idim, 1000, 1000, 1000, NUMLAB] 
    NLAYERS = len(LAYERS)

    # meanstates - running average of the mean activity of a hidden unit.
    meanstates = {l: 0.5 * np.ones(LAYERS[l], dtype=DTYPE) for l in range(1,NLAYERS-1)}

    model=[None]
    for l, (fanin,fanout) in enumerate(zip(LAYERS[:-1],LAYERS[1:])):
        d={'weights':(1/np.sqrt(fanin))*np.random.randn(fanin, fanout).astype(DTYPE),  
           'biases': 0.0 * np.ones(fanout,dtype=DTYPE)} 
        if l<NLAYERS-1:
            d['supweights'] = np.zeros((fanout, LAYERS[-1]), dtype=DTYPE) 
        model+=[d] 

    # gradients of probability of correct real/fake decision w.r.t. weights & biases - smoothed over minibatches
    posdCbydweights = [None]*NLAYERS
    negdCbydweights = [None]*NLAYERS
    posdCbydbiases = [None]*NLAYERS
    negdCbydbiases = [None]*NLAYERS
    weightsgrad = [None]+[np.zeros((LAYERS[l - 1], LAYERS[l]), dtype=DTYPE) for l in range(1,NLAYERS-1)]
    biasesgrad = [None]+[np.zeros((1, LAYERS[l]), dtype=DTYPE) for l in range(1,NLAYERS-1)]
    supweightsgrad = [None]+[np.zeros((LAYERS[l], LAYERS[-1]), dtype=DTYPE) for l in range(1,NLAYERS-1)]

    print("states per layer: ", LAYERS)
    MAXEPOCH = 125
    for epoch in range(0, MAXEPOCH):
        # number of times a negative example has higher goodness than the positive example
        pairsumerrs = collections.defaultdict(int)
        trainlogcost = 0.0
        # multiplier on all weight changes - decays linearly to zero after MAXEPOCH/2
        epsgain = 1.0 if epoch<MAXEPOCH/2 else (1.0 + 2.0 * (MAXEPOCH - epoch)) / MAXEPOCH

        for batch in range(numbatches):
            data = mnist_data["batchdata"][:, :, batch]  # 100x784
            targets = mnist_data["batchtargets"][:, :, batch]
            data[:, :NUMLAB] = LABELSTRENGTH * targets
            normstates = {0: ffnormrows(data)}
            posprobs = [None] * NLAYERS  # column vector of probs that positive cases are positive.
            negprobs = [None] * NLAYERS  # column vector of probs that negative cases are POSITIVE.
            for l in range(1, NLAYERS - 1):
                states,normstates[l]=layer_io(normstates[l-1], model[l])
                posprobs[l] = logistic( (np.sum(states**2, axis=1, keepdims=True) - LAYERS[l]) / TEMP )

                dCbydin = (1-posprobs[l]) * states # Element-wise multiplication
                # wrong sign: rate at which it gets BETTER not worse. Think of C as goodness.

                meanstates[l] = 0.9 * meanstates[l] + 0.1 * np.mean(states[l])  # Element-wise operations
                mean_meanstates = np.mean(meanstates[l])
                dCbydin = dCbydin + LAMBDAMEAN * (mean_meanstates - meanstates[l])  
                # This is a regularizer that encourages the average activity of a unit to match that for
                # all the units in the layer. Notice that we do not gate by (states>0) for this extra term.
                # This allows the extra term to revive units that are always off.  May not be needed.

                posdCbydweights[l] = normstates[l - 1].T @ dCbydin
                posdCbydbiases[l] = np.sum(dCbydin, axis=0)

            # NOW WE GET THE HIDDEN STATES WHEN THE LABEL IS NEUTRAL AND USE THE NORMALIZED HIDDEN STATES AS
            # INPUTS TO A SOFTMAX.  THIS SOFTMAX IS USED TO PICK HARD NEGATIVE LABELS

            ones_array = np.ones((batch_size, LAYERS[-1]), dtype=DTYPE) # dummy label in 1st 10 columns
            data[:, :NUMLAB] = LABELSTRENGTH * ones_array[:, :NUMLAB] / LAYERS[-1]
            normstates = {0: ffnormrows(data)}
            for l in range(1, NLAYERS - 1):
                states, normstates[l] = layer_io(normstates[l-1], model[l])
            labin = np.tile(model[NLAYERS-1]['biases'], (batch_size, 1))
            for l in range(MINLEVELSUP, NLAYERS - 1):
                labin += normstates[l] @ model[l]['supweights']
                # normstates seems to work better than states for predicting the label

            max_labin = np.max(labin, axis=1, keepdims=True)
            labin -= np.tile(max_labin, (1, LAYERS[-1]))
            unnormlabprobs = np.exp(labin)
            sum_unnormlabprobs = np.sum(unnormlabprobs, axis=1, keepdims=True)
            trainpredictions = unnormlabprobs / np.tile( sum_unnormlabprobs, (1, LAYERS[-1]) )
            correctprobs = np.sum(trainpredictions * targets, axis=1)
            trainlogcost += np.sum(-np.log(TINY+correctprobs))/numbatches # per batch (not per case)

            trainguesses = np.argmax(trainpredictions, axis=1)
            targetindices = np.argmax(targets, axis=1)
            trainerrors = np.sum(trainguesses != targetindices)

            dCbydin = targets - trainpredictions
            # dCbydbiases[-1] = np.sum(dCbydin[-1], axis=0)

            for l in range(MINLEVELSUP, NLAYERS - 1):
                dCbydsupweights = normstates[l].T @ dCbydin
                supweightsgrad[l] = \
                    DELAY * supweightsgrad[l] + (1 - DELAY) * dCbydsupweights / batch_size
                model[l]['supweights'] = model[l]['supweights'] + epsgain * EPSILONSUP * \
                    (supweightsgrad[l] - SUPWC * model[l]['supweights'])
            # HACK: it works better without predicting the label from the first hidden layer.

            # NOW WE MAKE NEGDATA
            negdata = data
            labinothers = (labin - 1000 * targets)  # big negative logits for the targets so we do not choose them
            chosen_labels = choosefrom( softmax(labinothers) )
            negdata[:, :NUMLAB] = LABELSTRENGTH * chosen_labels
            normstates = {0: ffnormrows(negdata)}
            for l in range(1, NLAYERS - 1):
                states, normstates[l] = layer_io(normstates[l-1], model[l])
                sum_states_l_squared = np.sum(states**2, axis=1, keepdims=True)
                # negprobs - probability of saying a negative case is POSITIVE.
                negprobs[l] = logistic( (sum_states_l_squared - LAYERS[l]) / TEMP)  
                dCbydin = -np.tile(negprobs[l], (1, LAYERS[l])) * states
                negdCbydweights[l] = normstates[l - 1].T @ dCbydin
                negdCbydbiases[l] = np.sum(dCbydin, axis=0)
                pairsumerrs[l] += np.sum(negprobs[l] > posprobs[l])

            for l in range(1, NLAYERS - 1):
                weightsgrad[l] = \
                    DELAY * weightsgrad[l] + (1 - DELAY) * (posdCbydweights[l] + negdCbydweights[l]) / batch_size
                biasesgrad[l] = \
                    DELAY * biasesgrad[l] + (1 - DELAY) * (posdCbydbiases[l] + negdCbydbiases[l]) / batch_size
                model[l]['biases'] = model[l]['biases'] + epsgain * EPSILON * biasesgrad[l]
                model[l]['weights'] += epsgain * EPSILON * (weightsgrad[l] - WC * model[l]['weights'])

                # Optionally, apply equicols(weights{l}) if equicols function is available
                # equicols makes the incoming weight vectors have the same L2 norm for all units in a layer.
                # weights[l] = equicols(weights[l])

        if True:
            print(f"ep: {epoch:3} gain: {epsgain:.3f} trainlogcost: {trainlogcost:.4f} PairwiseErrs:",
                ", ".join([f"{pairsumerrs[l]}" for l in range(1,NLAYERS-1)]))

        if (epoch + 1) % 5 == 0:
            tr_errors, tr_tests = fftest(ffenergytest, mnist_data["batchdata"][:100], mnist_data["batchtargets"], model)
            verrors, vtests = fftest(ffenergytest, mnist_data["validbatchdata"][:100], mnist_data["validbatchtargets"], model)
            print(f"Energy-based errs: Train {tr_errors}/{tr_tests} Valid {verrors}/{vtests}")
            verrors, vtests = fftest(ffsoftmaxtest, mnist_data["validbatchdata"][:100], mnist_data["validbatchtargets"], model)
            print(f"Softmax-based errs: Valid {verrors}/{vtests}")
            print("rms: ", ", ".join([f"{rms(model[l]['weights']):.4f}" for l in range(1, NLAYERS - 1)]))
            #print("rms: ", [rms(model[l]['weights']) for l in range(1, NLAYERS - 1)])
            print("suprms: ", ", ".join([f"{rms(model[l]['supweights']):.4f}" for l in range(MINLEVELSUP, NLAYERS - 1)]))
            # the magnitudes of the sup weights show how much each hidden layer contributes to the softmax.
    return model

if __name__ == "__main__":
    np.random.seed(1234)
    data=mnist.make_batches("MNIST")
    model=train(data)
    tr_errors, tr_tests = fftest(ffenergytest, data["batchdata"], data["batchtargets"], model)
    te_errors, te_tests = fftest(ffenergytest, data["testbatchdata"], data["testbatchtargets"], model)
    print(f"Energy-based errs: Train {tr_errors}/{tr_tests} Test {te_errors}/{te_tests}")
    tr_errors, tr_tests = fftest(ffsoftmaxtest, data["batchdata"], data["batchtargets"], model)
    te_errors, te_tests = fftest(ffsoftmaxtest, data["testbatchdata"], data["testbatchtargets"], model)
    print(f"Softmax-based errs: Train {tr_errors}/{tr_tests} Test {te_errors}/{te_tests}")

