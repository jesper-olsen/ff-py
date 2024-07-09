import collections
from jax import random, jit
from jax.scipy.special import expit as logistic # 1/(1+exp(-x))
import jax.numpy as jnp
import mnist

TINY = 1e-20  # for preventing divisions by zero
DTYPE=jnp.float32
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
    return a / (TINY + jnp.sqrt(jnp.mean(a**2, axis=1, keepdims=True))) * jnp.ones((num_comp), dtype=DTYPE)

@jit
def choosefrom(probs, key):
    """vectorized - probabilistically choose neg examples that are more like the targets"""
    batch_size, _ = probs.shape   # batch_size x nlabels
    random_values = random.uniform(key, (batch_size, 1))
    cumulative_probs = jnp.cumsum(probs, axis=1)
    chosen_labels = jnp.argmax(random_values < cumulative_probs, axis=1)
    postchoiceprobs = jnp.zeros_like(probs, dtype=DTYPE)
    postchoiceprobs = postchoiceprobs.at[jnp.arange(batch_size), chosen_labels].set(1.0)
    return postchoiceprobs

def softmax(scores):
    exp_scores = jnp.exp(scores)
    return exp_scores / jnp.sum(exp_scores, axis=1, keepdims=True)

def equicols(matrix):
    norms = jnp.linalg.norm(matrix, axis=0)  # Calculate L2 norms of each column
    return matrix / norms  # Normalize each column by its L2 norm

def rms(x) -> float:
    # Assumes x is a matrix, but should work for vectors and scalars too
    return jnp.sqrt(jnp.sum(x**2) / x.size)

@jit
def layer_io(vin, lmodel):
    states = jnp.maximum(0, vin @ lmodel['weights'] + lmodel['biases'])
    normstates = ffnormrows(states)
    return states,normstates

@jit
def ffenergytest(data, model):
    actsumsq = jnp.zeros((len(data), NUMLAB), dtype=DTYPE)
    for lab in range(NUMLAB):
        data = data.at[:, :NUMLAB].set(jnp.zeros((len(data), NUMLAB), dtype=DTYPE))
        data = data.at[:, lab].set(LABELSTRENGTH * jnp.ones(len(data), dtype=DTYPE))
        normstates_lm1 = ffnormrows(data)
        for l in range(1, len(model)):
            states, normstates_lm1 = layer_io(normstates_lm1, model[l])
            if l >= MINLEVELENERGY:
                actsumsq = actsumsq.at[:, lab].add(jnp.sum(states**2, axis=1))
    return jnp.argmax(actsumsq, axis=1)  # guesses

@jit
def ffsoftmaxtest(data, model):
    data = data.at[:, :NUMLAB].set(LABELSTRENGTH * jnp.ones((len(data), NUMLAB), dtype=DTYPE) / NUMLAB)
    normstates = {0: ffnormrows(data)}
    for l in range(1, len(model)):
        _, normstates[l] = layer_io(normstates[l-1], model[l])
    labin = jnp.tile(model[-1]['biases'], (len(data), 1))
    for l in range(MINLEVELSUP, len(model)):
        labin += normstates[l] @ model[l]['supweights']
    labin = labin - jnp.max(labin, axis=1, keepdims=True)
    unnormlabprobs = jnp.exp(labin)
    testpredictions = unnormlabprobs / jnp.sum(unnormlabprobs, axis=1, keepdims=True)
    # logcost += -np.sum(targets * np.log(TINY + testpredictions)) / numbatches
    return jnp.argmax(testpredictions, axis=1)  # guesses

def fftest(f_batch, batchdata, batchtargets, model):
    errors = tests = 0
    for batch in range(len(batchdata)):
        data = batchdata[:, :, batch]
        targets = batchtargets[:, :, batch]
        targetindices = jnp.argmax(targets, axis=1)
        guesses = f_batch(data, model)
        errors += jnp.sum(guesses != targetindices)
        tests += len(guesses)
    return errors, tests

def init_model(LAYERS,key):
    model = [None]
    for l, (fanin, fanout) in enumerate(zip(LAYERS[:-1], LAYERS[1:])):
        key, subkey = random.split(key)
        d = {'weights': (1/jnp.sqrt(fanin))*random.normal(subkey, (fanin, fanout), dtype=DTYPE),
             'biases': jnp.zeros(fanout, dtype=DTYPE),
             'supweights': jnp.zeros((fanout, LAYERS[-1]), dtype=DTYPE)
            }
        model.append(d)
    return model, key

def train(mnist_data, key):
    batch_size, idim, numbatches = mnist_data["batchdata"].shape
    print(f"Batchsize: {batch_size} Input-dim: {idim} #training batches: {numbatches}")

    LAYERS = [idim, 1000, 1000, 1000, NUMLAB]
    NLAYERS = len(LAYERS)
    print("states per layer: ", LAYERS)

    # meanstates - running average of the mean activity of a hidden unit.
    meanstates = {l: 0.5 * jnp.ones(LAYERS[l], dtype=DTYPE) for l in range(1, NLAYERS - 1)}
    model, key = init_model(LAYERS,key)

    # gradients of probability of correct real/fake decision w.r.t. weights & biases - smoothed over minibatches
    posdCbydweights = [None] * NLAYERS
    posdCbydbiases = [None] * NLAYERS
    weightsgrad = [None] + [jnp.zeros((LAYERS[l - 1], LAYERS[l]), dtype=DTYPE) for l in range(1, NLAYERS - 1)]
    biasesgrad = [None] + [jnp.zeros((1, LAYERS[l]), dtype=DTYPE) for l in range(1, NLAYERS - 1)]
    supweightsgrad = [None] + [jnp.zeros((LAYERS[l], LAYERS[-1]), dtype=DTYPE) for l in range(1, NLAYERS - 1)]

    MAXEPOCH = 125
    for epoch in range(0, MAXEPOCH):
        # multiplier on all weight changes - decays linearly to zero after MAXEPOCH/2
        epsgain = 1.0 if epoch<MAXEPOCH/2 else (1.0 + 2.0 * (MAXEPOCH - epoch)) / MAXEPOCH 
        # number of times a negative example has higher goodness than the positive example
        pairsumerrs = collections.defaultdict(int)
        trainlogcost = 0.0

        for batch in range(numbatches):
            data = mnist_data["batchdata"][:, :, batch]  # 100x784
            targets = mnist_data["batchtargets"][:, :, batch]
            data = data.at[:, :NUMLAB].set(LABELSTRENGTH * targets)
            normstates = {0: ffnormrows(data)}
            posprobs = [None] * NLAYERS  # column vector of probs that positive cases are positive.
            negprobs = [None] * NLAYERS  # column vector of probs that negative cases are POSITIVE.

            for l in range(1, NLAYERS - 1):
                states,normstates[l]=layer_io(normstates[l-1], model[l])
                posprobs[l] = logistic((jnp.sum(states**2, axis=1, keepdims=True) - LAYERS[l]) / TEMP)
                replicated_states = jnp.repeat(1-posprobs[l], LAYERS[l]).reshape(-1, LAYERS[l])
                # gradients of goodness w.r.t. total input to a hidden unit.
                dCbydin = replicated_states * states  # Element-wise multiplication
                # wrong sign: rate at which it gets BETTER not worse. Think of C as goodness.
                meanstates[l] = 0.9 * meanstates[l] + 0.1 * jnp.mean(states[l])  # Element-wise operations
                mean_meanstates = jnp.mean(meanstates[l])
                dCbydin = dCbydin + LAMBDAMEAN * (mean_meanstates - meanstates[l])  
                # This is a regularizer that encourages the average activity of a unit to match that for
                # all the units in the layer. Notice that we do not gate by (states>0) for this extra term.
                # This allows the extra term to revive units that are always off.  May not be needed.

                posdCbydweights[l] = normstates[l - 1].T @ dCbydin
                posdCbydbiases[l] = jnp.sum(dCbydin, axis=0)

            # NOW WE GET THE HIDDEN STATES WHEN THE LABEL IS NEUTRAL AND USE THE NORMALIZED HIDDEN STATES 
            # AS INPUTS TO A SOFTMAX.  THIS SOFTMAX IS USED TO PICK HARD NEGATIVE LABELS

            ones_array = jnp.ones((batch_size, LAYERS[-1]), dtype=DTYPE) # dummy label in 1st 10 columns
            data = data.at[:, :NUMLAB].set(LABELSTRENGTH * ones_array[:, :NUMLAB] / LAYERS[-1])
            normstates = {0: ffnormrows(data)}
            for l in range(1, NLAYERS - 1):
                states, normstates[l] = layer_io(normstates[l-1], model[l])
            labin = jnp.tile(model[NLAYERS - 1]['biases'], (batch_size, 1))
            for l in range(MINLEVELSUP, NLAYERS - 1):
                labin = labin+normstates[l] @ model[l]['supweights']
                # normstates seems to work better than states for predicting the label
            max_labin = jnp.max(labin, axis=1, keepdims=True)
            labin = labin - jnp.tile(max_labin, (1, LAYERS[-1]))
            unnormlabprobs = jnp.exp(labin)
            sum_unnormlabprobs = jnp.sum(unnormlabprobs, axis=1, keepdims=True)
            trainpredictions = unnormlabprobs / jnp.tile( sum_unnormlabprobs, (1, LAYERS[-1]) )
            correctprobs = jnp.sum(trainpredictions * targets, axis=1)
            trainlogcost += jnp.sum(-jnp.log(TINY+correctprobs))

            #trainguesses = jnp.argmax(trainpredictions, axis=1)
            #targetindices = jnp.argmax(targets, axis=1)
            #trainerrors = jnp.sum(trainguesses != targetindices)

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
            key, subkey = random.split(key)
            chosen_labels = choosefrom(softmax(labinothers), subkey)
            negdata = negdata.at[:, :NUMLAB].set(LABELSTRENGTH * chosen_labels)
            normstates = {0: ffnormrows(negdata)}

            for l in range(1, NLAYERS - 1):
                states, normstates[l] = layer_io(normstates[l-1], model[l])
                # negprobs - probability of saying a negative case is POSITIVE.
                negprobs[l] = logistic((jnp.sum(states**2, axis=1, keepdims=True) - LAYERS[l]) / TEMP)
                dCbydin = -jnp.tile(negprobs[l], (1, LAYERS[l])) * states
                pairsumerrs[l] += jnp.sum(negprobs[l] > posprobs[l])

                negdCbydweights = normstates[l - 1].T @ dCbydin
                negdCbydbiases = jnp.sum(dCbydin, axis=0)

                weightsgrad[l] = \
                    DELAY * weightsgrad[l] + (1 - DELAY) * (posdCbydweights[l] + negdCbydweights) / batch_size
                biasesgrad[l] = \
                    DELAY * biasesgrad[l] + (1 - DELAY) * (posdCbydbiases[l] + negdCbydbiases) / batch_size
                model[l]['biases'] = model[l]['biases'] + epsgain * EPSILON * biasesgrad[l]
                model[l]['weights'] += epsgain * EPSILON * (weightsgrad[l] - WC * model[l]['weights'])

                # Optionally, apply equicols(weights{l}) if equicols function is available
                # equicols makes the incoming weight vectors have the same L2 norm for all units in a layer.
                # weights[l] = equicols(weights[l])

        if True:
            trainlogcost/=numbatches 
            print(f"ep {epoch:3} gain {epsgain:.3f} trainlogcost {trainlogcost:.4f} PairwiseErrs:",
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

def convert_to_jax(data_dict):
    return {key: jnp.array(value) for key, value in data_dict.items()}

if __name__ == "__main__":
    key = random.PRNGKey(42)
    data=mnist.make_batches("MNIST")
    data = convert_to_jax(data)
    model=train(data, key)
    tr_errors, tr_tests = fftest(ffenergytest, data["batchdata"][:100], data["batchtargets"], model)
    te_errors, te_tests = fftest(ffenergytest, data["testbatchdata"][:100], data["testbatchtargets"], model)
    print(f"Energy-based errs: Train {tr_errors}/{tr_tests} Test {te_errors}/{te_tests}")
    te_errors, te_tests = fftest(ffsoftmaxtest, data["testbatchdata"][:100], data["testbatchtargets"], model)
    print(f"Softmax-based errs: Train {tr_errors}/{tr_tests} Test {te_errors}/{te_tests}")

