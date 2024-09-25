from getLlhChoice import getLlhChoice
import numpy as np
import math
from scipy.stats import lognorm, norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

# Script to generate and fit data with CASANDRE process model. In the paper
# associated with CASADRE we explored the recoverability of model parameters
# with changing the unique number of stimulus values (parameter: stimValue)
# and repetitions per stimulus value (parameter: stimReps).

# Parameters used to generate figure 4A & C: 
# guessRate   = 0;
# stimSens    = 1;
# stimCrit    = 0
# uncMeta     = [0.2 04 0.8 1.6 3.2];
# confCrit    = 0.75;
# asymFlag    = 0;

def giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag):
    choiceLlh = getLlhChoice(stimValue, paramVec, calcPrecision, asymFlag)
    return -np.sum(sum(nChoice * np.log(choiceLlh + 1e-12)))

stimValue = np.linspace(-3, 3, 11)  # The different stimulus conditions in units of stimulus magnitude (e.g., orientation in degrees)
stimReps = 200                      # The number of repeats per stimulus

# Set model parameters
guessRate = 0.000     # The fraction of guesses
stimSens = 0.5        # Stimulus sensitivity parameter, higher values produce a steeper psychometric function, strictly positive
stimCrit = 0          # The sensory decision criterion in units of stimulus magnitude (e.g., orientation in degrees)
uncMeta = 0.5         # Meta-uncertainty: the second stage noise parameter, only affects confidence judgments, strictly positive
confCrit = [0.75, 1]  # The confidence criteria, unitless (can include more than 1)
asymFlag = 0          # If set to 1, it allows for asymmetrical confidence criteria and confCrit needs two times as many elements

modelParams = [guessRate, stimSens, stimCrit, uncMeta] + confCrit
modelParamsLabel = ['guessRate', 'stimSens', 'stimCrit', 'uncMeta'] + ['confCrit'] * len(confCrit)

# Set calulation precision
calcPrecision = 100  # Higher values produce slower, more precise estimates. Precision saturates after ~25

# Get model predictions
choiceLlh = getLlhChoice(stimValue, modelParams, calcPrecision, asymFlag)
randNumbers = np.random.rand(stimReps, len(stimValue))

# Simulate choice data
criteria = np.zeros(np.shape(choiceLlh))
for i in range(len(stimValue)):
    criteria_i = np.cumsum(choiceLlh[i])
    for j in range((2 * (len(confCrit)) + 2)):
        criteria[i,j] += criteria_i[j]

nChoice = np.zeros((np.shape(criteria)))

for i in range(stimReps):
    for j in range(len(stimValue)):
        nChoice[j,np.digitize(randNumbers[i,j],criteria[j])] += 1

# Fit simulated data
options = {'disp': False, 'maxiter': 10**5, 'maxfun': 10**5}
obFun = lambda paramVec: giveNLL(paramVec, stimValue, nChoice, calcPrecision, asymFlag)
startVec = [.01, 1, -0.1, 0.5] + list(2 * np.random.rand(len(confCrit)))

# Search bounds:
LB = np.zeros(len(startVec))
UB = np.zeros(len(startVec))

LB[0] = 0; UB[0] = 0.1                  
LB[1] = 0; UB[1] = 10                
LB[2] = -3; UB[2] = 3                   
LB[3] = 0.01; UB[3] = 5
LB[4:] = 0; UB[4:] = 5
paramEst = minimize(obFun, startVec, bounds=list(zip(LB, UB)), options=options).x

# Computations for plotting
stimPlot = np.linspace(stimValue[0], stimValue[-1], 100)
genLlh = getLlhChoice(stimPlot, modelParams, calcPrecision, asymFlag)
fitLlh = getLlhChoice(stimPlot, paramEst, calcPrecision, asymFlag)

midpoint = genLlh.shape[1] // 2
genPF = np.sum(genLlh[:, midpoint:], axis=1)
fitPF = np.sum(fitLlh[:, midpoint:], axis=1)
obsPF = np.sum(nChoice[:, midpoint:], axis=1) / stimReps

# Plot results
fig, axs = plt.subplots(3, 1, figsize=(10, 20))
plt.subplots_adjust(hspace=0.5)

# Subplot 1: Parameter estimates
max_val = max(np.max(modelParams), np.max(paramEst))
min_val = -.5
axs[0].plot([min_val, max_val], [min_val, max_val], 'k--')

for i, (true_param, est_param, label) in enumerate(zip(modelParams, paramEst, modelParamsLabel)):
    axs[0].plot(true_param, est_param, 'o', markerfacecolor=[.8, .8, .8], markersize=10, linewidth=1, label=label)

axs[0].legend(loc='lower right')
axs[0].set_xlabel('Ground truth')
axs[0].set_ylabel('Parameter estimate')
axs[0].set_xlim(min_val, max_val)
axs[0].set_ylim(min_val, max_val)

# Subplot 2: Psychometric functions
axs[1].plot(stimPlot, genPF, 'r-', stimPlot, fitPF, 'k--', linewidth=2)
axs[1].plot(stimValue, obsPF, 'ko', linewidth=1, markerfacecolor=[1, 0, 0], markersize=12)
axs[1].axis([-3, 3, 0, 1])
axs[1].set(xlabel='Stimulus value', ylabel='Proportion clockwise')
axs[1].legend(['ground truth', 'model fit', 'observations'], loc='upper left')

# Subplot 3: Probability distributions
for i in range(genLlh.shape[1]):
    axs[2].plot(stimPlot, genLlh[:, i], 'r-', linewidth=2)
    axs[2].plot(stimPlot, fitLlh[:, i], 'k--', linewidth=2)
axs[2].set(xlabel='Stimulus value', ylabel='Probability')

plt.show()