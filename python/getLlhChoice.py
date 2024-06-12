import numpy as np
import math
from scipy.stats import lognorm, norm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import curve_fit

def getLlhChoice(stimValue, modelParams, calcPrecision, asFlag):
    
# Takes as input: 

# stimulusValue - different stimulus conditions in units of stimulus magnitude
# modelParams   - CASANDRE" paramters in order [guessRate, stimSens, stimCrit, uncMeta, confCrit]
# calcPrecision - calcPrecision(1) is the sample rate; 25 + recommended
# asymFlag      - fit CASANDRE with or without asymmetrical confidence criteria 
 
# Output:
# choiceLlh     - likelihood of each choice (2 * N confidence levels x N stimValues) 

    # Decode function arguments
    
    stimVal   = stimValue                    # The different stimulus conditions in units of stimulus magnitude (e.g., orientation in degrees)
    noiseSens = 1                            # If the sensory noise is set to 1, then distributions of decision variable and confidence variable can be compared directly
    guessRate = modelParams[0]               # The fraction of guesses
    stimSens  = modelParams[1]               # Stimulus sensitvity parameter, higher values produce a steeper psychometric function, strictly positive
    stimCrit  = modelParams[2]               # The sensory decision criterion in units of stimulus magnitude (e.g., orientation in degrees)
    uncMeta   = modelParams[3]               # Meta-uncertainty: the second stage noise parameter, only affects confidence judgments, strictly positive
    confCrit  = np.cumsum(modelParams[4:])   # The confidence criteria, unitless
    asymFlag  = asFlag;

    # Set calculation precision
    sampleRate = calcPrecision
    
    ## Compute model prediction
    # Step 0 - rescale sensory representation by sensitivity parameter
    sensMean = stimVal*stimSens
    sensCrit = stimCrit * stimSens
    
    llhC = np.zeros((len(stimVal),((2*len(confCrit))+2)))
    
    for iC in range(len(stimVal)):
        
        ## Compute llh of each response alternative
        # Step 1 - sample decision variable denominator in steps of constant cumulative density
        muLogN = np.log((noiseSens ** 2) / np.sqrt(uncMeta ** 2 + noiseSens ** 2))
        sigmaLogN = np.sqrt(np.log((uncMeta ** 2) / (noiseSens ** 2) + 1))
        dv_Den_x = lognorm.ppf(np.linspace(.5 / sampleRate, 1 - (.5 / sampleRate), sampleRate), s=sigmaLogN, scale=np.exp(muLogN))    
        
        # Step 2 - compute choice distribution under each scaled sensory distribution
        # Crucial property: linear transformation of normal variable is itself normal variable
        # Trick: we take inverse of denominator to work with products instead of ratios
        mu = (1 / dv_Den_x) * (sensMean[iC] - sensCrit)
        sigma = (1 / dv_Den_x) * noiseSens    
        
        mu = mu[:, np.newaxis]                                      # Reshape mu to (sampleRate, 1)
        sigma = sigma[:, np.newaxis]                                # Reshape sigma to (sampleRate, 1)
        confCrit = np.array(confCrit).flatten()                     # Make sure confCrit is a flat array

        x = sorted(np.concatenate([-confCrit, [0], confCrit]))
        
        try:
            if asymFlag == 1:
                confCrit = modelParams[4:]
                x = np.concatenate([np.sort(-np.cumsum(confCrit[1:numel(confCrit)/2])), [0], np.sort(np.cumsum(confCrit[numel(confCrit)/2 + 1:]))])
        except:
            pass          
        
        P = norm.cdf(np.tile(x, (sampleRate, 1)), np.tile(mu, (1, len(x))), np.tile(sigma, (1, len(x))))
        
        # Step 3 - average across all scaled sensory distributions to get likelihood functions
        ratio_dist_p = np.mean(P, axis=0)
        
        llhC_i = []
        for iX in range(len(x)+1):
            # print(len(x))
            if iX == 0:
                llhC[iC,iX] += ((guessRate / (len(x) + 1)) + (1 - guessRate) * ratio_dist_p[0])
            elif 0 < iX < len(x):
                llhC[iC,iX] += ((guessRate / (len(x) + 1)) + (1 - guessRate) * (ratio_dist_p[iX] - ratio_dist_p[iX - 1]))
            else:  
                llhC[iC,iX] += ((guessRate / (len(x) + 1)) + (1 - guessRate) * (1 - ratio_dist_p[iX - 1]))
            
    return llhC
