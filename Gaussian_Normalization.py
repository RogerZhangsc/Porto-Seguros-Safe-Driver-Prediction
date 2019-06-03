import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter, OrderedDict

def RationalApproximation(t:float)->float: 
    c = [2.515517, 0.802853, 0.010328]
    d = [1.432788, 0.189269, 0.001308]
    return t - ((c[2]*t + c[1])*t + c[0]) / (((d[2]*t + d[1])*t + d[0])*t + 1.0)

def NormalCDFInverse(p:float) -> float:

    if (p <= 0.0 or p >= 1.0):
        raise Exception('0<p<1. The value of p was: {}'.format(p))
    if (p < 0.5):
        return -RationalApproximation(math.sqrt(-2.0*math.log(p)) )
    return RationalApproximation( math.sqrt(-2.0*math.log(1-p)) )

def vdErfInvSingle01(x:float) -> float:
    if x == 0:
        return 0
    elif x < 0:
        return -NormalCDFInverse(-x)*0.7
    else:
        return NormalCDFInverse(x)*0.7

def buildRankGaussTrafo(dataIn:list) -> OrderedDict:
    trafoMap = OrderedDict()
    hist = Counter(dataIn)
    if len(hist) == 0:
        pass
    elif len(hist) == 1:
        key = list(hist.keys())[0]
        trafoMap[key] = 0.0
    elif len(hist) == 2:
        keys = sorted(list(hist.keys()))
        trafoMap[keys[0]] = 0.0
        trafoMap[keys[1]] = 1.0
    else:
        N = cnt = 0
        for it in hist:
            N += hist[it]
        assert (N == len(dataIn))
        mean = 0.0
        for it in sorted(list(hist.keys())):
            rankV = cnt / N
            rankV = rankV * 0.998 + 1e-3
            rankV = vdErfInvSingle01(rankV)
            assert(rankV >= -3.0 and rankV <= 3.0)
            mean += hist[it] * rankV
            trafoMap[it] = rankV
            cnt += hist[it]
        mean /= N
        for it in trafoMap:
            trafoMap[it] -= mean
    
    return trafoMap

def binary_search(keys, val):
    start, end = 0, len(keys)-1
    while start+1 < end:
        mid = (start + end) // 2
        if val < keys[mid]:
            end = mid
        else:
            start = mid
    return keys[start], keys[end]

def applyRankTrafo(dataIn:list, trafoMap:dict) -> dict:
    dataOut = []
    keys = list(trafoMap.keys())
    if len(keys) == 0:
        raise Exception('No transfermation map')
    for i in range(len(dataIn)):
        val = dataIn[i]
        trafoVal = 0.0
        if val <= keys[0]:
            trafoVal = trafoMap[keys[0]]
        elif val >= keys[-1]:
            trafoVal = trafoMap[keys[-1]]
        elif val in trafoMap:
            trafoVal = trafoMap[val]
        else:
            lower_key, upper_key = binary_search(keys, val)
            x1, y1 = lower_key, trafoMap[lower_key]
            x2, y2 = upper_key, trafoMap[upper_key]
            
            trafoVal = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        dataOut.append(trafoVal)
    return dataOut  
