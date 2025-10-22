import numpy as np
import pandas as pd

def rsi_signal(rsi):
    if rsi < 30: return 1  # buy
    if rsi > 70: return 2  # sell
    return 0              # hold

def macd_signal(macd, signal):
    if macd > signal: return 1
    if macd < signal: return 2
    return 0

def bb_signal(close, lower, upper):
    if close <= lower: return 1
    if close >= upper: return 2
    return 0

def majority_vote(*signals):
    # returns 0=hold,1=buy,2=sell by majority; ties -> hold
    vals, counts = np.unique([s for s in signals], return_counts=True)
    return int(vals[counts.argmax()])