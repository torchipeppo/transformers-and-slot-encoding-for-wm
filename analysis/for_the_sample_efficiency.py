"""
Notes:

https://ai.stackexchange.com/questions/21992/how-to-measure-sample-efficiency-of-a-reinforcement-learning-algorithm

Shixiang Gu et al., Q-Prop: Sample-Efficient Policy Gradient with An Off-Policy Critic, ICLR 2017
https://openreview.net/forum?id=SJ3rcZcxl

"""

import numpy as np
import pandas as pd

from common import *

F1TAG = "EVAL-f1_score"

def calculate_first_sample_to_exceed_f1_threshold(thr, use_mean, consistency_win_len=1):
    for arch in RUNS_BY_ARCH:
        if use_mean:
            df, _ = calculate_line(F1TAG, arch)
            rolled = df.rolling(consistency_win_len).min()
            # filter on the minimum of the window, but take original rows
            the_step = df.loc[rolled.value >= thr].head(1).step      # head(1), so it only has one row or is empty
            print(arch)
            if len(the_step) > 0:
                print(the_step.iloc[0])
            else:
                print("N/A")
        
        else:
            the_steps = []
            for run in RUNS_BY_ARCH[arch]:
                df = read_run(run, F1TAG)
                rolled = df.rolling(consistency_win_len).min()
                the_steps.append(df.loc[rolled.value >= thr].head(1).step)
            concatted = pd.concat(the_steps)
            print(arch)
            print("len   ", len(concatted), sep="")
            print("mean  ", concatted.mean(), sep="")
            print("sem   ", concatted.sem(), sep="")


def calculate_f1_auc(step_limit, use_mean):
    for arch in RUNS_BY_ARCH:
        if use_mean:
            df, _ = calculate_line(F1TAG, arch)
            df = df.loc[df.step <= step_limit]
            auc = np.trapz(y=df.value, x=df.step)
            print(arch)
            print(auc)
        
        else:
            the_aucs = []
            for run in RUNS_BY_ARCH[arch]:
                df = read_run(run, F1TAG)
                df = df.loc[df.step <= step_limit]
                df[np.isnan(df)] = 0
                the_aucs.append(np.trapz(y=df.value, x=df.step))
            concatted = pd.Series(the_aucs)
            print(arch)
            print("mean  ", concatted.mean(), sep="")
            print("sem   ", concatted.sem(), sep="")



calculate_first_sample_to_exceed_f1_threshold(0.95, use_mean=False, consistency_win_len=4)
