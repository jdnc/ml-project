'''
Created on Sep 6, 2013
fixed constants

@author: ook59
'''
'''
Run Commmands:
Launch training script:
launch -s run_script -e 4way -p 200 -r 05:00:00 -n cvrun2 -m sanmi.k@gmail.com
'''

import itertools as it
# known models

base_models = [ss+"_majority" for ss in 'one', 'probability']#'zero', 
#reg_pre = ['l1', 'l2']
reg_pre = ['l2']
reg_base = ['hinge', 'squared', 'log']
reg_weight = ['weighted', '']
reg_models = ["_".join(s for s in (a, b, c)) for a, b, c in it.product(reg_pre, reg_weight, reg_base)]
known_models = base_models + reg_models

def clean_name(model_all):
    # regularization
    if "zero" in model_all:
        return "zero_majority"
    if "one" in model_all:
        return "one_majority"
    if "prob" in model_all:
        return "probability_majority"
    
    
    if 'l1' in model_all:
        model = 'l1'
    else:
        model ='l2'
    # weighted or not
    if 'weight' in model_all:
        model+='_weighted' 
    # loss func
    if 'hinge' in model_all or 'svm' in model_all:
        model+='_hinge'
    elif 'square' in model_all or "regress" in model_all:
        model+='_squared'
    else:
        model+='_log'
    return model

def clean_model(model_all):
    return clean_name(model_all)

def clean_scorer(model_all):
        # loss func
    if 'rank' in model_all or 'svm' in model_all:
        scorer='rankloss'
    else:
        scorer = "hamming"
    return scorer

known_losses = ['hamming', 'acc', 'lab_acc', 'prec', 'recall', 'f1score', 'one_err', 'coverage', 'rkloss', 'rklossb']
print_losses = ['hamming', 'lab_acc', 'prec', 'recall', 'f1score', 'rklossb']
print_lablosses = ['hamming_sub', 'lab_acc_sub', 'prec_sub', 'recall_sub', 'f1score_sub', 'rklossb_sub']

'''
    Note: Loss (Lower is better) -OR - Score (Higher is better)
    metrics:
    {
    # set accuracy (full set of clases)
    hamming : # of wrong / total predicted : LOSS
    acc: # of correct / total predicted : SCORE
    
    # label accuracy (computed at label level for each example)
    label_acc: label accuracy : SCORE
    prec: precision : SCORE
    recall: recall : SCORE
    fiscore: F1 score : SCORE
    
    one_err: one error, count if top prediction is not in true label set : LOSS
    coverage: how far in ranking to get all true labels : LOSS
    rkloss: rank loss : LOSS
    rklossb: corrected rank loss = 1 - AUC : LOSS
'''
loss_flag = {'hamming':True, 
           'acc': False, 
           'lab_acc': False, 
           'prec': False, 
           'recall':False, 
           'f1score': False, 
           'one_err': True, 
           'coverage': True, 
           'rkloss': True,
           'rklossb': True}


