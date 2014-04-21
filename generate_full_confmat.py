#!/usr/bin/env python
"""
Code to generate the classification accuracy values for each pair of terms from the cross validated confusion matrices.
"""

import numpy as np
import os

def getPairAccuracy(cv_confmats):
  allAccuracy=[]
  for cm in cv_confmats:
    accuracy=(cm[0,0]+cm[1,1])*1.0/(cm[0,1]+cm[1,0])
    allAccuracy.append(accuracy)
  return (np.mean(allAccuracy),np.std(allAccuracy))

def getAllConfMatFiles(folderLoc):
  termpairs_confmat_nparrays=[]
  for afile in os.listdir(folderLoc)
    if "_vs_" in afile:
      termpairs_confmat_nparrays.append(afile)
  return termpairs_confmat_nparrays

def create2DpairAccuracy(folderLoc):
  pairwise_accuracy={} #{term1: {term2:(acc,std), t3:(), ..}..}
  allPairsFiles=getAllConfMatFiles(folderLoc)
  for pair_cv_cm_file in allPairsFiles:
    terms=pair_cv_cm_file[:-4].split("_vs_")
    term1=terms[0]
    term2=terms[1]
    pair_cv_confmats=np.load(pair_cv_cm_file)
    (accuracy,stddev)=getPairAccuracy(pair_cv_confmats)
    if pairwise_accuracy.get(term1) is None:
      pairwise_accuracy[term1]={}
    if pairwise_accuracy.get(term2) is None:
      pairwise_accuracy[term2]={}
    pa_t1=pairwise_accuracy[term1]
    pa_t1[term2]=(accuracy, stddev)
    pairwise_accuracy[term1]=pa_t1
    pa_t2=pairwise_accuracy[term2]
    pa_t2[term1]=(accuracy, stddev)
    pairwise_accuracy[term2]=pa_t2
    #pairwise_accuracy[term1].append((term2, accuracy, stddev))
    #pairwise_accuracy[term2].append((term1, accuracy, stddev))
  return pairwise_accuracy

def createOutputMatrix(pairwise_accuracy, outputfile):
  order=[]
  for term in pairwise_accuracy:
    order.append[term]
  width=len(order)
  height=len(order)
  pair_mat=np.zeros((width,height))
  for row in range(1,len(order)):
    row_vals=np.zeros((width))
    row_term_accs=pairwise_accuracy[order[row]]
    for col in range(1,row):
      pair_mat[row,col]=row_term_accs[order[col]]
  pair_col_mean=np.mean(pair_mat, axis=0)
  for i in range(0,len(pair_col_mean)):
    pair_mat[i,i]=pair_col_mean[i]
