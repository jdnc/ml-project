#!/usr/bin/env python
"""
Code to generate the classification accuracy values for each pair of terms from the cross validated confusion matrices.

Usage: python generate_full_confmat.py <location of folder containing .npy confusion matrices>

"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def getPairAccuracy(cv_confmats):
  allAccuracy=[]
  for cm in cv_confmats:
    accuracy=((cm[0,0]+cm[1,1])*1.0)/(cm[0,1]+cm[1,0]+cm[0,0]+cm[1,1])
    #print accuracy
    allAccuracy.append(accuracy)
  return (np.mean(allAccuracy),np.std(allAccuracy))

def getAllConfMatFiles(folderLoc):
  termpairs_confmat_nparrays=[]
  for afile in os.listdir(folderLoc):
    if "_vs_" in afile:
      termpairs_confmat_nparrays.append(afile)
  return termpairs_confmat_nparrays

def create2DpairAccuracy(folderLoc):
  pairwise_accuracy={} #{term1: {term2:(acc,std), t3:(), ..}..}
  allPairsFiles=getAllConfMatFiles(folderLoc)
  for pair_cv_cm_file in allPairsFiles:
    terms=pair_cv_cm_file[:-4].split("_")
    term1=terms[0]
    term2=terms[2]
    pair_cv_confmats=np.load(folderLoc+pair_cv_cm_file)
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
  #print pairwise_accuracy
  return pairwise_accuracy

def makePlot(np_acc_matrix, order, plot_name):
  fig = plt.figure(figsize=(20,20))
  plt.clf()
  ax = fig.add_subplot(111, aspect='auto')
  #ax.set_aspect(1)
  res = ax.imshow(np.array(np_acc_matrix), cmap=plt.cm.summer, interpolation='nearest', vmin=50.0, vmax=95.0)
  width = len(np_acc_matrix)
  height = len(np_acc_matrix)

  for x in xrange(width):
    for y in xrange(height):
      if(y>x):
        ax.annotate('', xy=(y, x), horizontalalignment='center', verticalalignment='center')
      else:
        ax.annotate(str(np_acc_matrix[x][y]), xy=(y, x), horizontalalignment='center', verticalalignment='center')

  cb = fig.colorbar(res)
  plt.xticks(range(width), order[:width], rotation=30)
  plt.yticks(range(height), order[:height])
  plt.savefig(plot_name, format='jpg',bbox_inches='tight')

def getAvgTermAcc(row_term_accs):
  termAcc=[]
  for t2 in row_term_accs:
    termAcc.append(row_term_accs[t2][0])
  return np.mean(termAcc)

def createOutputMatrix(pairwise_accuracy, outputfile):
  order=[]
  for term in pairwise_accuracy:
    order.append(term)
  width=len(order)
  height=len(order)
  pair_mat=np.zeros((width,height))
  for row in range(0,len(order)):
    row_vals=np.zeros((width))
    row_term_accs=pairwise_accuracy[order[row]]
    for col in range(0,row):
      if row_term_accs.get(order[col]) is None:
        pair_mat[row,col]=0.0
      else:
        pair_mat[row,col]=np.around((row_term_accs[order[col]][0]*100),2)
    avg_term_acc=getAvgTermAcc(row_term_accs)
    pair_mat[row,row]=np.around((avg_term_acc*100),2)
  print order
  print pair_mat
  makePlot(pair_mat, order, outputfile)

def main():
  folderLoc="output/nb_sample"
  if(len(sys.argv)>1):
    folderLoc=sys.argv[1]
  folderLoc+="/"
  pairwise_accuracy=create2DpairAccuracy(folderLoc)
  outputfile=folderLoc+"confusion_matrix.jpg"
  createOutputMatrix(pairwise_accuracy, outputfile)

if __name__=="__main__":
  main()

