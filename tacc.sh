#!/bin/bash	   
#$ -M mparikh@cs.utexas.edu
#$ -N NaiveBayes
#$ -o naive_bayes.out
module load pylauncher
python /home1/02863/mparikh/ml-project/naive_bayes.py -c /scratch/02863/mparikh/data/docdict.txt  -f /scratch/02863/mparikh/data/features.txt -s $SCRATCH/ 
