#!/bin/bash	   
#$ -M vsub@cs.utexas.edu
#$ -N NaiveBayes
#$ -o naive_bayes.out
module load pylauncher
#python /home1/02869/vsub/ml-project/multi-check.py
python /home1/02869/vsub/ml-project/naive_bayes.py -c /scratch/02869/vsub/data/docdict.txt  -f /scratch/02869/vsub/data/features.txt -s $SCRATCH/ 
