#!/bin/bash	   
#$ -M mparikh@cs.utexas.edu
module load pylauncher
python /home1/02863/mparikh/ml-project/naive_bayes.py -s /scratch/02863/mparikh/data/conf.npy
