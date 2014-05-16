The master branch has all the code for the single-label and multi-label case. Thus all the branches dealing with them - efficiency, multi may be safely ignored.

For the data and experiments for the transfer learning case, please look at the branch transfer.


The master branch has the following modules and directories:

Contents
--------

- codesamples/  ....................  all the code borrowed from sanmi
- data/         ....................  all the data including masks necessary for the experiments
- neurosynth/   ....................  the entire neurosynth package, not really required if neurosynth already installed
- references/   ....................  all the papers relevant as well as our project report
- experiment.py ....................  module for filtering out studies, based on activated voxels, etc
- preprocess.py ....................  module to get data in right format, synthesize niimg from coords, etc
- single_label.py ..................  module that has all the single label classiifiers
- multi_label.py  ..................  module that has all the multi label classifiers
- utils.py        ..................  metrics for the multi-label algorithms (borrowed from Sanmi)
- generate_full_confmat.py .........  generate the color-coded confusion matrix for single label case
- histogram.py  ....................  generate the bar chart for label distribution in multi label case

