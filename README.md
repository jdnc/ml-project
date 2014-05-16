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

References
----------
- Tal Yarkoni, Russell A Poldrack, Thomas E Nichols, David C Van Essen, and Tor D Wager. Large-scale automated synthesis of human functional neuroimaging data. Nature methods, 8(8):665–670, 2011
- Koyejo Oluwasanmi and Russell A Poldrack. Decoding cognitive processes from functional mri. NIPS workshop on Machine Learning and Interpretation in Neuroimaging, 2013
- Sinno Jialin Pan and Qiang Yang. A survey on transfer learning. Knowledge and Data Engineering, IEEE Transactions on, 22(10):1345–1359, 2010.

Other links
------------
Blog : http://ml-glob.blogspot.com

Bugs
----
If any bugs please open an issue on this github repo or shoot a mail:
  
  Maintainer : Madhura Parikh (@jdnc)
  
  Contact: madhuraparikh@gmail.com
