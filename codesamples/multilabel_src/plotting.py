import numpy as np
import os
import itertools as it
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True

showme=False
if not showme:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from nipy.labs.viz_tools import cm

def SetPlotRC():
    #If fonttype = 1 doesn't work with LaTeX, try fonttype 42.
    plt.rc('pdf',fonttype = 42)
    plt.rc('ps',fonttype = 42)

SetPlotRC()

def ApplyFont(ax):

    ticks = ax.get_xticklabels() + ax.get_yticklabels()

    text_size = 15.0

    for t in ticks:
        t.set_fontname('Times New Roman')
        t.set_fontsize(text_size)

    txt = ax.get_xlabel()
    txt_obj = ax.set_xlabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_ylabel()
    txt_obj = ax.set_ylabel(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

    txt = ax.get_title()
    txt_obj = ax.set_title(txt)
    txt_obj.set_fontname('Times New Roman')
    txt_obj.set_fontsize(text_size)

def plot_confusion(M, cfile, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(M, cmap=plt.cm.jet, 
                    interpolation='nearest')
    
    width, height = M.shape
    [ ax.annotate( "%d"%(int(round(M[x][y]*100.0)),), xy=(y, x), 
            horizontalalignment='center', verticalalignment='center') for x, y in it.product(xrange(width), xrange(height))]
    
    cb = fig.colorbar(res)
    
    plt.xticks(range(width), alphabet[:width], fontsize=15)
    plt.yticks(range(height), alphabet[:height], fontsize=15)
    plt.savefig(cfile+'.pdf', format='pdf')

from nipy.labs.viz import plot_map
import nibabel as nb

from subprocess import Popen, PIPE, call, STDOUT

def plot_brain(x, affine, template, template_affine, imgfile):
    
    new_brain = x
    img = nb.Nifti1Image(new_brain, affine)
    nb.save(img, imgfile+".nii.gz")
    
    #title = imgfile.split("/")[-1]
    #slicer = plot_map(new_brain, affine, anat=template, anat_affine=template_affine, cmap = plt.cm.jet, title=title)
    slicer = plot_map(new_brain, affine, anat=template, anat_affine=template_affine, cmap=cm.cold_hot, black_bg=True)#.cm.jet
    slicer.contour_map(template, template_affine, cmap=plt.cm.binary, black_bg=True)# plt.cm.Greys
    #plt.show()
    #plt.savefig(imgfile+'.png', format='png')
    plt.savefig(imgfile+'.pdf', format='pdf', facecolot='k', edgecolor='k')
    
def load_concepts(concept_file = "data/mental_concepts.txt"):
    concepts = []
    
    with open(concept_file, 'r') as infile:
        for line in infile:
            concepts.append(line.rstrip('\n').replace(" ", "_"))
    return concepts

from info import clean_model, clean_scorer
from glob import glob
indirectory = os.path.expanduser(os.path.join("~", "workdir", "data", "may_openfMRI"))
#maskfile  = os.path.join(indirectory,  "goodvoxmask.nii.gz")
mainfile   = os.path.join(indirectory,  "zstat_run1.nii.gz")
#maskfile = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "raw", "goodvoxmask.nii.gz"))

def plot_all_brains(mainfile=mainfile):
    ''' look at results of cv and plot it '''
    
    # parse to get general skeleton
    #saveinfo, runinfo, verbose = parser()
    datadir = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "preprocessed"))
    
    # setup resultfile
    #cv_dir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "results", "doublecv"))
    cv_dir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "results", "wardcv"))
    resultdir = os.path.join(cv_dir, "processed")
    
    # check which models have completed training
    infiles = [os.path.split(ss)[-1] for ss in glob(resultdir+"/*.npz")]
    run_models = [ss.split('.')[0] for ss in infiles if ("majority" not in ss and "results" not in ss)] # all model without npz
    
    # get mask
    mainimg = nb.load(mainfile)
    img_affine = mainimg.get_affine()
    
    templatefile = os.path.expanduser(os.path.join("~", "workdir", "data", "standard", "MNI152lin_T1_2mm.nii.gz"))
    templateimg = nb.load(templatefile)
    template = templateimg.get_data()
    template_affine = templateimg.get_affine()
    
    # get concept list
    targetfile = os.path.join(datadir, "targets_run1.npz")
    concept_list = [concept.rstrip("\r") for concept in np.load(targetfile)["concepts"] ]
    
    for all_model in run_models:
        print all_model
        model  = clean_model(all_model)
        scorer = clean_scorer(all_model)
        
        # load data
        datafile = os.path.join(resultdir, all_model+".npz")
        data = np.load(datafile)
        brain = data["mean_W"] # (Dx, Dy, Dz, L)
            
        # create weight matrix image
        print "plotting %s"%(all_model,)
        imgfile = os.path.join(resultdir, all_model+"_brain")
        try: os.makedirs(imgfile)
    	except: pass
        
        nbrain = brain.shape[-1]
        #print brain.shape
        for idx in range(nbrain):
            brainfile = os.path.join(imgfile, concept_list[idx])
            plot_brain(brain[:,:, :, idx], img_affine, template, template_affine, brainfile)
    	   #plot_brain(W, img_affine, template, template_affine, imgfile, alphabet=concept_list)
            
    print "done!"

def single_plot(plot_data, outfile, legend, inlabel, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    
    K, L = plot_data.shape
    ind = np.arange(L)  # the x locations for the groups
    width = 0.8/K       # the width of the bars
    pickme = ['r','b','g', 'k', 'c', 'm', 'y']

    fig, ax = plt.subplots()
    rects = []
    for idx in range(K):
        rects.append( ax.bar(ind+idx*width, plot_data[idx], width, color=pickme[idx]))# , yerr=var_data[idx]) )
    
    # add some
    ax.set_ylabel(inlabel, fontsize=15)
    #ax.set_xlabel("Cognitive Concepts", fontsize=15)
    ax.set_xticks(ind+0.4)
    ax.set_xticklabels( alphabet[:L], fontsize=15)
    ax.legend( [rec[0] for rec in rects], legend , fontsize=15)
    
    ax.axis("tight")
    new_axis = list(ax.axis())
    new_axis[1] += width*2
    new_axis[3] = 1.0
    ax.axis(new_axis)
    
    #def autolabel(rects):
    #    # attach some text labels
    #    for rect in rects:
    #        height = rect.get_height()
    #        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
    #                ha='center', va='bottom')
    # 
    #autolabel(rects1)
    #autolabel(rects2)
    
    
    #plt.tight_layout()
    ApplyFont(plt.gca())
    aspect = 7.0
    ax.set_aspect(aspect, 'box', 'C')
    
    if showme:
        plt.show()
        ddd
    else:
        plt.savefig(outfile)
        plt.clf()
        plt.close("all")
     
 
from info import clean_model, clean_scorer   
def plot_label_scores():
    ''' Plot per label performance 
    For every metric
    label1-[loss1, loss2, loss3] | ... | labelN-[loss1, loss2, loss3]
    '''
    # parse to get general skeleton
    #saveinfo, runinfo, verbose = parser()
    datadir = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "preprocessed"))
    cv_dir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "results", "wardcv"))
    resultdir = os.path.join(cv_dir, "processed")
    labelresultfile = os.path.join(resultdir, "all_label_results.npz")
    label_results = np.load(labelresultfile)
    
    hamming_models =label_results["hamming_models"] # models trained to optimize hamming, pick by def
    all_models = ["l2_hinge_hamming", "l2_log_hamming", "l2_squared_hamming", "probability_majority_hamming"]
    #ranking_models = label_results["ranking_models"]
    label_scores = label_results["label_scores"] # set of scores we used
    label_losses = label_results["label_losses"]   # and set of losses
    L = label_results[hamming_models[0]+"_scores"].shape[-1]
    
    # setup output directory for pdfs
    lossdir = os.path.join(resultdir, "label_scores")
    try: os.makedirs(lossdir)
    except: pass
    
    #legend = [clean_model(model)[3:].title() for model in all_models]
    legend = ["SVM", "Logistic", "Ridge", "Popularity"]
    Losses = ['Accuracy', 'Precision', 'Recall', 'F1Score', '1 - Hamming Loss', '1 - Rank Loss']
    for lx, loss in enumerate(it.chain(label_scores, label_losses)):
        
        outfile = os.path.join(lossdir, "label_scores_"+loss[:-4]+".pdf")
        plot_data = np.zeros((len(all_models), L))
        
        print lx    
        for mx, model in enumerate(all_models):
            if loss in label_scores:
                plot_data[mx] = label_results[model+"_scores"][0, lx]
            else:
                plot_data[mx] = 1.0 - label_results[model+"_losses"][0, lx-len(label_scores)]
            
        # plot it!
        single_plot(plot_data, outfile, legend, Losses[lx])
        
def plot_label_scores2():
    ''' Plot per label performance 
    For every metric
    label1-[loss1, loss2, loss3] | ... | labelN-[loss1, loss2, loss3]
    '''
    # parse to get general skeleton
    #saveinfo, runinfo, verbose = parser()
    datadir = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "preprocessed"))
    cv_dir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "results", "wardcv"))
    resultdir = os.path.join(cv_dir, "processed")
    labelresultfile = os.path.join(resultdir, "all_label_results.npz")
    label_results = np.load(labelresultfile)
    
    hamming_models =label_results["hamming_models"] # models trained to optimize hamming, pick by def
    all_models = ["l2_hinge_hamming", "l2_log_hamming", "l2_squared_hamming", "probability_majority_hamming"]
    #ranking_models = label_results["ranking_models"]
    label_scores = label_results["label_scores"] # set of scores we used
    label_losses = label_results["label_losses"]   # and set of losses
    L = label_results[hamming_models[0]+"_scores"].shape[-1]
    
    # setup output directory for pdfs
    lossdir = os.path.join(resultdir, "label_scores")
    try: os.makedirs(lossdir)
    except: pass
    
    #legend = [clean_model(model)[3:].title() for model in all_models]
    legend = ["SVM", "Logistic", "Ridge", "Popularity"]
    
    which_plot =['Accuracy' 'Precision' 'Recall' 'F1Score']
    plot_data = np.zeros((len(label_scores), len(all_models), L))
    
    for lx, loss in enumerate(label_scores):
        
        outfile = os.path.join(lossdir, "label_scores_"+loss[:-4]+".pdf")
        
        if loss in label_scores:
            ender = "scores"
        else:
            ender = "losses"
            lx -= len(label_scores)
                    
        for mx, model in enumerate(all_models):
            plot_data[mx] = label_results[model+"_"+ender][0, lx]
            var_data[mx]  = label_results[model+"_"+ender][1, lx]
            shuf_data[mx] = label_results[model+"_shuf"+ender][-2, lx]
            
        # plot it!
        single_plot(plot_data, var_data, shuf_data, outfile, legend, ender)
    
def plot_popularity(plotfile="popularity.pdf", ylabel="Fraction of Examples with Process Label", alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
    
    
    datadir = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "preprocessed"))
    # get concept list
    targetfile = os.path.join(datadir, "targets_run1.npz")
    Z = np.load(targetfile)["Z"]
    P = Z.sum(axis=0)/float(len(Z)) # popularity for each label
    
    N = len(P)
    ind = np.arange(N)  # the x locations for the groups
    width = 0.8       # the width of the bars
    plt.ylabel(ylabel, fontsize=15)
    plt.xlabel("Cognitive Process Label", fontsize=15)
    plt.bar(ind, P, width, color='b')
    plt.xticks(ind+width/2.0, alphabet[:N], fontsize=15)
    
    #plt.show()
    plt.savefig(plotfile, format='pdf')
    
    plt.axis("tight")
    #new_axis = list(plt.axis())
    #new_axis[1] += width/4*2
    #plt.axis(new_axis)
    
    plt.tight_layout()
    ApplyFont(plt.gca())
    
    if showme:
        plt.show()
    else:
        plt.savefig(plotfile)
        plt.clf()
        plt.close("all")
    
    
if __name__ == "__main__":
    if 0:
        plot_all_brains()
    else:
        #plot_label_scores()
        plot_popularity()
    
