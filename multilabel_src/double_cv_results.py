'''
Created on Sep 6, 2013
Compile results and compute resulting CV
@author: ook59
'''
from optparse import OptionParser
import sys, os, time
from info import known_models, print_losses, loss_flag
from info import clean_name
from glob import glob
import numpy as np
import itertools as it
import copy
from utils import Percentiles, Maxentiles
from info import clean_model, clean_scorer
from plotting import plot_confusion

# extract losses, scores names
losses = [ss for ss in print_losses if loss_flag[ss] == True]
scores = [ss for ss in print_losses if loss_flag[ss] != True]
    
def read_cv(runinfo, saveinfo, verbose, losses=losses, scores=scores):
    '''
    Returns means of scores
    '''
    model   = runinfo["model"]
    cv_dir  = runinfo["cv_dir"]
    scorer  = runinfo["scorer"]
    get_w = False if "majority" in model else True
    
    fheader = os.path.join(cv_dir, model+"_"+scorer)
    resultfiles = [os.path.split(ss)[-1] for ss in glob(fheader+"*.npz")]
    Ncv = len(resultfiles)
    if verbose:
        print "num cv = %d"%(Ncv, )
    
    SAVE_LOSSES = np.zeros(( Ncv, len(losses) ))
    SAVE_SCORES = np.zeros(( Ncv, len(scores) ))
    SAVE_CONFUS = []
    
    if get_w:
        SAVE_W = []
    
    for icv, fh in enumerate(resultfiles):
        readfile = os.path.join(cv_dir, fh)
        data = np.load(readfile)
        escore = data['escore'].item()
        for ix, loss in enumerate(losses):
            SAVE_LOSSES[icv, ix] += escore[loss] 
        for ix, score in enumerate(scores):
            SAVE_SCORES[icv, ix] += escore[score] 
            
        SAVE_CONFUS.append(data['econfusion'])
        
        if get_w:    
            W = data["W"]
            SAVE_W.append(W)
    
    mean_score = SAVE_SCORES.mean(axis=0)
    std_score  = SAVE_SCORES.std(axis=0)
    mod_score  = np.vstack((mean_score, std_score))
    
    mean_loss = SAVE_LOSSES.mean(axis=0)
    std_loss  = SAVE_LOSSES.std(axis=0)
    mod_loss  = np.vstack((mean_loss, std_loss))
    if verbose:
        print "scores:", ["%s: %f (%f)"%(t, s, v) for t, s, v in it.izip(scores, mean_score, std_score) ]
        print "losses:", ["%s: %f (%f)"%(t, s, v) for t, s, v in it.izip(losses, mean_loss, std_loss) ]
    
    sum_confus = np.array(SAVE_CONFUS).sum(axis=0)
    
    if get_w:
        mean_W = np.mean( np.array(SAVE_W), axis=0)
    else:
        mean_W = None
    return mod_score, mod_loss, mean_W, sum_confus

def read_shuffle(runinfo, saveinfo, verbose, losses=losses, scores=scores):
    '''
    return 95_percentile thresholds
    '''
    model   = runinfo["model"]
    cv_dir = runinfo["cv_dir"] 
    shuffle_dir = os.path.join(cv_dir, 'shuffle')
    scorer  = runinfo["scorer"]
    get_w = False #if "majority" in model else True
    
    fheader = os.path.join(shuffle_dir, model+"_"+scorer)
    shufflefiles = [os.path.split(ss)[-1] for ss in glob(fheader+"*.npz")]
    
    nshuffle = len(shufflefiles)
    if verbose:
        print "num shuffle = %d"%(nshuffle, )
    
    SAVE_LOSSES = np.zeros(( nshuffle, len(losses) ))
    SAVE_SCORES = np.zeros(( nshuffle, len(scores) ))
    
    if get_w:
        D, K     = np.load(os.path.join(shuffle_dir, shufflefiles[0]))["W"].shape
        weight_sig  = Percentiles(nshuffle, D, K, 0.01)
        weight_vsig = Percentiles(nshuffle, D, K, 0.001)
        weight_max  = Maxentiles(nshuffle, D, K)
        
        
    for icv, fh in enumerate(shufflefiles):
        readfile = os.path.join(shuffle_dir, fh)
        data = np.load(readfile)
        escore = data['escore'].item()
        for ix, loss in enumerate(losses):
            SAVE_LOSSES[icv, ix] += escore[loss] 
        for ix, score in enumerate(scores):
            SAVE_SCORES[icv, ix] += escore[score] 
        
        if get_w:
            W = data["W"]
            weight_sig.add_data(W)
            weight_vsig.add_data(W)
            weight_max.add_data(W)
    
    perc_score  = np.percentile(SAVE_SCORES, 95.0, axis=0, )
    vperc_score = np.percentile(SAVE_SCORES, 99.5, axis=0, )
    max_score   = np.max(SAVE_SCORES, axis=0, )
    aperc_score = np.vstack((perc_score, vperc_score, max_score))
    
    perc_loss   = np.percentile(SAVE_LOSSES, 5.0, axis=0, )
    vperc_loss  = np.percentile(SAVE_LOSSES, 0.5, axis=0, )
    min_loss    = np.min(SAVE_LOSSES, axis=0, )
    aperc_loss  = np.vstack((perc_loss, vperc_loss, min_loss))
    
    if verbose:
        print "95 percentile scores:", ["%s: %f "%(t, s) for t, s in it.izip(scores, perc_score) ]
        print "99.5 percentile scores:", ["%s: %f "%(t, s) for t, s in it.izip(scores, vperc_score) ]
        print "max scores:", ["%s: %f "%(t, s) for t, s in it.izip(scores, max_score) ]
        
        print "5 percentile losses:", ["%s: %f "%(t, s) for t, s in it.izip(losses, perc_loss)]
        print ".5 percentile losses:", ["%s: %f "%(t, s) for t, s in it.izip(losses, vperc_loss)]
        print "min losses:", ["%s: %f "%(t, s) for t, s in it.izip(losses, min_loss)]
    
    # select percentile as second to last items
    if get_w:                                                    
        sig_W  = weight_sig.results() # .05 level
        vsig_W = weight_vsig.results() # .005 level
        max_W = weight_max.results() # .005 level
        result_sig_W = np.dstack((sig_W, vsig_W, max_W)) # (D, K, 2 x 3)
    else:
        result_sig_W = None
        
    return aperc_score, aperc_loss, result_sig_W

def read_base(runinfo, saveinfo, verbose):
    copy_run = copy.copy(runinfo)
    copy_run["model"] = "*majority*"
    return read_shuffle(copy_run, saveinfo, verbose)
    

def parser(def_model='l2svm', def_loss='f1score', def_verbose=True):
    
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-m", "--model", dest="model", default='l2 square', type='string',
                      #choices=known_models, 
                      help="model selection from {%s}"%', '.join(s for s in known_models))
    parser.add_option("-i", "--indir", dest="indir",
                  help="input directory", metavar="INDIR")
    parser.add_option("-o", "--outdir", dest="outdir",
                  help="output directory", metavar="OUTDIR")
    parser.add_option("-l", "--lscore", dest="scorer", default='hamming', type='choice',
                      choices=['hamming', 'rankloss'], help="score type, hamming or rankloss (def=rankloss)")
    #parser.add_option("-s", "--shuffle", dest="shuffle", default=False,
    #                  action="store_true", help="shuffle labels (for random baseline)")
    parser.add_option("-t", "--toy", default="0", type="choice",
                      choices=["0", "1"], dest="istoy", 
                      help='generate toy data instead of real data (-t 1) or no (-t 0)' )
    parser.add_option("-q", "--quiet", default=False,
                      action="store_true", dest="quiet")
    
    (options, args) = parser.parse_args()
    
    # extract options
    model_all   = options.model # model type
    #shuffle = options.shuffle # shuffle yes/no
    verbose = not options.quiet
    scorer  = options.scorer
    istoy   = int(options.istoy) # yes/no running smaller test data
    
    if istoy:
        cv_dir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "toy", "results", "doublecv"))
    else:
        cv_dir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "results", "doublecv"))
    
    resultdir = os.path.join(cv_dir, "processed")
        
    #if shuffle:
    #    cv_dir = os.path.join(cv_dir, "shuffle")
    
    if options.indir is not None:
        cv_dir = options.indir
    if options.outdir is not None:
        resultdir = options.outdir
    
    # make result directory
    try:
        os.makedirs(resultdir)
    except:
        pass
    
    # extract model name
    model = clean_name(model_all)
    
    saveinfo = {'resultdir':resultdir}
    runinfo  = {'model':model, "cv_dir":cv_dir, "scorer":scorer}#, "shuffle":shuffle}
    
    return saveinfo, runinfo, verbose
    

def get_single(saveinfo, runinfo, verbose):
    if verbose: 
        print "get CV results...\nmodel=%s, scoring=%s"%(runinfo["model"], runinfo["scorer"])
        T = time.time()    
    # return data for this model
    if verbose:
        print "running cv"
    mod_scores, mod_losses, mean_W, sum_confus = read_cv(runinfo, saveinfo, verbose)
    #mod_results = mod_score, mod_loss, mean_W
    # return data for shuffle model
    if verbose:
        print "running shuffle"
    model  = runinfo['model']
    scorer = runinfo['scorer']
    if model.endswith("majority"):
        copy_run = copy.copy(runinfo)
        copy_run["model"] = "*majority*"
    else:
        copy_run = runinfo
    shuf_scores, shuf_losses, shuf_W = read_shuffle(copy_run, saveinfo, verbose)
    
    get_w = False #if "majority" in model else True
    if get_w:
        dims = float(np.prod(mean_W.shape))
        sparse     = (np.abs(mean_W)>1E-3).sum()/dims
        sparse_s   = np.logical_or( mean_W < shuf_W[:,:, 0], mean_W > shuf_W[:,:, 1]).sum()/dims
        sparse_ss  = np.logical_or( mean_W < shuf_W[:,:, 2], mean_W > shuf_W[:,:, 3]).sum()/dims
        sparse_max = np.logical_or( mean_W < shuf_W[:,:, 4], mean_W > shuf_W[:,:, 5]).sum()/dims
        sparsity   = 1.0 - np.array([sparse, sparse_s, sparse_ss, sparse_max ])
    else:
        sparsity   = np.ones(4)
        
    # save relevant results to disk
    resultdir = saveinfo["resultdir"]
    npfile = os.path.join(resultdir, model+"_"+scorer)
    np.savez(npfile, scores=scores, losses= losses, mod_scores=mod_scores, mod_losses=mod_losses,\
             mean_W=mean_W, shuf_scores=shuf_scores, shuf_losses=shuf_losses, shuf_W=shuf_W, \
             sum_confus=sum_confus, sparsity=sparsity)
    
    # create and save confusion matrix image
    confile = os.path.join(resultdir, model+"_"+scorer+"_confusion")
    plot_confusion(sum_confus, confile)
    
    if verbose: print "CV results time = %s"%(time.time()-T,) 
    return mod_scores, mod_losses, shuf_scores, shuf_losses, sparsity
    #shuf_results = shuf_SCORES, shuf_LOSSES, shuf_W
    # return data for base shuffle
    #if verbose:
    #    print "running base"
    #base_SCORES, base_LOSSES, _ = read_base(runinfo, saveinfo, verbose)
    #base_results = base_SCORES, base_LOSSES
    

def get_all(saveinfo, runinfo, verbose):
    '''
    For all models / scoring
    print:
    mean / std score, not_sig / sig / very sig
    save:
    + sig levels
    mean map, sig_map, very_sig_map, sig_levels
    '''
    
    # setup resultfile
    resultdir = saveinfo["resultdir"]
    resultfile = os.path.join(resultdir, "resultfile.txt")
    fh = open(resultfile, 'wb')
    
    # check which models have completed training
    cv_dir = runinfo["cv_dir"]
    infiles = [os.path.split(ss)[-1] for ss in glob(cv_dir+"/*.npz")]
    run_models = np.unique(np.array([ss[:ss.find('_cv')] for ss in infiles]))
    hamming_models = sorted([mod for mod in run_models if "hamming" in mod])
    ranking_models = sorted([mod for mod in run_models if "rankloss" in mod])
    
    # start writing
    writestring = "MODELS | "
    writestring += " | ".join(ss for ss in scores)
    writestring += " | "
    writestring += " | ".join(ss for ss in losses)
    #writestring += " | "
    #writestring += " | ".join(ss for ss in ("sparse", "sparse_s", "sparse_ss", "sparse_max"))
    
    if verbose:
        print writestring
    fh.write(writestring)
    #verbose = False
    for model in it.chain(hamming_models, ranking_models):
        runinfo["model"]  = clean_model(model)
        runinfo["scorer"] = clean_scorer(model)
        
        single_results = get_single(saveinfo, runinfo, False)#verbose)
        mod_scores, mod_losses, shuf_scores, shuf_losses, sparsity = single_results
        
        # print scores to file
        writestring ="\n%s | "%model
        writestring +=" | ".join("%.2f (%.2f) %s"%(mloss[0], mloss[1], ("***" if mloss[0] > sloss[2] else ("**" if mloss[0] > sloss[1] else ("*" if mloss[0] > sloss[0] else "")))) \
                                                  for (mloss, sloss) in it.izip(mod_scores.T, shuf_scores.T))
        writestring += " | "
        writestring +=" | ".join("%.2f (%.2f) %s"%(mloss[0], mloss[1], ("***" if mloss[0] < sloss[2] else ("**" if mloss[0] < sloss[1] else ("*" if mloss[0] < sloss[0] else "")))) \
                                                  for (mloss, sloss) in it.izip(mod_losses.T, shuf_losses.T))
        #writestring += " | "
        #writestring +=" | ".join("%.2f"%(ss,) for ss in sparsity)
        if verbose:
            print writestring
        fh.write(writestring)
    fh.close()

def sub_just_confusion(saveinfo, runinfo, verbose, ordering):
    model   = runinfo["model"]
    cv_dir  = runinfo["cv_dir"]
    scorer  = runinfo["scorer"]
    resultdir = saveinfo["resultdir"]
    fheader = os.path.join(cv_dir, model+"_"+scorer)
    resultfiles = [os.path.split(ss)[-1] for ss in glob(fheader+"*.npz")]
    Ncv = len(resultfiles)
    
    #:wqprint resultfiles
    SAVE_CONFUS = []
    for icv, fh in enumerate(resultfiles):
        readfile = os.path.join(cv_dir, fh)
        data = np.load(readfile)
        confus = data['econfusion']
        # normalize over rows
        msum = confus.sum(axis=1)[:, None]
        msum[msum==0.0] = 1.0
        confus /= msum
        SAVE_CONFUS.append(confus)
        
    sum_confus = np.array(SAVE_CONFUS).mean(axis=0)
    sum_confus = sum_confus[np.ix_(ordering, ordering)] # re-order
    
    confile = os.path.join(resultdir, model+"_"+scorer+"_confusion")
    plot_confusion(sum_confus, confile)
 
from plotting import load_concepts, plot_list
   
def just_confusion(saveinfo, runinfo, verbose):
    # setup
    resultdir = saveinfo["resultdir"]
    datadir = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI"))
    cv_dir = runinfo["cv_dir"]
    infiles = [os.path.split(ss)[-1] for ss in glob(cv_dir+"/*.npz")]
    run_models = np.unique(np.array([ss[:ss.find('_cv')] for ss in infiles]))
    hamming_models = sorted([mod for mod in run_models if "hamming" in mod])
    ranking_models = sorted([mod for mod in run_models if "rankloss" in mod])
    
    # get label ordeing
    zdatafile = os.path.join(datadir, 'concept.npz')
    Z = np.load(zdatafile)['Z_class']
    P = Z.sum(axis=0)/float(len(Z)) # popularity for each label
    ordering = P.argsort()[::-1] # max to min
    P = P[ordering]
    
    # print re-ordered list of concepts
    concepts  = load_concepts()
    with open("data/rconcepts", 'w') as outfile:
        for idx in ordering:
            outfile.write(concepts[idx])
            outfile.write("\n")
        
    # plot histogram of label ordering
    plot_list(P, "data/popularity", "Fraction")
    
    # get/plot confusion matrices
    for model in it.chain(hamming_models, ranking_models):
        runinfo["model"]  = clean_model(model)
        runinfo["scorer"] = clean_scorer(model) 
        sub_just_confusion(saveinfo, runinfo, verbose, ordering)
    
if __name__ == "__main__":
    
    if 0:
        get_single(*parser())
    elif 0:
        get_all(*parser())    
    else:
        just_confusion(*parser())
    
    #from plotting import plot_all_brains
    #plot_all_brains()
    
