'''
Created on Aug 20, 2013
Multilabel training model
Coarse paralellization for CV
in-memory parallelization for parameter selection per CV
Algorithm:
1) Separate CV
2) Launch parameter selection in parallel (using grid_searchCV)
3) test re-learned parameter

TODO:
===================
train for either hamming OR rankloss (run separately)
score for all metrics

======================================
model: [L1, L2] x [hinge, log, sqloss] 
parameters:
CV: [0...4]

@author: Sanmi Koyejo; sanmi.k@gmail.com
'''

# utility
import numpy as np
from optparse import OptionParser
import sys, os
import time
# sub-models
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from utils import SquaredRegression, DummyEstimator, EmpiricalNull
from utils import WeightedClassifier, WeightedRegressor, multilabel_weights
# wrapper and scoring
from utils import all_score_results, get_scorer
from utils import parafit
from sklearn.grid_search import GridSearchCV
# helper
from sklearn.cross_validation import KFold
from utils import enet_alpha_grid
from utils import bool_to_binary
from utils import Yieldcv
from info import known_models, reg_models, base_models, clean_name

def save_multilabel(results, runinfo, saveinfo, verbose):
    ''' save results of this run
    results = (p, q, W, tscore, escore)
    saveinfo = {'resultdir':resultdir, 'cv':cv}
    runinfo = {'model':model}
    ===========================================
    resultdir: save directory
    model: name of model from known_models
    cv: which cv fold
    p: parameter being tested
    q: secondary being tested
    W: resulting model matrices
    tscore: training score
    escore: test score
    
    see utils/score_results.py for info on scores
    '''
    # expansion
    model = runinfo["model"]
    shuffle = runinfo["shuffle"]
    (W, B, tscore, escore, tconfusion, econfusion, tsub_score, esub_score) = results
    resultdir, cv = [saveinfo[s] for s in ("resultdir", "cv")]
    scorer = runinfo["scorer"]
    
    # save the result
    outdata = (model, cv)+tuple([escore[s] for s in "hamming", "rklossb", "f1score"])
    if verbose:
        print "saving model=%s, cv=%d, hamming=%.3f, rankloss=%.3f, f1score=%.3f"%outdata
    if shuffle:
        savefile = os.path.join( resultdir, "%s_%s_id%d"%(model, scorer, np.random.randint(1E6) ) )
    else:
        savefile = os.path.join( resultdir, "%s_%s_cv%d"%(model, scorer, cv) )
    
    
    np.savez(savefile, W=W, B=B, tscore=tscore, escore=escore, \
             tconfusion=tconfusion, econfusion=econfusion, tsub_score=tsub_score, esub_score=esub_score,\
             param_list=saveinfo["param_list"], shuffle=shuffle, scorer=scorer)

def train_multilabel_base(clf, param_list, score_func, xdata, verbose):
    #train_multilabel_base(clf, p, data, verbose):
    '''base function for model fiiting.
    1) fits classifier
    2) Score training data
    3) Score testing data
    '''
    Xt, Xe, Zt, Ze, subtrain, subtest, n_jobs = xdata
    ######################################
    # training
    ######################################
    estimator = parafit(clf, score_func, n_jobs=n_jobs, pre_dispatch='n_jobs') # parallelize on class level
    #cv = getcv(KFold(len(Xt), 3), 3)
    cv = Yieldcv(subtrain, subtest, 5)
    grid_clf  = GridSearchCV(estimator=estimator, param_grid=param_list, cv=cv, n_jobs=1, pre_dispatch='n_jobs') # iterative CV
    grid_clf.fit(Xt, Zt)
    best_model = grid_clf.best_estimator_
    
    # training score
    Spt = best_model.predict(Xt)
    tscore, tconfusion, tsub_score = all_score_results(Zt, Spt)
    
    # testing score
    #Zpe = inverter(grid_clf.predict(Xe), L)
    Spe = best_model.predict(Xe)
    escore, econfusion, esub_score = all_score_results(Ze, Spe)
    
    return best_model.W, best_model.B, tscore, escore, tconfusion, econfusion, tsub_score, esub_score

def train_multilabel(idata, runinfo, saveinfo, verbose):
    ''' wrapper for training/testing sklearn multilabel classifier
    model = choice of model from known_models
    p = log_10(regularization parameter)
    X = feature matrix (N, D)
    Z = multilabel matrix (N, K)
    train = binary training index-indicator (N,)
    test = binary training index-indicator (N,)
    
    # get two models - > rankloss and hamming loss
    '''
    
    # expand sub
    model = runinfo['model'].lower()
    score_func = get_scorer(runinfo['scorer'])
    p = runinfo["p"]
    shuffle = runinfo["shuffle"]
    n_jobs  = runinfo["n_jobs"]
    
    # train/test split
    (Xt, Xe, Z, train, test, subtrain, subtest) = idata
    if shuffle:
        N = Z.shape[0]
        index = np.random.permutation(N)
        Z = Z[index]
    
    Zt = Z[train]
    Ze = Z[test]
    
    # some setup
    if model in set(base_models):
        param_list = {"p":[None]}
        if model.endswith('majority'):
            if model.startswith('prob'):
                clf = EmpiricalNull()
            else:
                pred = 0.0 if model.startswith('zero') else 1.0
                clf = DummyEstimator(pred)
        else:
            pass
    else: # all other
        plist = enet_alpha_grid(Xt, Zt, n_alphas=p, l1_ratio=1e-2, eps=1e-4) # extra params to scale regularization up by 100 => 
        reg_type = "l1" if 'l1' in model else "l2" # defaut if you do not use l1
        param_list = {"C":plist}
        if "weight" in model: #
            # stack weights on xt so sub-cv can get access
            weights = multilabel_weights(Zt)[:, None]#np.array(, ndmin=2).T
            Xt = np.hstack((Xt, weights ))
            # pick model
            if 'hinge' in model: # SVM's
                clf = WeightedClassifier(loss="hinge", penalty=reg_type, n_iter=10)
            elif "squared" in model: # ridge regression model
                clf = WeightedRegressor() # no L! for now... oh well...
            else: #model.endswith('log'): #assumes logistic
                clf = WeightedClassifier(loss="log", penalty=reg_type, n_iter=10)
        else:
            dual_type = True if reg_type=="l2" else False
            # pick model
            if 'hinge' in model: # SVM's
                estimator = LinearSVC
            elif "squared" in model: # ridge regression model
                estimator = SquaredRegression
            else: #if model.endswith('log'): # assumes logistic
                estimator = LogisticRegression
            clf = estimator(penalty=reg_type, dual=dual_type)
            
    # train
    xdata = Xt, Xe, Zt, Ze, subtrain, subtest, n_jobs
    if verbose:
        print clf
    results = train_multilabel_base(clf, param_list, score_func, xdata, verbose)
    # save results
    saveinfo["param_list"] = param_list
    save_multilabel(results, runinfo, saveinfo, verbose)
    
def extract_data(datadir, cv, verbose, istoy):
    # set directories
    xdatafile = os.path.join(datadir, 'features_run1_cv%d.npz'%(cv,))
    zdatafile = os.path.join(datadir, 'targets_run1.npz')
    tdatafile = os.path.join(datadir, 'meta_run1.npz')
    
    # extract feature data
    data = np.load(xdatafile)
    if istoy:
        Xt = data['Xselect_1000_train']
        Xe = data['Xselect_1000_test']
    else:
        Xt = data['Xselect_50000_train']
        Xe = data['Xselect_50000_test']
        
    # extract target data
    Z = bool_to_binary( np.load(zdatafile)['Z'])
    
    data   = np.load(tdatafile)
    train  = np.array(data["train"][cv], dtype=bool)
    test   = np.array(data["test"][cv], dtype=bool)
    Zt = Z[train]
    Ze = Z[test]
    subtrain = data["subtrain"][cv]
    subtest  = data["subtest"][cv]
    
    return Xt, Xe, Z, train, test, subtrain, subtest
        
def main():
    usage = "usage: %prog [options] arg"
    parser = OptionParser(usage)
    parser.add_option("-m", "--model", dest="model", default='l2 svm', type='string',
                      #choices=known_models, 
                      help="model selection from {%s}"%', '.join(s for s in known_models))
    parser.add_option("-i", "--indir", dest="indir",
                  help="input directory", metavar="INDIR")
    parser.add_option("-o", "--outdir", dest="outdir",
                  help="output directory", metavar="OUTDIR")
    parser.add_option("-c", "--cv", dest="CV", type="int", default=0,
                  help="which CV set (typically 0...4)")
    parser.add_option("-l", "--lscore", dest="scorer", default='hamming', type='choice',
                      choices=['hamming', 'rankloss'], help="score type, hamming or rankloss (def=rankloss)")
    parser.add_option("-p", dest="param", type=float, default=10,
                  help="number of model hyperparameter to test (def=10)")
    parser.add_option("-n", "--n_jobs", dest="n_jobs", type=int, default=6,
                  help="number of processors used in parallel fit (def=6)")
    parser.add_option("-s", "--shuffle", dest="shuffle", default=False,
                      action="store_true", help="shuffle labels (for random baseline)")
    parser.add_option("-t", "--toy", default="1", type="choice",
                      choices=["0", "1"], dest="istoy", 
                      help='generate toy data instead of real data (-t 1) or no (-t 0)' )
    parser.add_option("-q", "--quiet", default=False,
                      action="store_true", dest="quiet")
    
    (options, args) = parser.parse_args()
    
    # extract options
    model_all   = options.model # model type
    cv      = options.CV    # which cv set
    p       = options.param # number of params
    shuffle = options.shuffle # shuffle yes/no
    verbose = 0 if options.quiet else 1
    scorer  = options.scorer
    istoy   = int(options.istoy) # yes/no running smaller test data
    n_jobs  = options.n_jobs # number f parallel fit jobs
    
    datadir = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "preprocessed"))
    
    if istoy:
        resultdir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "toy", "results", "wardcv"))
    else:
        resultdir = os.path.expanduser(os.path.join("~", "workdir", "multilabel", "results", "wardcv"))
        
    if shuffle:
        resultdir = os.path.join(resultdir, "shuffle")
        
    if options.indir is not None:
        datadir = options.indir
    if options.outdir is not None:
        resultdir = options.outdir
    
    # make result directory
    try:
        os.makedirs(resultdir)
    except:
        pass
    
    # extract model name
    model = clean_name(model_all)
    
    # run it!
    ###################################################
    # extract_data
    if verbose:
        print "Extracting data..." 
        T = time.time()
    
    Xt, Xe, Z, train, test, subtrain, subtest = extract_data(datadir, cv, verbose, istoy)
    
    if verbose:
        dims = Xt.shape+(Z.shape[1],)
        print "data dims: N=%d, D=%d, K=%d"%dims
    
    if verbose: 
        print "extract time = %s"%(time.time()-T,)
        if shuffle: print "shuffling output labels..."
    
    if verbose: 
        print "Training model...\nmodel=%s, cv=%d, scorer=%s"%(model, cv, scorer)
        T = time.time()
    
    # run model (or multimodel). 
    saveinfo = {'resultdir':resultdir, 'cv':cv}
    runinfo = {'model':model, "scorer":scorer, "p":p, "shuffle":shuffle, "n_jobs":n_jobs}
    data = (Xt, Xe, Z, train, test, subtrain, subtest)
    train_multilabel(data, runinfo, saveinfo, verbose)
    if verbose: print "Combined Train time = %s"%(time.time()-T,)
         
if __name__ == "__main__":
    main()    
