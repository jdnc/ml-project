
import numpy as np
from collections import defaultdict

from sklearn.multiclass import OneVsRestClassifier#, _fit_binary
from sklearn.base import BaseEstimator, clone#, ClassifierMixin, is_classifier
#from sklearn.multiclass import MetaEstimatorMixin
from sklearn.cross_validation import Bootstrap

from sklearn.linear_model import Ridge, Lasso, SGDClassifier, SGDRegressor
from sklearn.externals.joblib import Parallel, delayed
        
###############################################################################
# enet helper functions
###############################################################################
def enet_alpha_grid(X, y, Xy=None, l1_ratio=1.0, fit_intercept=True,
                eps=1e-3, n_alphas=100, normalize=False, copy_X=True):
    """ 
    function modified from scikit-learn 
    /scikit-learn/sklearn/linear_model/coordinate_descent.py
    Compute the grid of alpha values for elastic net parameter search

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Training data. Pass directly as Fortran-contiguous data to avoid
        unnecessary memory duplication

    y : ndarray, shape = (n_samples,)
        Target values

    Xy : array-like, optional
        Xy = np.dot(X.T, y) that can be precomputed.

    l1_ratio : float
        The ElasticNet mixing parameter, with ``0 <= l1_ratio <= 1``.
        For ``l1_ratio = 0`` the penalty is an L2 penalty. ``For
        l1_ratio = 1`` it is an L1 penalty.  For ``0 < l1_ratio <
        1``, the penalty is a combination of L1 and L2.

    eps : float, optional
        Length of the path. ``eps=1e-3`` means that
        ``alpha_min / alpha_max = 1e-3``

    n_alphas : int, optional
        Number of alphas along the regularization path

    fit_intercept : bool
        Fit or not an intercept

    normalize : boolean, optional, default False
        If ``True``, the regressors X will be normalized before regression.

    copy_X : boolean, optional, default True
        If ``True``, X will be copied; else, it may be overwritten.
    """
    if Xy is None:
        Xy = np.dot(X.T, y)
        n_samples = X.shape[0]
    else:
        n_samples = len(y)

    alpha_max = np.abs(Xy).max() / (n_samples * l1_ratio)
    alphas = np.logspace(np.log10(alpha_max * eps), np.log10(alpha_max),
                         num=n_alphas)[::-1]
    return alphas

########################################################################
# Helper for warm start models
########################################################################
def _subfit(estimator, X, y, classes=None):
    """Simpole fit (no checks!) for a single binary estimator."""
    estimator.fit(X, y)
    return estimator

class MultilabelwState(OneVsRestClassifier):
    ''' add on a re_fit method to OneVsrest '''
    
    def _first_fit(self, X, y):
        """
        Wrapper methops for fit, main purpose is to make sure data does not change for refit call''
        """
        self.fit(X, y)
        self.saveX = X
        self.saveY = self.label_binarizer_.transform(y)
        return self
        
    def refit(self, X, y, **kwargs):
        """
        Fit method that keeps the current state. Used for warm start models!
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        y : array-like, shape = [n_samples]
         or sequence of sequences, len = n_samples
            Multi-class targets. A sequence of sequences turns on multilabel
            classification.

        Returns
        -------
        self
        """
        # Get error: cannot pickle objects if we use more njobs, so set self.njobs = 1
        self.n_jobs = 1
        
        # check if fitted
        if not hasattr(self, "estimators_"): # not yet fittted
            self.estimator.set_params(**kwargs)
            self._first_fit(X, y)
        else:
            # recover data
            X = self.saveX
            Y = self.saveY
            # fix parameters
            E = [est.set_params(**kwargs) for est in self.estimators_]
            
            # train
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
            delayed(_subfit)(E[i], X, Y[:, i], classes=["not %s" % i, i]) 
            for i in range(Y.shape[1]))

        return self
    
########################################################################
# Helper for parallelization (older sklearn)
########################################################################
from sklearn.preprocessing import LabelBinarizer
import warnings
warnings.simplefilter("ignore")
# TODO: AVOID IMPORT of private function
from sklearn.multiclass import _fit_binary

def fit_ovr(estimator, X, y, n_jobs=1):
    """Fit a one-vs-the-rest strategy."""
    
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    
    estimators = Parallel(n_jobs=n_jobs)(
        delayed(_fit_binary)(estimator, X, Y[:, i], classes=["not %s" % i, i])
        for i in range(Y.shape[1]))
    return estimators, lb

class OneVsRestClassifierp(OneVsRestClassifier):
    """for parallelization (older sklearn)
    """
    def fit(self, X, y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.

        y : array-like, shape = [n_samples]
         or sequence of sequences, len = n_samples
            Multi-class targets. A sequence of sequences turns on multilabel
            classification.

        Returns
        -------
        self
        """
        self.n_jobs = 1
        self.estimators_, self.label_binarizer_ = fit_ovr(self.estimator, X, y,
                                                          n_jobs=self.n_jobs)
        return self



########################################################################
# converter for list-of-list from to binary matrix format
# TODO: replace one Vs Rest llabel_binarizer with pre-set matrices
########################################################################
def converter(Z):
    ''' converts Z from binary matrix format into Z in tuple of tuple'''
    N, K = Z.shape
    indexer = np.arange(K)
    Zc = []
    for row in Z:
        Zc.append(indexer[row>0].tolist())
    return tuple(Zc)

def inverter(Zc, K=None, dtype=int):
    ''' converts Z from tuple of tuple into binary matrix'''
    N = len(Zc)
    if K is None:# estimate K assuming 0, ... ,K-1 labelling
        K = max(max(s) for s in Zc) +1
    
    Z = np.zeros((N, K), dtype=dtype)
    for ix, row in enumerate(Zc):
        if len(row)==0: continue
        Z[ix][np.array(row)] = 1
    return Z

def sub_rankloss(t, s):
    ''' compute rankloss
     - AND - 
    computes modified rankloss handling ties: 1 - AUC '''
    # setup
    sp = s[t>0] # scores for all positive labelled examples
    sn = s[np.logical_not(t>0)] # scores for all negative labelled examples
    Np = float(len(sp) * len(sn)) # N_pos x N_neg
    Np = 1.0 if Np < 1.0 else Np
    
    # computation
    rscore  = float(sum((xsp<=sn).sum() for xsp in sp))
    rscoreb = float(sum((xsp< sn).sum() for xsp in sp))
    rscoreb += 0.5*sum((xsp==sn).sum() for xsp in sp)
    return rscore/Np, rscoreb/Np

########################################################################
# Compute large class of multilabel metrics
########################################################################
def score_results(Zt, Zp, Sp):
    ''' Score results of classification 
    Zt = training labels (N, L)
    Zp = predicted labels (N, L)
    Sp = predicted scores (for ranking metrics) (N, L)
    
    Note: Loss (Lower is better) -OR - Score (Higher is better)
    metrics:
    {
    # set accuracy (full set of clases)
    hamming : fraction of label differences (either direction): LOSS
    acc: Accuracy: 1 - fraction of set errors vs true : SCORE
    
    # label accuracy (computed at label level for each example)
    label_acc: Label: accuracy: 1 - fraction of set errors vs true : SCORE
    prec: precision :  fraction of predicted true labels that are true : SCORE
    recall: recall: fraction of true labels that are predicted as true : SCORE
    f1score: F1 score : weighted average of precision and recall: SCORE
    
    one_err: one error, count if top predicted label is not true label : LOSS
    coverage: how far in ranking to get all true labels : LOSS
    rkloss: rank loss : LOSS
    rklossb: corrected rank loss = 1 - AUC : LOSS
    
    '''
    N, L = [float(a) for a in Zt.shape]
    # TODO: assert here - assumes N> 0, L>0
    score = defaultdict(float)
    #score = {"hamming":0.0, "acc":0.0, "prec":0.0, "recall":0.0}
    for n, t in enumerate(Zt): # for each label
        
        if t.sum()==0: # no true labels
            N -=1
            continue # skip this run
        
        p = Zp[n]
        s = Sp[n]
        
        tmask = t>0
        pmask = p>0
        Nt = float(tmask.sum())
        Np = float(pmask.sum())
        Ni = float(np.logical_and( tmask, pmask ).sum())
        Nu =  float(np.logical_or( tmask, pmask ).sum())
        Nh = float(np.logical_xor( tmask, pmask ).sum())
        
        # set accuracy metrics
        hamming = Nh / L
        acc = 0.0 if (t!=p).sum()>0.0 else 1.0
        
        # label accuracy matrics
        lab_acc = Ni / Nu if Nu> 0.0 else 0.0
        prec = Ni / Np if Np> 0.0 else 0.0
        recall = Ni / Nt if Nt> 0.0 else 0.0
        f1score = (2.0*prec * recall)/ (prec + recall) if (prec + recall) > 0 else 0.0
        
        # ranking metrics
        one_err = 1-t[s.argmax()]
        # NOTE: Depends on argmax selecting the smallest index in a set of equal values
        # i.e. depends on argmax([0, 1, 3, 4, 4]) = 3, not 4. 
        # Specified as current expected behavior according to Numpuy docs
        coverage = t[np.argsort(s)[::-1]].cumsum().argmax()/L
        rkloss, rklossb = sub_rankloss(t, s)
        
        score['hamming'] += hamming
        score['acc'] += acc
        
        score['lab_acc'] += lab_acc
        score['prec'] += prec
        score['recall'] += recall
        score['f1score'] += f1score
        
        score['one_err'] += one_err
        score['coverage'] += coverage
        score['rkloss'] += rkloss
        score['rklossb'] += rklossb
        
        
    score = dict(score)
    # avg over N
    for k in score.iterkeys():
        score[k] /= N
    
    return score


########################################################################
# methods for double multilabel
########################################################################

########################################################################
# Compute large class of metrics per label
########################################################################
def label_scores(Zt, Zp, Sp):
    ''' Score results of classification for each label independently
    Zt = training labels (N, L)
    Zp = predicted labels (N, L)
    Sp = predicted scores (for ranking metrics) (N, L)
    
    Returns a score for each label.
    
    Note: Loss (Lower is better) -OR - Score (Higher is better)
    metrics:
    {
    # set accuracy (full set of clases)
    hamming : fraction of label differences (either direction): LOSS
    acc: Accuracy: 1 - fraction of set errors vs true : SCORE
    
    # label accuracy (computed at label level for each example)
    label_acc: Label: accuracy: 1 - fraction of set errors vs true : SCORE
    prec: precision :  fraction of predicted true labels that are true : SCORE
    recall: recall: fraction of true labels that are predicted as true : SCORE
    f1score: F1 score : weighted average of precision and recall: SCORE
    
    one_err: one error, count if top predicted label is not true label : LOSS
    coverage: how far in ranking to get all true labels : LOSS
    rkloss: rank loss : LOSS
    rklossb: corrected rank loss = 1 - AUC : LOSS
    
    '''
    N, L = Zt.shape
    # TODO: assert here - assumes N> 0, L>0
    score = defaultdict(lambda: np.zeros(L))
    #score = {"hamming_sub":[list], "acc_sub":[list], "prec_sub":[list], "recall_sub":[list]}
    for l in range(L): # for each label
        
        t = Zt[:, l]
        #if t.sum()==0: # no true labels
        #    continue # skip this run
        
        p = Zp[:, l]
        s = Sp[:, l]
        
        tmask = t>0
        pmask = p>0
        Nt = float(tmask.sum())
        Np = float(pmask.sum())
        Ni = float(np.logical_and( tmask, pmask ).sum())
        Nu =  float(np.logical_or( tmask, pmask ).sum())
        Nh = float(np.logical_xor( tmask, pmask ).sum())
        
        # set accuracy metrics
        hamming = Nh / float(N)
        
        # label accuracy metrics
        lab_acc = Ni / Nu if Nu> 0.0 else 0.0
        prec = Ni / Np if Np> 0.0 else 0.0
        recall = Ni / Nt if Nt> 0.0 else 0.0
        f1score = (2.0*prec * recall)/ (prec + recall) if (prec + recall) > 0 else 0.0
        
        # ranking metrics
        # NOTE: Depends on argmax selecting the smallest index in a set of equal values
        # i.e. depends on argmax([0, 1, 3, 4, 4]) = 3, not 4. 
        # Specified as current expected behavior according to Numpuy docs
        coverage = t[np.argsort(s)[::-1]].cumsum().argmax()/float(N)
        rkloss, rklossb = sub_rankloss(t, s)
        
        score['hamming_sub'][l] = hamming
        
        score['lab_acc_sub'][l] = lab_acc
        score['prec_sub'][l] = prec
        score['recall_sub'][l] = recall
        score['f1score_sub'][l] = f1score
        
        score['coverage_sub'][l] = coverage
        score['rkloss_sub'][l] = rkloss
        score['rklossb_sub'][l] = rklossb
    
    score = dict(score)
    return score

########################################################################
# OTHER UTILITY
########################################################################
''' convert 0/1 into -1, +1 '''
bool_to_binary = lambda Z: (Z*2)-1.0

########################################################################
# Helper for soft classifiers
########################################################################
class softOneVsRest(OneVsRestClassifier):
    ''' add on a re_fit method to OneVsrest '''
    def fit(self, X, Z, **kwargs):
        #super(OneVsRestClassifier, self).fit(X, converter(Z), **kwargs)
        OneVsRestClassifier.fit(self, X, converter(Z), **kwargs)
        return self
    def predict(self, X, **kwargs):
        W = self.coef_.T # mod.coef_() is K * D weight vector for each class 
        B = self.intercept_.T
        return np.dot(X, W) + B
    def set_params(self, **kwargs):
        self.estimator.set_params(**kwargs)
        return self

########################################################################
# scoring
########################################################################
def sub_hamming(t, s):
    L = len(t)
    tmask = t>0
    pmask = s>0
    Nh = float(np.logical_xor( tmask, pmask ).sum())
    return Nh / float(L)
        
        
def get_scorer(losstype):
    ''' both hamming and rankloss are loss functions. Convert to score functions usingh 1 - s'''
    if losstype=='hamming':
        sub_loss = sub_hamming
    elif losstype=="rankloss":
        sub_loss = lambda a, b: sub_rankloss(a, b)[1]
    else:
        pass
    
    def scorer(Zt, Se, sub_loss=sub_loss):
        N = len(Zt)
        loss = sum(sub_loss(Zt[n], Se[n]) for n in range(N))
        return 1.0 - (loss / float(N))
    
    return scorer

########################################################################
# confusion matrix
########################################################################
def multilabel_confusion(Zt, Ze, normalize=False):
    L = Zt.shape[1]
    M = np.zeros((L, L))
    
    for ix in range(L):
        t = Zt[:, ix]>0
        for jx in range(L):
            p = Ze[:, jx]>0
            M[ix, jx] = np.logical_and(t, p).sum()
    
    if normalize:
        msum = M.sum(axis=1)[:, None]
        msum[msum==0.0] = 1.0
        M /= msum
    
    return M

########################################################################
# scoring wrapper
########################################################################
all_score_results = lambda Zt, Sp: (score_results(Zt, Sp>0, Sp), multilabel_confusion(Zt, Sp), label_scores(Zt, Sp>0, Sp))

########################################################################
# sq loss wrapper
########################################################################
class SquaredRegression(Lasso, Ridge):
    def __init__(self, penalty='l1', dual=None, C=None, alpha=None):
        
        self.l1 = True if penalty=="l1" else False
        if self.l1:
            Lasso.__init__(self, alpha=alpha)
        else:
            Ridge.__init__(self, alpha=alpha)
            
    def set_params(self, C=None):
        if self.l1:
            Lasso.set_params(self, alpha=C)
        else:
            Ridge.set_params(self, alpha=1.0/(2.0*C))
        return self

    def fit(self, *args, **kwargs):
        if self.l1:
            Lasso.fit(self, *args, **kwargs)
        else:
            Ridge.fit(self, *args, **kwargs)
        return self

########################################################################
# weighted multilabel
########################################################################
def multilabel_weights(Z):
    L = Z.shape[1]
    n = (Z>0.0).sum(axis=1)# count +1 in each row
    w = (n)*(L-n)
    # For all pos, and all-neg, use closest adjacent weight @ L*(L-1)
    # corresponds to a low weight
    w[w==0.0] = L*(L-1.0)
    return 1.0/w

class WeightedClassifier(SGDClassifier):
    def fit(self, XX, y):
        sample_weight = XX[:, -1] # may need to copy
        X = XX[:, :-1] # may need to copy?
        super(WeightedClassifier, self).fit(X, y, sample_weight=sample_weight)
        return self
        
    def set_params(self, C=None):
        super(WeightedClassifier, self).set_params(alpha=C)
        return self

class WeightedRegressor(Ridge):
    def fit(self, XX, y):
        sample_weight = XX[:, -1] # may need to copy
        X = XX[:, :-1] # may need to copy?
        super(WeightedRegressor, self).fit(X, y, sample_weight=sample_weight)
        return self
        
    def set_params(self, C=None):
        super(WeightedRegressor, self).set_params(alpha=C)
        return self
    
class DummyEstimator(BaseEstimator):
    def __init__(self, pred=1.0, p=None):
        ''' p is a placeholder parameter, not used'''
        self.pred = pred
        
    def fit(self, X, y):
        self.coef_ = None
        self.intercept_ = np.array([self.pred])
        return self
    
class EmpiricalNull(BaseEstimator):
    def __init__(self, p=None):
        ''' p is a placeholder parameter, not used'''
        pass
    
    def fit(self, X, y):
        self.coef_ = None
        self.intercept_ = np.array([ (y>0).sum() / float(len(y)) ])
        return self

class _ConstantEstimator(BaseEstimator):
    def __init__(self, p=None):
        ''' p is a placeholder parameter, not used'''
        pass
    def fit(self, D, y):
        self.coef_ = np.zeros(D)
        self.intercept_ = np.array([y[0]*1E-6]) # close to zero, but keeping sign information
        return self
        
########################################################################
# Parallel fit wrapper
########################################################################
#_subfit = lambda estimator, X, Y: clone(estimator).fit(X, Y)
def _subfit_clone(estimator, X, y):
    """Fit a single estimator."""
    unique_y = np.unique(y)
    if isinstance(estimator, DummyEstimator) or isinstance(estimator, EmpiricalNull):
        estimator = clone(estimator)
        estimator.fit(X, y)
    elif len(unique_y) == 1:
        if isinstance(estimator, WeightedClassifier) or isinstance(estimator, WeightedRegressor):
            D = X.shape[1]-1
        else:
            D = X.shape[1]
        estimator = _ConstantEstimator().fit(D, unique_y)
    else:
        estimator = clone(estimator)
        estimator.fit(X, y)
    return estimator

class parafit(BaseEstimator):#, ClassifierMixin, MetaEstimatorMixin):
    """implement parallel fit for multilabel classifier
    clf should have coef_ and intercept_ objects
    """
    def __init__(self, estimator, scorer, n_jobs=1, pre_dispatch="n_jobs"):
        self.estimator = estimator
        self.scorer = scorer
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
    
    def fit(self, X, Y):
        """Fit underlying estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Data.
            
        y : array-like, shape = [n_samples]
         or sequence of sequences, len = n_samples
            Multi-class targets. A sequence of sequences turns on multilabel
            classification.

        Returns
        -------
        self
        """
        estimator = self.estimator
        self.all_estimators = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)(
            delayed(_subfit_clone)(estimator, X, Y[:, i]) for i in range(Y.shape[1]))
        
        if isinstance(estimator, DummyEstimator) or isinstance(estimator, EmpiricalNull):
            # special case with no fit
            self.dummy_flag = True
            self.W = 0.0
            self.B = 0.0
            self.probs = self.intercept_.ravel()
        else:
            self.dummy_flag = False # dummy model
            self.W = self.coef_.T
            self.B = self.intercept_.T
            self.D = self.W.shape[0]
        return self
    
    def predict(self, X):
        N = X.shape[0]
        if self.dummy_flag:
            Y = np.hstack(( np.random.binomial(1, p, (N,1)) for p in self.probs ))
            Y[Y<1] = -1
            Y = np.array(Y, dtype=float)
        else:
            Y = np.dot(X[:, :self.D], self.W) + self.B
            
        return Y
    
    def score(self, X, Y):
        return self.scorer(Y, self.predict(X))
    
    def set_params(self, **kwargs):
        self.estimator.set_params(**kwargs)
        return self
    
    @property
    def coef_(self):
        return np.array([e.coef_.ravel() for e in self.all_estimators])
    
    @property
    def intercept_(self):
        return np.array([e.intercept_.ravel() for e in self.all_estimators])
    
########################################################################
# AUTOMATED PERCENTILES
########################################################################
class Percentiles(object):
    ''' computes running two sided percentiles i.e. return (l, h) s.t.
    a) prom(x > l) = 1-p
    b) prob(x < h) = p
    '''
    def __init__(self, N, D, K=1, p=.05):
        self.P = int(round(p*N))+2
        # get data ready
        self.X_LOW = np.zeros((D, K, self.P))
        self.X_HIG = np.zeros((D, K, self.P))
        self.c = 0 # count
        
    def add_data(self, x):
        if self.c < self.P:
            self.X_LOW[:, :, self.c] = x
            self.X_HIG[:, :, self.c] = x
        else:
            self.X_LOW[:, :, -1] = x
            self.X_LOW.sort()
            self.X_HIG[:, :, 0] = x
            self.X_HIG.sort()
            
        self.c+=1
            
    def results(self):
        return np.dstack((self.X_LOW[:, :, -2], self.X_HIG[:, :, 1]))
    
class Maxentiles(object):
    ''' computes rmax and min i.e. return (l, h) s.t.
    a) prom(x > l) = 0.0
    b) prob(x < h) = 1.0
    '''
    def __init__(self, N, D, K=1):
        # get data ready
        self.X_LOW = np.zeros((D, K))
        self.X_HIG = np.zeros((D, K))
        
    def add_data(self, x):
        self.X_LOW = np.minimum(self.X_LOW, x)
        self.X_HIG = np.maximum(self.X_HIG, x)
            
    def results(self):
        return np.dstack((self.X_LOW, self.X_HIG))

class Yieldcv(Bootstrap):
    # convenient wrapper for turning list sof indices into cv's
    def __init__(self, trainlist, testlist, k):
        train_size = len(trainlist[0])
        test_size = len(testlist[0])
        n = train_size+test_size
        n_iter = k
        super(Yieldcv, self).__init__(n, n_iter, train_size, test_size)
        self.trainlist = trainlist
        self.testlist = testlist
        
    def __iter__(self):
        for ix in range(self.n_iter):
            yield (self.trainlist[ix], self.testlist[ix])
            
