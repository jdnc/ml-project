''' Creates forward and reverse mapping for brain data analysis
Saved files 
* features:
 - Transformed Images (full)
 - Transformed Images (reduced)
 - Connectivity
* Pickled ward class
* Pickled masker class
* targets:
 - y_data in binary format
 - z_data in binary format
 - raw string names
* metadata:
 - 5 fold CV by stratified cv
'''
import os
import numpy as np
from sklearn.feature_extraction import image
from sklearn.cluster import WardAgglomeration
import nibabel as nib
from nilearn import input_data
from sklearn.externals import joblib
from sklearn.cross_validation import StratifiedKFold, Bootstrap
from sklearn.feature_selection import SelectPercentile, f_classif

def feature_extractor(imgfile, maskfile, featurefile, maskerfile, wardfile, nclusters=[1000,], selectfile=None, targetfile=None, metafile=None, cachefile=None):
    
    resultdict = {"imgfile":imgfile, "maskfile":maskfile}
    # load data
    print "--loading data"
    nifti_masker = input_data.NiftiMasker(mask=maskfile, memory=cachefile, memory_level=1,
                              standardize=False)
    fmri_masked = nifti_masker.fit_transform(imgfile)
    print "--getting mask"
    mask = nifti_masker.mask_img_.get_data().astype(np.bool)
    
    # saveit
    joblib.dump(nifti_masker, maskerfile)
    resultdict["mask"]  = mask
    resultdict["Xmask"] = fmri_masked
    resultdict["maskerfile"] = maskerfile
    
    # get connectivity
    print "--getting connectivity"
    shape = mask.shape
    connectivity = image.grid_to_graph(n_x=shape[0], n_y=shape[1],
                                   n_z=shape[2], mask=mask)
    # saveit
    resultdict["connectivity"]    = connectivity
    print "--save main file"
    np.savez(featurefile+"_main.npz", **resultdict)
    
    # run  ward
    y     = np.load(targetfile)["ymap"]
    meta  = np.load(metafile)
    train = meta["train"]
    test  = meta["test"]
    ncv   = meta['ycv']
    
    # for each cv set
    for cvx in range(ncv):
        trainidx = train[cvx]
        testidx  = test[cvx]
        resultdict = {}        
        wardfiles = []
        selectfiles = []
        print "--Running ward %d"%(cvx, )
        for ix, nc in enumerate(nclusters):
            ward = WardAgglomeration(n_clusters=nc, connectivity=connectivity, memory=cachefile)
            ward.fit(fmri_masked[trainidx])
            fmri_reduced_train = ward.transform(fmri_masked[trainidx])
            fmri_reduced_test  = ward.transform(fmri_masked[testidx])
            
            # saveit
            subwardfile = wardfile+"_D%d_cv%d.pkl"%(nc, cvx,)
            joblib.dump(ward, subwardfile)
            resultdict["Xward_%d_train"%(nc,)] = fmri_reduced_train
            resultdict["Xward_%d_test"%(nc,)]  = fmri_reduced_test
            wardfiles.append(subwardfile)
            
            # additional feature selection
            selector = SelectPercentile(f_classif, percentile=30)
            selector.fit(fmri_reduced_train, y[trainidx])
            fmri_select_train = selector.transform(fmri_reduced_train)
            fmri_select_test  = selector.transform(fmri_reduced_test)
            
            # saveit
            subselectfile = selectfile+"_D%d_cv%d.pkl"%(nc, cvx,)
            joblib.dump(selector, subselectfile)
            resultdict["Xselect_%d_train"%(nc,)] = fmri_select_train
            resultdict["Xselect_%d_test"%(nc,)]  = fmri_select_test
            selectfiles.append(subselectfile)
            
        resultdict["wardfiles"]   = wardfiles
        resultdict["selectfiles"] = selectfiles
        
        # save results
        print "--save cv result"
        np.savez(featurefile+"_cv%d.npz"%(cvx, ), **resultdict)

def data_inversion(X, maskerfile=None, wardfile=None, selectfile=None, memoryfile=None, verbose=True):
    
    #print X.shape
    if selectfile is None: 
        fmri_select = X
    else: # apply inverse of selection
        if verbose:
            print "--inverting selection process"
        select = joblib.load(selectfile) 
        fmri_select = select.inverse_transform(X)
    #print fmri_select.shape 
        
    if wardfile is None:
        fmri_ward = fmri_select
    else: # apply inverse of masking
        if verbose:
            print "--inverting ward process"
        ward = joblib.load(wardfile) 
        fmri_ward = ward.inverse_transform(fmri_select)
    #print fmri_ward.shape
    
    if verbose:
        print "--inverting mask process"
    nifti_masker = joblib.load(maskerfile)
    # check if memory location exists:
    
    if not os.path.exists(nifti_masker.memory):
        nifti_masker.memory = memoryfile
        #nifti_masker.memory = None
    Xout = nifti_masker.inverse_transform(fmri_ward).get_data()
    Xout = np.ma.masked_equal(Xout, 0) # compressed data in 3-d format
    #print Xout.shape
    
    return Xout

def target_extractor(yfile, zfile, znames, targetfile):
    
    resultdict = {}
    # extract y
    print "--extracting y"
    y = np.loadtxt(yfile, dtype=int, usecols=[0])
    N = len(y)
    y_uniq, y2 = np.unique(y, return_inverse=True)
    C = len(y_uniq)
    yy = np.zeros(N*C, dtype=int)
    indx = np.ravel_multi_index( (np.arange(N), y2), (N,C) )
    yy[indx] = 1
    Y = yy.reshape((N, C))
    print "--y shape", N, C
    
    # saveit
    resultdict["yshape"] = (N, C)
    resultdict["yraw"] = y # initial raw classes
    resultdict["ymap"] = y2 # after call to uniq. Recover yraw = y[ymap]
    resultdict["Y"] = Y # binary mapped matrix form of ymap
    
    
    # extract z names
    print "--extracting concept names"
    concepts = []
    with open(znames, 'r') as infile:
        for line in infile:
            concepts.append(line.rstrip('\n').replace(" ", "_"))
    concepts = np.array(concepts, dtype='object')
    
    # extract z
    print "--extracting concept"
    z = np.array(np.loadtxt(zfile), dtype=int) # (C, K) matrix
    K = z.shape[1]
    Z = z[y2]
    
    print "--re-order by popularity"
    # sort Z by popularity
    P = Z.sum(axis=0)/float(len(Z)) # popularity for each label
    ordering = P.argsort()[::-1] # max to min
    
    # reorder
    concepts = concepts[ordering]
    z = z[:, ordering]
    Z = z[y2]
    
    # saveit
    resultdict["zshape"] = (N, K)
    resultdict["concepts"] = concepts
    resultdict["z"] = z # concept coding matrix per class 
    resultdict["Z"] = Z # concept coding matrix per example
    
    # save results
    print "--save final result"
    np.savez(targetfile, **resultdict)
    
def meta_extractor(targetfile, metafile, yfolds=5, zfolds=10):
    
    # setup CV sets
    resultdict = {}
    
    # load data
    print "--loading data"
    target = np.load(targetfile)
    y = target['ymap']
    Z = target['Z']
    
    # compute cv
    print "--computing cv sets"
    TRAIN, TEST = [], []
    skf = StratifiedKFold(y, yfolds, False)
    for train, test in skf:
        TRAIN.append(np.array(train, dtype=bool))
        TEST.append(np.array(test, dtype=bool))
    train = np.vstack(TRAIN)
    test  = np.vstack(TEST)
    
    resultdict['ycv'] = yfolds
    resultdict["train"] = train
    resultdict["test"]  = test
    
    # compute Z cv
    # subtrain[ix, jx] = indexes correponsing to trainset ix, subtrain set jx
    # subtest[ix, jx] = indexes correponsing to traintest ix, subtest set jx
    print "--computing Z cv sets"
    Nt = int(len(y)*((yfolds-1)/float(yfolds)) )
    Kt = int( Nt*((zfolds-1)/float(zfolds)) )
    Ke = Nt - Kt
    SUBTRAIN = np.zeros((yfolds, zfolds, Kt), dtype=bool)
    SUBTEST  = np.zeros((yfolds, zfolds, Ke), dtype=bool)
    for idx, trainset in enumerate(train): # each trainig set
        Zt = Z[trainset]
        Nt = len(Zt)
        skf = Bootstrap(Nt, 10000, Kt, Ke)
        for fold in range(zfolds):
            for ztrain, ztest in skf:
                ntrain = Zt[ztrain].sum(axis=0)
                ntest  = Zt[ztest].sum(axis=0)
                if np.alltrue(ntrain>0) and np.alltrue(ntest>0): #accept this
                    SUBTRAIN[idx, fold] = ztrain
                    SUBTEST[idx, fold]  = ztest
                    break
                    
    
    resultdict['zcv']      = zfolds
    resultdict["subtrain"] = SUBTRAIN
    resultdict["subtest"]  = SUBTEST
    
    # save results
    print "--save final result"
    np.savez(metafile, **resultdict)
    
def main(indirectory, outdirectory, modeldirectory):
    # name all used files
    try:
        os.makedirs(modeldirectory)
    except:
        pass
    
    cachefile = os.path.join(outdirectory, "cachefile")
    maskfile  = os.path.join(indirectory,  "goodvoxmask.nii.gz")
    imgfile   = os.path.join(indirectory,  "zstat_run1.nii.gz")
    yfile     = os.path.join(indirectory,  "data_key_run1.txt")
    zfile     = os.path.join(indirectory,  "cognitive_concepts", "cognitive_concepts.txt")
    znames    = os.path.join(indirectory,  "cognitive_concepts", "mental_concepts.txt")
    
    # call target extractor
    print "extracting targets..."
    targetfile = os.path.join(outdirectory, "targets_run1.npz")
    target_extractor(yfile, zfile, znames, targetfile)
    
    # call metadata extractor
    print "extracting metadata..."
    metafile = os.path.join(outdirectory, "meta_run1.npz")
    meta_extractor(targetfile, metafile)
    
    # call data extractor
    print "extracting features..."
    featurefile = os.path.join(outdirectory,  "features_run1")
    wardfile = os.path.join(modeldirectory,   "ward_run1")
    selectfile = os.path.join(modeldirectory, "select_run1")
    maskerfile = os.path.join(modeldirectory, "masker_run1.pkl")
    nclusters = [50000, 10000, 1000]
    
    feature_extractor(imgfile, maskfile, featurefile, maskerfile, wardfile, nclusters, selectfile, targetfile, metafile, cachefile)
    
    print "Done!"
    
    

if __name__ == "__main__": 
    indirectory = os.path.expanduser(os.path.join("~", "workdir", "data", "may_openfMRI"))
    outdirectory = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "preprocessed"))
    modeldirectory = os.path.expanduser(os.path.join("~", "workdir", "data", "openfMRI", "preprocessed", "models"))
    
    #main(indirectory, outdirectory, modeldirectory)
    
    featurefile = os.path.join(outdirectory,  "features_run1")
    wardfile = os.path.join(modeldirectory,   "ward_run1")
    selectfile = os.path.join(modeldirectory, "select_run1")
    maskerfile = os.path.join(modeldirectory, "masker_run1.pkl")
    cachefile = os.path.join(outdirectory, "cachefile")
    
    # temporary sanity check
    #print "checking feats"
    X = np.load(featurefile+"_cv0.npz")["Xselect_1000_train"]
    selectfilex = selectfile+"_D1000_cv0.pkl"
    wardfilex   = wardfile+"_D1000_cv0.pkl"
    print data_inversion(X, maskerfile, wardfilex, selectfilex, cachefile).shape
    
