# -*- coding: utf-8 -*-

import numpy as np
import scipy.spatial.distance as spd
from ..simimeasures import cy as cy
from ..simimeasures import py as py


class RFSE(object):

    def __init__(self, sim_func, itrs, sigma, feat_size, bagging=0.0):

        if sim_func == 'cosine_sim':
            self.sim_func = cy.cosine_sim
            self.sim_min_value = -1.0
        elif sim_func == 'eucl_sim':
            self.sim_func = cy.eucl_sim
            self.sim_min_value = 0.0
        # elif sim_func == 'minmax_sim':
        #     self.sim_func = cy.minmax_sim
        #     self.sim_min_value = 0.0
        elif sim_func == 'py_cosine_sim':
            self.sim_func = py.cosine_sim
            self.sim_min_value = -1.0
        elif sim_func == 'py_cos_sim_sprs':
            self.sim_func = py.cos_sim_sprs
            self.sim_min_value = -1.0
        elif sim_func == 'py_minmax_sim':
            self.sim_func = py.minmax_sim
            self.sim_min_value = 0.0
        elif sim_func == 'py_jaccard_sim':
            self.sim_func = py.jaccard_sim
            self.sim_min_value = 0.0
        elif sim_func == 'py_hamming_sim':
            self.sim_func = py.hamming_sim
            self.sim_min_value = 0.0
        elif simfunc == 'py_correl_sim':
            self.sim_func = py.correl_sim
            self.sim_min_value = 0.0

        if bagging:
            print 'Init: RFSE with Bagging'
        else:
            print 'Init: RFSE'

        self.bagging = bagging

        # RFSE paramters
        self.itrs = itrs
        self.sigma = sigma
        self.feat_size = feat_size

        self.ci2gtag = dict()
        self.gnr_classes = list()

    def fit(self, trn_mtrx, cls_tgs):
        # It should be cls_tgs = cls_gnr_tgs[cls_tgs]
        # It should be trn_mtrx corpus_mtrx_lst[cls_tgs]

        if not self.bagging:

            for i, gnr_tag in enumerate(np.unique(cls_tgs)):
                self.ci2gtag[i] = gnr_tag
                self.gnr_classes.append(trn_mtrx[np.where((cls_tgs == gnr_tag))].mean(axis=0))

        else:

            self.trn_mtrx = trn_mtrx
            self.cls_tgs = cls_tgs

            for gnr_tag in np.unique(cls_tgs):

                # # # # # # #
                shuffled_train_idxs = np.random.permutation(np.where((cls_tgs == gnr_tag)))
                # print shuffled_train_idxs
                # keep bagging_parram percent
                bg_trn_ptg = int(np.trunc(shuffled_train_idxs.size * bagging_param))
                # print bg_trn_ptg
                bagg_idxs = shuffled_train_idxs[0:bg_trn_ptg]
                # print bag_idxs
                self.ci2gtag[i] = gnr_tag
                self.gnr_classes.append(trn_mtrx[bagg_idxs].mean(axis=0))

        # Converting the list to narray.
        self.gnr_classes = np.vstack(self.gnr_classes)

        return self.gnr_classes

    def predict(self, tst_mtrx):
        # It should be tst_mtrx = corpus_mtrx[crv_idxs]
        # It should be cls_tgs = cls_gnr_tgs[crv_idxs]

        mtrx_feat_idxs = np.arange(tst_mtrx.shape[1])

        max_sim_scores_per_iter = np.zeros((self.itrs, tst_mtrx.shape[0]))
        predicted_classes_per_iter = np.zeros((self.itrs, tst_mtrx.shape[0]))

        # Measure similarity for i iterations i.e. for i different feature subspaces Randomly...
        # ...selected
        for i in range(self.itrs):

            # Construct Genres Class Vectors form Training Set. In case self.bagging is True.
            if self.bagging:
                self.gnr_classes = self.contruct_classes(self.trn_mtrx, self.cls_tgs)

            # Randomly select some of the available features
            feat_subspace = np.random.permutation(mtrx_feat_idxs)[0:self.feat_size]

            # Initialized Predicted Classes and Maximum Similarity Scores Array for this i iteration
            sim_scrs = np.array(self.sim_func(tst_mtrx, self.gnr_classes, feat_subspace))
            max_sim_inds = np.argmax(sim_scrs, axis=1)
            max_sim_scores = sim_scrs[np.arange(sim_scrs.shape[0]), max_sim_inds]

            # Store Predicted Classes and Scores for this i iteration
            max_sim_scores_per_iter[i, :] = max_sim_scores[:]
            predicted_classes_per_iter[i, :] = np.array([self.ci2gtag[i] for i in max_sim_inds[:]])

        predicted_Y = np.zeros((tst_mtrx.shape[0]), dtype=np.float)
        predicted_scores = np.zeros((tst_mtrx.shape[0]), dtype=np.float)

        for i_prd_cls, prd_cls in enumerate(predicted_classes_per_iter.transpose()):
            genres_occs = np.histogram(
                prd_cls.astype(np.int), bins=np.arange(len(self.ci2gtag.keys()) + 2)
            )[0]
            # NOTE: One Bin per Genre plus one i.e the first to be always zero
            # print genres_occs
            genres_probs = genres_occs.astype(np.float) / np.float(self.itrs)
            # print genres_probs
            if np.max(genres_probs) >= self.sigma:
                predicted_Y[i_prd_cls] = np.argmax(genres_probs)
                predicted_scores[i_prd_cls] = np.max(genres_probs)

        return predicted_Y, predicted_scores, max_sim_scores_per_iter, predicted_classes_per_iter


class RFSEDMPG(RFSE):

    def __init__(self, *args, **kwrgs):

        # Initializing RFSE.
        super(RFSEDMPG, self).__init__(*args, **kwrgs)

    def predict(self, *args):

        # Get Input arguments in given sequence
        crv_idxs = args[0]
        corpus_mtrx_lst = args[1]
        cls_gnr_tgs = args[2]

        # Store the argument 5 (6th) to the proper variable
        if self.bagging and isinstance(args[4], np.ndarray):
            trn_idxs = args[4]

        elif not self.bagging and isinstance(args[4], dict):
            gnr_classes = args[4]

        else:
            raise Exception(
                'predict(): Invalid Argument, either bagging trigged with not train-index' +
                'array or non-bagging with not genre-classes argument'
            )

        # Get the part of matrices or arrays required for the model prediction phase.
        # crossval_X = corpus_mtrx_lst[self.gnrlst_idx[g]][crv_idxs, :]
        # NOTE: EXTREMELY IMPORTANT! corpus_mtrx_lst[X] where X=[<idx1>,<idx2>,...,<idxN>]...
        # returns ERROR HDF5 when using pytables Earray.  For scipy.sparse there is no such a...
        # ...problem. Therefore it always should be used this expression corpus_mtrx_lst[X, :]

        # Get the part of matrices required for the model prediction phase.
        # ###crossval_Y =  cls_gnr_tgs [crv_idxs, :]

        crossval_len = len(crv_idxs)
        crps_mtrx_shape = corpus_mtrx_lst[0].shape

        max_sim_scores_per_iter = np.zeros((self.itrs, crossval_len))
        predicted_classes_per_iter = np.zeros((self.itrs, crossval_len))

        # Measure similarity for i iterations i.e. for i different feature subspaces Randomly...
        # ...selected
        for i in range(self.itrs):

            # print "Construct classes"
            # Construct Genres Class Vectors form Training Set. In case self.bagging is True.
            if self.bagging:
                gnr_classes = self.contruct_classes(
                    trn_idxs, corpus_mtrx_lst, cls_gnr_tgs, bagging
                )

            # Randomly select some of the available features
            shuffled_vocabilary_idxs = np.random.permutation(np.arange(crps_mtrx_shape[1]))
            feat_subspace = shuffled_vocabilary_idxs[0: self.feat_size]

            # Initialized Predicted Classes and Maximum Similarity Scores Array for this i iteration
            predicted_classes = np.zeros(crossval_len)
            max_sim_scores = np.zeros(crossval_len)

            # Measure similarity for each Cross-Validation-Set vector to each available Genre...
            # ...Class(i.e. Class-Vector). For This feature_subspace.
            for i_vect, crv_i in enumerate(crv_idxs):

                # Convert TF vectors to Binary
                # vect_bin = np.where(vect[:, :].toarray() > 0, 1, 0)
                # NOTE: with np.where Always use A[:] > x instead of A > x in case of...
                # ...Sparse Matrices
                # print vect.shape

                max_sim = self.sim_min_value
                for g in gnr_classes.keys():

                    # Get the part of matrices or arrays required for the model prediction phase.
                    # crossval_X[crv_i] <== Equivalent
                    vect = corpus_mtrx_lst[self.gnrlst_idx[g]][crv_i, feat_subspace]

                    # Convert TF vectors to Binary
                    # gnr_cls_bin = np.where(gnr_classes[g][:, feat_subspace] > 0, 1, 0)
                    # print gnr_cls_bin.shape

                    # Measure Similarity
                    if gnr_classes[g].ndim == 2:
                        # This case is called when a Sparse Matrix is used which is alway 2D...
                        # ...with first dim == 1
                        sim_score = self.sim_func(vect, gnr_classes[g][:, feat_subspace])

                    elif gnr_classes[g].ndim == 1:
                        # This case is called when a Array or pyTables-Array is used which it...
                        # ...this case should be 1D
                        sim_score = self.sim_func(vect, gnr_classes[g][feat_subspace])

                    else:
                        raise Exception(
                            "Unexpected Centroid Vector Dimensions: its shape should be " +
                            "(x,) for 1D array or (1,x) for 2D array or matrix"
                        )

                    # Just for debugging for
                    # if sim_score < 0.0:
                    #     print "ERROR: Similarity score unexpected value ", sim_score

                    # Assign the class tag this vector is most similar and keep the respective...
                    # ...similarity score.
                    if sim_score > max_sim:
                        predicted_classes[i_vect] = self.genres_lst.index(g) + 1
                        # ###plus 1 is the real class tag 0 means uncategorized.
                        max_sim_scores[i_vect] = sim_score
                        max_sim = sim_score

            # Store Predicted Classes and Scores for this i iteration
            max_sim_scores_per_iter[i, :] = max_sim_scores[:]
            predicted_classes_per_iter[i, :] = predicted_classes[:]

        predicted_Y = np.zeros((crossval_len), dtype=np.float)
        predicted_scores = np.zeros((crossval_len), dtype=np.float)

        for i_prd_cls, prd_cls in enumerate(predicted_classes_per_iter.transpose()):
            genres_occs = np.histogram(prd_cls.astype(np.int), bins=np.arange(self.gnrs_num+2))[0]
            # NOTE: One Bin per Genre plus one i.e the first to be always zero
            # print genres_occs
            genres_probs = genres_occs.astype(np.float) / np.float(self.itrs)
            # print genres_probs
            if np.max(genres_probs) >= self.sigma:
                predicted_Y[i_prd_cls] = np.argmax(genres_probs)
                predicted_scores[i_prd_cls] = np.max(genres_probs)

        return predicted_Y, predicted_scores, max_sim_scores_per_iter, predicted_classes_per_iter


def cosine_similarity(vector, centroid):

    # Convert Arrays and Matrices to Matrix Type for preventing unexpected error, such as...
    # ...returning vector instead of scalar
    vector = np.matrix(vector)
    centroid = np.matrix(centroid)

    return vector * np.transpose(centroid) / (np.linalg.norm(vector) * np.linalg.norm(centroid))


def cosine_similarity_sparse(vector, centroid):

    return cosine_similarity(vector.todense(), centroid)


def minmax_similarity(v1, v2):

    return np.sum(np.min(np.vstack((v1, v2)), axis=0)) / np.sum(np.max(np.vstack((v1, v2)), axis=0))


def jaccard_similarity_binconv(v0, v1):

    v0 = np.where((v0 > 0), 1, 0)
    v1 = np.where((v0 > 0), 1, 0)

    return 1.0 - spd.jaccard(v0, v1)


def hamming_similarity(vector, centroid):

    return 1.0 - spd.hamming(centroid, vector)


def correlation_similarity(vector, centroid):

    vector = vector[0]
    centroid = np.array(centroid)[0]

    vector_ = np.where(vector > 0, 0, 1)
    centroid_ = np.where(centroid > 0, 0, 1)

    s11 = np.dot(vector, centroid)
    s00 = np.dot(vector_, centroid_)
    s01 = np.dot(vector_, centroid)
    s10 = np.dot(vector, centroid_)

    denom = np.sqrt((s10+s11)*(s01+s00)*(s11+s01)*(s00+s10))
    if denom == 0.0:
        denom = 1.0

    return (s11*s00 - s01*s10) / denom
