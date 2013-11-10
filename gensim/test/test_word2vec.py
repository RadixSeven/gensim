#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2010 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import tempfile
import itertools
import bz2
import bisect

import sys
import gc

import numpy

from gensim import utils, matutils
from gensim.models import word2vec

module_path = os.path.dirname(__file__) # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class LeeCorpus(object):
    def __iter__(self):
        for line in open(datapath('lee_background.cor')):
            yield utils.simple_preprocess(line)


sentences = [
    ['human', 'interface', 'computer'],
    ['survey', 'user', 'computer', 'system', 'response', 'time'],
    ['eps', 'user', 'interface', 'system'],
    ['system', 'human', 'system', 'eps'],
    ['user', 'response', 'time'],
    ['trees'],
    ['graph', 'trees'],
    ['graph', 'minors', 'trees'],
    ['graph', 'minors', 'survey']
]


def testfile():
    # temporary data will be stored to this file
    return os.path.join(tempfile.gettempdir(), 'gensim_word2vec.tst')


class MemoryAllocationFailsList(object):
    """ A pseudo list which has members indicating if a numpy array of that many floats could be allocated 

    For example: 

    if `maf = MemoryAllocationFails()`

    `maf[1000]` would attempt to allocate an array of 1000 floats and
    return 1 if it failed and 0 if it succeeded.

    This is useful to allow binary search for memory size. For this
    use, it is important that return value for successful allocations
    is lower than that for unsuccessful allocations.
    """
    def __init__(self):
        pass

    def __getitem__(self, key):
        sys.stderr.write('Try %s entries: ' % key)
        try:
            gc.collect()
            [0]*key
            sys.stderr.write('ok\n')
            return 0
        except MemoryError:
            sys.stderr.write('FAILED\n')
            return 1

class IncreasingNumberSentences(object):
    """Each sentence is an increasing list of numbers.

    The first 4 sentences are:
    0
    0 1
    0 1 2
    0 1 2 3

    You can specify the number of sentences in the collection in the
    constructor.

    In the resulting vocabulary 0 will appear `num_sentences` times, 1
    will appear `num_sentences-1` times and `i` will appear
    `num_sentences-i` times.
    """
    def __init__(self, num_sentences):
        """Create a set of `num_sentences` sentences """
        self.num_sentences = num_sentences
    def __iter__(self):
        """Iterate through the sentences"""
        for length in range(self.num_sentences):
            yield ' '.join(map(str, range(length)))

class TestWord2VecModel(unittest.TestCase):
    def testPersistence(self):
        """Test storing/loading the entire model."""
        model = word2vec.Word2Vec(sentences, min_count=1)
        model.save(testfile())
        self.models_equal(model, word2vec.Word2Vec.load(testfile()))

    def testVocab(self):
        """Test word2vec vocabulary building."""
        corpus = LeeCorpus()
        total_words = sum(len(sentence) for sentence in corpus)

        # try vocab building explicitly, using all words
        model = word2vec.Word2Vec(min_count=1)
        model.build_vocab(corpus)
        self.assertTrue(len(model.vocab) == 6981)
        # with min_count=1, we're not throwing away anything, so make sure the word counts add up to be the entire corpus
        self.assertTrue(sum(v.count for v in model.vocab.itervalues()) == total_words)
        # make sure the binary codes are correct
        numpy.allclose(model.vocab['the'].code, [1, 1, 0, 0])

        # test building vocab with default params
        model = word2vec.Word2Vec()
        model.build_vocab(corpus)
        self.assertTrue(len(model.vocab) == 1750)
        numpy.allclose(model.vocab['the'].code, [1, 1, 1, 0])

        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, word2vec.Word2Vec, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, word2vec.Word2Vec, corpus, min_count=total_words+1)

    def testVocabAutoadjust(self):
        "Test that the autoadjustment happens when memory is filled"
        
        # First, find how much memory is available
        gc.collect()
        alloc_fails = MemoryAllocationFailsList()
        # Find the first power of 2 for which allocation fails
        mem_upper_bound = 1
        while not alloc_fails[mem_upper_bound]:
            mem_upper_bound *= 2
        # Now use the known bounds to find out exactly how much memory
        # is available
        num_avail_entries = bisect.bisect(
            alloc_fails, 0.5, mem_upper_bound/2, mem_upper_bound)
        
        # Now allocate all but 64000 entries of it
        gc.collect()
        fill_memory = [0]*(num_avail_entries-64000)

        # Find the first power of 2 for which allocation fails
        mem_upper_bound = 1
        while not alloc_fails[mem_upper_bound]:
            mem_upper_bound *= 2
        # Now use the known bounds to find out exactly how much memory
        # is available
        num_avail_entries = bisect.bisect(
            alloc_fails, 0.5, mem_upper_bound/2, mem_upper_bound)

        sys.stderr.write('Number of avail entries after the filling: '+
                         str(num_avail_entries)+'\n');

        # Now allocate as much as possible of the rest of it
        gc.collect()
        fill_memory2 = [0]*(num_avail_entries-64000)

        # Now find out how many sentences are required for a memory error
        num_sentences = 200
        try:
            while True:
                sentences = IncreasingNumberSentences(num_sentences);
                model = word2vec.Word2Vec(sentences, size=4, min_count=1, 
                                          min_count_autoadjust=False)
                num_sentences = num_sentences * 2
        except MemoryError:
            pass

        # Train with that number of sentences using autoadjust - the
        # adjust shoud change min_count
        sentences = IncreasingNumberSentences(num_sentences);
        model = word2vec.Word2Vec(sentences, size=4, min_count=1, 
                                  min_count_autoadjust=True)
        self.assertTrue(model.min_count > 1, 'No entries were eliminated');

        # Ensure our memory-filling array doesn't get optimized away
        fill_memory[1] = 1;
    def testTraining(self):
        """Test word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(size=2, min_count=1)
        model.build_vocab(sentences)
        self.assertTrue(model.syn0.shape == (len(model.vocab), 2))
        self.assertTrue(model.syn1.shape == (len(model.vocab), 2))

        model.train(sentences)
        sims = model.most_similar('graph')
        self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, size=2, min_count=1)
        self.models_equal(model, model2)


    def testParallel(self):
        """Test word2vec parallel training."""
        if word2vec.FAST_VERSION < 0:  # don't test the plain NumPy version for parallelism (too slow)
            return

        corpus = utils.RepeatCorpus(LeeCorpus(), 10000)

        for workers in [2, 4]:
            model = word2vec.Word2Vec(corpus, workers=workers)
            sims = model.most_similar('israeli')
            # the exact vectors and therefore similarities may differ, due to different thread collisions
            # so let's test only for top3
            self.assertTrue('palestinian' in [sims[i][0] for i in xrange(3)])


    def testRNG(self):
        """Test word2vec results identical with identical RNG seed."""
        model = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        model2 = word2vec.Word2Vec(sentences, min_count=2, seed=42, workers=1)
        self.models_equal(model, model2)


    def models_equal(self, model, model2):
        self.assertEqual(len(model.vocab), len(model2.vocab))
        self.assertTrue(numpy.allclose(model.syn0, model2.syn0))
        self.assertTrue(numpy.allclose(model.syn1, model2.syn1))
        most_common_word = max(model.vocab.iteritems(), key=lambda item: item[1].count)[0]
        self.assertTrue(numpy.allclose(model[most_common_word], model2[most_common_word]))
#endclass TestWord2VecModel

class TestWord2VecSentenceIterators(unittest.TestCase):
    def testLineSentenceWorksWithFilename(self):
        """Does LineSentence work with a filename argument?"""
        with open(datapath('lee_background.cor')) as orig:
            sentences = word2vec.LineSentence(datapath('lee_background.cor'))
            for words in sentences:
                self.assertEqual(words, orig.readline().split())

    def testLineSentenceWorksWithCompressedFile(self):
        """Does LineSentence work with a compressed file object argument?"""
        with open(datapath('head500.noblanks.cor')) as orig:
            sentences = word2vec.LineSentence(
                bz2.BZ2File(
                    datapath('head500.noblanks.cor.bz2')))
            for words in sentences:
                self.assertEqual(words, orig.readline().split())

    def testLineSentenceWorksWithNormalFile(self):
        """Does LineSentence work with a normal file object argument?"""
        with open(datapath('head500.noblanks.cor')) as orig:
            sentences = word2vec.LineSentence(
                open(datapath('head500.noblanks.cor')))
            for words in sentences:
                self.assertEqual(words, orig.readline().split())
#endclass TestWord2VecSentenceIterators


if __name__ == '__main__':
    logging.root.setLevel(logging.DEBUG)
    unittest.main()
