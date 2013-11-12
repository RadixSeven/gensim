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
        model.build_vocab(model.vocab_counts(corpus))
        self.assertTrue(len(model.vocab) == 6981)
        # with min_count=1, we're not throwing away anything, so make sure the word counts add up to be the entire corpus
        self.assertTrue(sum(v.count for v in model.vocab.itervalues()) == total_words)
        # make sure the binary codes are correct
        numpy.allclose(model.vocab['the'].code, [1, 1, 0, 0])

        # test building vocab with default params
        model = word2vec.Word2Vec()
        model.build_vocab(model.vocab_counts(corpus))
        self.assertTrue(len(model.vocab) == 1750)
        numpy.allclose(model.vocab['the'].code, [1, 1, 1, 0])

        # no input => "RuntimeError: you must first build vocabulary before training the model"
        self.assertRaises(RuntimeError, word2vec.Word2Vec, [])

        # input not empty, but rather completely filtered out
        self.assertRaises(RuntimeError, word2vec.Word2Vec, corpus, min_count=total_words+1)


    def testTraining(self):
        """Test word2vec training."""
        # to test training, make the corpus larger by repeating its sentences over and over
        # build vocabulary, don't train yet
        model = word2vec.Word2Vec(size=2, min_count=1)
        model.build_vocab(model.vocab_counts(sentences))
        self.assertTrue(model.syn0.shape == (len(model.vocab), 2))
        self.assertTrue(model.syn1.shape == (len(model.vocab), 2))

        model.train(sentences)
        sims = model.most_similar('graph')
        self.assertTrue(sims[0][0] == 'trees', sims)  # most similar

        # build vocab and train in one step; must be the same as above
        model2 = word2vec.Word2Vec(sentences, size=2, min_count=1)
        self.models_equal(model, model2)

    def testMaxWords(self):
        """Test that build_vocab enforces the max_words constraint"""
        s = [
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

        # Test with min_count = 1, max_words=None
        from numpy import array, uint8, uint32

        model=word2vec.Word2Vec(min_count=1)

        model.build_vocab( model.vocab_counts(s) )
        expected_vocab = {}
        expected_vocab['computer'] = word2vec.Vocab(
            count = 2, index = 5, code=array(dtype=uint8, object=[1, 1, 1, 0]),
            point = array(dtype=uint32, object=[10,  9,  6,  1]))
        expected_vocab['eps'] = word2vec.Vocab(
            count = 2, index = 4, code=array(dtype=uint8, object=[1, 0, 0, 1]),
            point = array(dtype=uint32, object=[10,  9,  7,  2]))
        expected_vocab['graph'] = word2vec.Vocab(
            count = 3, index = 1, code=array(dtype=uint8, object=[0, 0, 0]),
            point = array(dtype=uint32, object=[10,  8,  4]))
        expected_vocab['human'] = word2vec.Vocab(
            count = 2, index = 8, code=array(dtype=uint8, object=[1, 0, 1, 1]),
            point = array(dtype=uint32, object=[10,  9,  7,  3]))
        expected_vocab['interface'] = word2vec.Vocab(
            count = 2, index = 10, code=array(dtype=uint8, object=[1, 0, 0, 0]),
            point = array(dtype=uint32, object=[10,  9,  7,  2]))
        expected_vocab['minors'] = word2vec.Vocab(
            count = 2, index = 0, code=array(dtype=uint8, object=[1, 1, 1, 1]),
            point = array(dtype=uint32, object=[10,  9,  6,  1]))
        expected_vocab['response'] = word2vec.Vocab(
            count = 2, index = 11, code=array(dtype=uint8, object=[1, 1, 0, 1]),
            point = array(dtype=uint32, object=[10,  9,  6,  0]))
        expected_vocab['survey'] = word2vec.Vocab(
            count = 2, index = 6, code=array(dtype=uint8, object=[1, 1, 0, 0]),
            point = array(dtype=uint32, object=[10,  9,  6,  0]))
        expected_vocab['system'] = word2vec.Vocab(
            count = 4, index = 2, code=array(dtype=uint8, object=[0, 1, 1]),
            point = array(dtype=uint32, object=[10,  8,  5]))
        expected_vocab['time'] = word2vec.Vocab(
            count = 2, index = 9, code=array(dtype=uint8, object=[1, 0, 1, 0]),
            point = array(dtype=uint32, object=[10,  9,  7,  3]))
        expected_vocab['trees'] = word2vec.Vocab(
            count = 3, index = 3, code=array(dtype=uint8, object=[0, 1, 0]),
            point = array(dtype=uint32, object=[10,  8,  5]))
        expected_vocab['user'] = word2vec.Vocab(
            count = 3, index = 7, code=array(dtype=uint8, object=[0, 0, 1]),
            point = array(dtype=uint32, object=[10,  8,  4]))
        expected_index2word = [
            'minors', 'graph', 'system', 'trees', 'eps', 'computer',
            'survey', 'user', 'human', 'time', 'interface', 'response']


        self.maxDiff = 8192
        self.assertEqual(model.vocab, expected_vocab)
        self.assertEqual(model.index2word, expected_index2word)

        # Test with min_count = 3, max_words = None

        model=word2vec.Word2Vec(min_count=3)

        model.build_vocab( model.vocab_counts(s) )
        expected_vocab = {}
        expected_vocab['graph'] = word2vec.Vocab(
            count = 3, index = 0, code=array(dtype=uint8, object=[0, 1]),
            point = array(dtype=uint32, object=[2, 0]))
        expected_vocab['system'] = word2vec.Vocab(
            count = 4, index = 1, code=array(dtype=uint8, object=[1, 1]),
            point = array(dtype=uint32, object=[2, 1]))
        expected_vocab['trees'] = word2vec.Vocab(
            count = 3, index = 2, code=array(dtype=uint8, object=[0, 0]),
            point = array(dtype=uint32, object=[2, 0]))
        expected_vocab['user'] = word2vec.Vocab(
            count = 3, index = 3, code=array(dtype=uint8, object=[1, 0]),
            point = array(dtype=uint32, object=[2, 1]))
        expected_index2word = [
            'graph', 'system', 'trees', 'user']


        self.maxDiff = 4096
        self.assertEqual(model.vocab, expected_vocab)
        self.assertEqual(model.index2word, expected_index2word)

        # Test with min_count = 1, max_words = 5

        model=word2vec.Word2Vec(min_count=1, max_words = 5)

        model.build_vocab( model.vocab_counts(s) )
        expected_vocab = {}
        expected_vocab['graph'] = word2vec.Vocab(
            count = 3, index = 0, code=array(dtype=uint8, object=[0, 1]),
            point = array(dtype=uint32, object=[2, 0]))
        expected_vocab['system'] = word2vec.Vocab(
            count = 4, index = 1, code=array(dtype=uint8, object=[1, 1]),
            point = array(dtype=uint32, object=[2, 1]))
        expected_vocab['trees'] = word2vec.Vocab(
            count = 3, index = 2, code=array(dtype=uint8, object=[0, 0]),
            point = array(dtype=uint32, object=[2, 0]))
        expected_vocab['user'] = word2vec.Vocab(
            count = 3, index = 3, code=array(dtype=uint8, object=[1, 0]),
            point = array(dtype=uint32, object=[2, 1]))
        expected_index2word = [
            'graph', 'system', 'trees', 'user']
        expected_index2word = [
            'graph', 'system', 'trees', 'user']


        self.maxDiff = 4096
        self.assertEqual(model.vocab, expected_vocab)
        self.assertEqual(model.index2word, expected_index2word)

        # Test with min_count = 1, max_words = 1

        model=word2vec.Word2Vec(min_count=1, max_words = 1)

        model.build_vocab( model.vocab_counts(s) )
        expected_vocab = {}
        expected_vocab['system'] = word2vec.Vocab(
            count = 4, index = 0, code = [], point=[])
        expected_index2word = [
            'system']


        self.maxDiff = 4096
        self.assertEqual(model.vocab, expected_vocab)
        self.assertEqual(model.index2word, expected_index2word)

    def testVocabCounts(self):
        """Test that vocab_counts returns the right counts on some simple vocabularies"""
        s = [
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

        # Test with min_count = 1

        model=word2vec.Word2Vec(min_count=1)
        
        model.create_indexed_vocab( model.vocab_counts(s) )
        expected_vocab = {}
        expected_vocab['computer'] = word2vec.Vocab(count = 2, index = 5)
        expected_vocab['eps'] = word2vec.Vocab(count = 2, index = 4)
        expected_vocab['graph'] = word2vec.Vocab(count = 3, index = 1)
        expected_vocab['human'] = word2vec.Vocab(count = 2, index = 8)
        expected_vocab['interface'] = word2vec.Vocab(count = 2, index = 10)
        expected_vocab['minors'] = word2vec.Vocab(count = 2, index = 0)
        expected_vocab['response'] = word2vec.Vocab(count = 2, index = 11)
        expected_vocab['survey'] = word2vec.Vocab(count = 2, index = 6)
        expected_vocab['system'] = word2vec.Vocab(count = 4, index = 2)
        expected_vocab['time'] = word2vec.Vocab(count = 2, index = 9)
        expected_vocab['trees'] = word2vec.Vocab(count = 3, index = 3)
        expected_vocab['user'] = word2vec.Vocab(count = 3, index = 7)
        expected_index2word = [
            'minors', 'graph', 'system', 'trees', 'eps', 'computer',
            'survey', 'user', 'human', 'time', 'interface', 'response']


        self.maxDiff = 4096
        self.assertEqual(model.vocab, expected_vocab)
        self.assertEqual(model.index2word, expected_index2word)

        # Test with min_count = 3

        model=word2vec.Word2Vec(min_count=3)
        
        model.create_indexed_vocab( model.vocab_counts(s) )
        expected_vocab = {}
        expected_vocab['graph'] = word2vec.Vocab(count = 3, index = 0)
        expected_vocab['system'] = word2vec.Vocab(count = 4, index = 1)
        expected_vocab['trees'] = word2vec.Vocab(count = 3, index = 2)
        expected_vocab['user'] = word2vec.Vocab(count = 3, index = 3)
        expected_index2word = [
            'graph', 'system', 'trees', 'user']


        self.maxDiff = 4096
        self.assertEqual(model.vocab, expected_vocab)
        self.assertEqual(model.index2word, expected_index2word)


    def testCreateIndexedVocab(self):
        """Test that create_indexed_vocab eliminates the correct vocab entries"""
        s = [
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

        model=word2vec.Word2Vec(min_count=1)
        
        uni = model.vocab_counts(s)
        expected_uni = {}
        expected_uni['computer'] = word2vec.Vocab(count = 2)
        expected_uni['eps'] = word2vec.Vocab(count = 2)
        expected_uni['graph'] = word2vec.Vocab(count = 3)
        expected_uni['human'] = word2vec.Vocab(count = 2)
        expected_uni['interface'] = word2vec.Vocab(count = 2)
        expected_uni['minors'] = word2vec.Vocab(count = 2)
        expected_uni['response'] = word2vec.Vocab(count = 2)
        expected_uni['survey'] = word2vec.Vocab(count = 2)
        expected_uni['system'] = word2vec.Vocab(count = 4)
        expected_uni['time'] = word2vec.Vocab(count = 2)
        expected_uni['trees'] = word2vec.Vocab(count = 3)
        expected_uni['user'] = word2vec.Vocab(count = 3)

        uni_str = '{'+', '.join([str(k)+":"+str(v) for k,v in uni.iteritems()])+'}'
        expected_uni_str = '{'+', '.join([str(k)+":"+str(v) 
                                          for k,v in expected_uni.iteritems()])+'}'
        self.assertEqual(uni, expected_uni, uni_str+
                         " NOT EQUAL TO "+expected_uni_str)

        empty_sentences = [[]]
        uni = model.vocab_counts(empty_sentences)
        expected_uni = {}

        self.assertEqual(uni, expected_uni)

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
