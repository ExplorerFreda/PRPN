import os
import re
import pickle
import copy

import numpy
import torch
import nltk
from nltk.corpus import ptb
from nltk.tree import Tree

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']


class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.items():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print('Number of words:', len(self.idx2word))
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        dict_file_name = os.path.join(path, 'dict.pkl')
        train_trees = [Tree.fromstring(line) for line in open(os.path.join(path, 'train.txt'))]
        valid_trees = [Tree.fromstring(line) for line in open(os.path.join(path, 'dev.txt'))]
        test_trees = [Tree.fromstring(line) for line in open(os.path.join(path, 'test.txt'))]
        if os.path.exists(dict_file_name):
            self.dictionary = pickle.load(open(dict_file_name, 'rb'))
        else:
            self.dictionary = Dictionary()
            self.add_words(train_trees)
            self.add_words(valid_trees)
            self.add_words(test_trees)
            self.dictionary.rebuild_by_freq()
            pickle.dump(self.dictionary, open(dict_file_name, 'wb'))
        
        data_file_name = os.path.join(path, 'data.pkl')
        if os.path.exists(data_file_name):
            info = pickle.load(open(data_file_name, 'rb'))
            self.train, self.train_sens, self.train_trees = info['train']
            self.valid, self.valid_sens, self.valid_trees = info['valid']
            self.test, self.test_sens, self.test_trees = info['test']
        else:
            self.train, self.train_sens, self.train_trees = self.tokenize(train_trees)
            self.valid, self.valid_sens, self.valid_trees = self.tokenize(valid_trees)
            self.test, self.test_sens, self.test_trees = self.tokenize(test_trees)
            pickle.dump(
                {
                    'train': (self.train, self.train_sens, self.train_trees), 
                    'valid': (self.valid, self.valid_sens, self.valid_trees),
                    'test': (self.test, self.test_sens, self.test_trees)
                }, 
                open(data_file_name, 'wb')
            )

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            w = w.lower()
            w = re.sub('[0-9]+', 'N', w)
            words.append(w)
        return words

    def add_words(self, trees):
        # Add words to the dictionary
        for sen_tree in trees:
            words = self.filter_words(sen_tree)
            words = ['[CLS]'] + words + ['[SEP]']
            for word in words:
                self.dictionary.add_word(word)

    def tokenize(self, trees):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in word_tags:
                    return tree.leaves()[0]
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        sens_idx = []
        sens = []
        for sen_tree in trees:
            words = self.filter_words(sen_tree)
            words = ['[CLS]'] + words + ['[SEP]']
            sens.append(words)
            idx = []
            for word in words:
                idx.append(self.dictionary[word])
            sens_idx.append(torch.LongTensor(idx))

        return sens_idx, sens, trees
