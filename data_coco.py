import json
import os
import re
import pickle
import copy

import numpy
import torch
import nltk
from nltk.corpus import ptb


class Dictionary(object):
    def __init__(self, path=None):
        if path is not None:
            self.word2idx = json.load(open(path))
            self.idx2word = ['' for _ in range(len(self.word2idx))]
            for word in self.word2idx:
                self.idx2word[self.word2idx[word]] = word
        else:
            self.word2idx = {'<unk>': 0}
            self.idx2word = ['<unk>']
            self.word2frq = {}

    def __getitem__(self, item):
        if item in self.word2idx:
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path):
        dict_file_name = os.path.join(path, 'vocab.json')
        if os.path.exists(dict_file_name):
            self.dictionary = Dictionary(dict_file_name)
        else:
            raise Exception('File not found: {:s}'.format(dict_file_name))

        self.train, self.train_sens, self.train_trees = self.tokenize(
            os.path.join(path, 'train.txt'), os.path.join(path, 'train_trees.txt'))
        self.valid, self.valid_sens, self.valid_trees = self.tokenize(
            os.path.join(path, 'dev.txt'), os.path.join(path, 'dev_trees.txt'))
        self.test, self.test_sens, self.test_trees = self.tokenize(
            os.path.join(path, 'test.txt'), os.path.join(path, 'test_trees.txt'))

    def tokenize(self, main_file, tree_file):

        def tree2list(tokens):
            tree = []
            list_stack = []
            list_stack.append(tree)
            stack_top = tree
            for token in tokens:
                if token == '(':
                    new_span = []
                    stack_top.append(new_span)
                    list_stack.append(new_span)
                    stack_top = new_span
                elif token == ')':
                    list_stack = list_stack[:-1]
                    if len(list_stack) != 0:
                        stack_top = list_stack[-1]
                else:
                    stack_top.append(token)
            return tree

        sens_idx = []
        sens = []
        trees = []
        sentences = open(main_file).readlines()
        raw_trees = open(tree_file).readlines()
        for i, sentence in enumerate(sentences):
            words = sentence.strip().split()
            words = ['<start>'] + words + ['<end>']
            sens.append(words)
            idx = []
            for word in words:
                idx.append(self.dictionary[word])
            sens_idx.append(torch.LongTensor(idx))
            trees.append(tree2list(raw_trees[i].strip().split()))
        return sens_idx, sens, trees
