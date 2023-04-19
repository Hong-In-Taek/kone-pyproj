import pandas as pd
import numpy as np
import nltk
import string
from nltk.classify import NaiveBayesClassifier
import os
nltk.download('stopwords')
nltk.download('punkt')


class mbtiNPmodel:
    def __init__(self):
        self.data_set = pd.read_csv('static/mbtiData.csv')
        self.types = np.unique(np.array(self.data_set['type']))
        self.all_posts = pd.DataFrame()
        for j in self.types:
            temp1 = self.data_set[self.data_set['type'] == j]['posts']
            temp2 = []
            for i in temp1:
                temp2 += i.split('|||')
            temp3 = pd.Series(temp2)
            self.all_posts[j] = temp3
        self.useless_words = nltk.corpus.stopwords.words(
            "english") + list(string.punctuation)
        self.train()

    def build_bag_of_words_features_filtered(self, words):
        words = nltk.word_tokenize(words)
        return {
            word: 1 for word in words
            if not word in self.useless_words}

    def train(self):
        self.features = []
        for j in self.types:
            temp1 = self.all_posts[j]
            temp1 = temp1.dropna()  # not all the personality types have same number of files
            self.features += [[(self.build_bag_of_words_features_filtered(i), j)
                               for i in temp1]]

        self.split = []
        for i in range(16):
            self.split += [len(self.features[i]) * 0.8]
        self.split = np.array(self.split, dtype=int)
        # Features for the bag of words model
        features = []
        featuresJP = []
        featuresNS = []
        featuresIE = []
        for j in self.types:
            temp1 = self.all_posts[j]
            temp1 = temp1.dropna()  # not all the personality types have same number of files
            if ('T' in j):
                features += [[(self.build_bag_of_words_features_filtered(i), 'Thinking')
                              for i in temp1]]
            if ('F' in j):
                features += [[(self.build_bag_of_words_features_filtered(i), 'Feeling')
                              for i in temp1]]
            if ('J' in j):
                featuresJP += [[(self.build_bag_of_words_features_filtered(i), 'Judging')
                                for i in temp1]]
            if ('P' in j):
                featuresJP += [[(self.build_bag_of_words_features_filtered(i), 'Percieving')
                                for i in temp1]]
            if ('N' in j):
                featuresNS += [[(self.build_bag_of_words_features_filtered(i), 'Intuition')
                                for i in temp1]]
            if ('S' in j):
                featuresNS += [[(self.build_bag_of_words_features_filtered(i), 'Sensing')
                                for i in temp1]]
            if ('I' in j):
                featuresIE += [[(self.build_bag_of_words_features_filtered(i), 'introvert')
                                for i in temp1]]
            if ('E' in j):
                featuresIE += [[(self.build_bag_of_words_features_filtered(i), 'extrovert')
                                for i in temp1]]

        train = []
        trainJP = []
        trainNS = []
        trainIE = []
        for i in range(16):
            train += features[i][:self.split[i]]
            trainJP += featuresJP[i][:self.split[i]]
            trainNS += featuresNS[i][:self.split[i]]
            trainIE += featuresIE[i][:self.split[i]]
        self.ThinkingFeeling = NaiveBayesClassifier.train(train)
        self.JudgingPercieiving = NaiveBayesClassifier.train(trainJP)
        self.IntuitionSensing = NaiveBayesClassifier.train(trainNS)
        self.IntroExtro = NaiveBayesClassifier.train(trainIE)

    def MBTI(self, input):
        tokenize = self.build_bag_of_words_features_filtered(input)
        ie = self.IntroExtro.classify(tokenize)
        Is = self.IntuitionSensing.classify(tokenize)
        tf = self.ThinkingFeeling.classify(tokenize)
        jp = self.JudgingPercieiving.classify(tokenize)

        mbt = ''

        if (ie == 'introvert'):
            mbt += 'I'
        if (ie == 'extrovert'):
            mbt += 'E'
        if (Is == 'Intuition'):
            mbt += 'N'
        if (Is == 'Sensing'):
            mbt += 'S'
        if (tf == 'Thinking'):
            mbt += 'T'
        if (tf == 'Feeling'):
            mbt += 'F'
        if (jp == 'Judging'):
            mbt += 'J'
        if (jp == 'Percieving'):
            mbt += 'P'
        return (mbt)

    def tellmemyMBTI(self, input, name, traasits=[]):
        a = []
        trait1 = pd.DataFrame([0, 0, 0, 0], ['I', 'N', 'T', 'J'], ['count'])
        trait2 = pd.DataFrame([0, 0, 0, 0], ['E', 'S', 'F', 'P'], ['count'])
        for i in input:
            a += [self.MBTI(i)]
        for i in a:
            for j in ['I', 'N', 'T', 'J']:
                if (j in i):
                    trait1.loc[j] += 1
            for j in ['E', 'S', 'F', 'P']:
                if (j in i):
                    trait2.loc[j] += 1
        trait1 = trait1.T
        trait1 = trait1*100/len(input)
        trait2 = trait2.T
        trait2 = trait2*100/len(input)

        # Finding the personality
        YourTrait = ''
        for i, j in zip(trait1, trait2):
            temp = max(trait1[i][0], trait2[j][0])
            if (trait1[i][0] == temp):
                YourTrait += i
            if (trait2[j][0] == temp):
                YourTrait += j
        traasits += [YourTrait]
        return YourTrait
