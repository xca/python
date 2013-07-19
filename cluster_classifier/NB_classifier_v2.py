#-------------------------------------------------------------------------------
# Name:        TAMU csce 670 hw3 part2
# Purpose:  work with Bing API; NB classification
#
# Author:      Xixu Cai
#
#-------------------------------------------------------------------------------

import sys
import os
import re
import json
import array
from collections import defaultdict
from numpy import *
import math
import time
import pickle
import requests # Get from https://github.com/kennethreitz/requests
import string
import random

class classification:
    def __init__(self):
        self.cat = ['entertainment', 'business', 'politics']
        self.train_ent_file = 'class_train_ent_file'
        self.train_bus_file = 'class_train_bus_file'
        self.train_pol_file = 'class_train_pol_file'
        self.test_ent_file = 'class_test_ent_file'
        self.test_bus_file = 'class_test_bus_file'
        self.test_pol_file = 'class_test_pol_file'
        
        self.ent_train = defaultdict(list) #self.ent_train[docID] = ['title', 'description']
        self.pol_train = defaultdict(list)
        self.bus_train = defaultdict(list)
        self.ent_test = defaultdict(list)
        self.pol_test = defaultdict(list)
        self.bus_test = defaultdict(list)
        #when parsing json, calculate these numbers of documents for prior probability
        self.total_train = 0
        self.num_ent_train = 0
        self.num_bus_train = 0
        self.num_pol_train = 0
        
        self.total_test = 0
        self.num_ent_test = 0
        self.num_bus_test = 0
        self.num_pol_test = 0

        #prior class probabilities
        self.prob = []
        
        #conditional probabilities
        self.tf = defaultdict(list) #tf[term][0] = int (0: ent; 1:bus; 2:pol)
        self.condprob = defaultdict(list) #self.condprob[term][0] = float
        #the following are for feature selection chi-square score
        self.df = defaultdict(list) #df[term][0] = int (number of docs in cat[0] that contains term)
        self.score = defaultdict(dict) #score[cat[0]][term]=float; score[cat[1]][term] = float; score[cat[2]][term] = float; etc.
        self.features = [] #terms that can be used for calculation

        self.train_queries = [['bing', 'amazon', 'twitter', 'yahoo', 'google'],
                        ['beyonce', 'bieber', 'television', 'movie', 'music'],
                        ['obama', 'america', 'congress', 'senate', 'lawmakers']]
        self.test_queries = ['apple', 'facebook', 'westeros', 'gonzaga', 'banana']
        self.v = [] #all terms in the training set
        #self.true_train_class = defaultdict(list) #true_class[docID] = 'category'
        self.true_test_class = {} #true_class[docID] = 'category'
        self.test_class = {} #test_class[docID] = 'category'
        
    def getTerms(self, line):
        """helper function; parse and get terms from a string"""
        line = line.lower()
        line = re.split(r'[\W]+', line, 0, re.UNICODE)
        return line
    def parse_train_json(self, line, category):
        """parse doc IDs, titles and description from one line json format string;
           store the infor into ent_train['ID'] = str('Title' + 'Descriptioin')"""
        json_data = re.sub('\r\n', '', line)
        jdata = json.loads(json_data)
        num_results = size(jdata['d']['results'])
        #print "num of results: ", num_results
        if category == self.cat[0]:
            self.num_ent_train += num_results
            for i in xrange(num_results):
                #print i,
                docid = repr(jdata['d']['results'][i]['ID']).strip('\u')
                title = repr(jdata['d']['results'][i]['Title']).strip('\u')
                self.ent_train[docid] = [title]
                description = repr(jdata['d']['results'][i]['Description']).strip('\u')
                self.ent_train[docid].append(description)
                #self.true_train_class[docid]= category
        elif category == self.cat[1]:
            self.num_bus_train += num_results
            for i in xrange(num_results):
                #print i,
                docid = repr(jdata['d']['results'][i]['ID']).strip('\u')
                title = repr(jdata['d']['results'][i]['Title']).strip('\u')
                self.bus_train[docid] = [title]
                description = repr(jdata['d']['results'][i]['Description']).strip('\u')
                self.bus_train[docid].append(description)
                #self.true_train_class[docid]= category
        elif category == self.cat[2]:
            self.num_pol_train += num_results
            for i in xrange(num_results):
                #print i,
                docid = repr(jdata['d']['results'][i]['ID']).strip('\u')
                title = repr(jdata['d']['results'][i]['Title']).strip('\u')
                self.pol_train[docid] = [title]
                description = repr(jdata['d']['results'][i]['Description']).strip('\u')
                self.pol_train[docid].append(description)
                #self.true_train_class[docid]= category
        else:
            print "Can't find the train category: ", category

    def parse_test_json(self, line, category):
        """parse doc IDs, titles and description from one line json format string;
           store the infor into ent_test['ID'] = str('Title' + 'Descriptioin'), bus_test, pol_test"""
        json_data = re.sub('\r\n', '', line)
        jdata = json.loads(json_data)
        num_results = size(jdata['d']['results'])
        #print "num of results: ", num_results
        if category == self.cat[0]:
            self.num_ent_test += num_results
            for i in xrange(num_results):
                #print i,
                docid = repr(jdata['d']['results'][i]['ID']).strip('\u')
                title = repr(jdata['d']['results'][i]['Title']).strip('\u')
                self.ent_test[docid] = [title]
                description = repr(jdata['d']['results'][i]['Description']).strip('\u')
                self.ent_test[docid].append(description)
                self.true_test_class[docid]= category
        elif category == self.cat[1]:
            self.num_bus_test += num_results
            for i in xrange(num_results):
                #print i,
                docid = repr(jdata['d']['results'][i]['ID']).strip('\u')
                title = repr(jdata['d']['results'][i]['Title']).strip('\u')
                self.bus_test[docid] = [title]
                description = repr(jdata['d']['results'][i]['Description']).strip('\u')
                self.bus_test[docid].append(description)
                self.true_test_class[docid]= category
        elif category == self.cat[2]:
            self.num_pol_test += num_results
            for i in xrange(num_results):
                #print i,
                docid = repr(jdata['d']['results'][i]['ID']).strip('\u')
                title = repr(jdata['d']['results'][i]['Title']).strip('\u')
                self.pol_test[docid] = [title]
                description = repr(jdata['d']['results'][i]['Description']).strip('\u')
                self.pol_test[docid].append(description)
                self.true_test_class[docid]= category
        else:
            print "Can't find the test category: ", category
            
    def write_train_file(self):
        """using requests to download json files from Bing API;
           print the json data directly into the class_train_file
        """
        f_ent = open(self.train_ent_file, 'w')
        f_bus = open(self.train_bus_file, 'w')
        f_pol = open(self.train_pol_file, 'w')
              
        request0 = 'https://user:YOUR BING API KEY=@api.datamarket.azure.com/Bing/Search/News?Query=%27'
        request1 = '%27&$format=json&$skip=0'
        request2 = '%27&$format=json&$skip=15'
        ent = '&NewsCategory=%27rt_Entertainment%27'
        bus = "&NewsCategory=%27rt_Business%27"
        pol = "&NewsCategory=%27rt_Politics%27"
 
        for m in self.train_queries:
            for query in m:
                req1 = request0 + query + request1
                req = [req1 + ent, req1 + bus, req1 + pol]
                #print req[0]
                #print req[1]
                #print req[2]
                r1_ent = requests.get(req[0]).json()
                json.dump(r1_ent, f_ent)
                print >> f_ent, "\n"
                r1_bus = requests.get(req[1]).json()
                json.dump(r1_bus, f_bus)
                print >> f_bus, "\n"
                r1_pol = requests.get(req[2]).json()
                json.dump(r1_pol, f_pol)
                print >> f_pol, "\n"

                req2 = request0 + query + request2
                req = [req2 + ent, req2 + bus, req2 + pol]
                r2_ent = requests.get(req[0]).json()
                json.dump(r2_ent, f_ent)
                print >> f_ent, "\n"
                r2_bus = requests.get(req[1]).json()
                json.dump(r2_bus, f_bus)
                print >> f_bus, "\n"
                r2_pol = requests.get(req[2]).json()
                json.dump(r2_pol, f_pol)
                print >> f_pol, "\n"           
        f_ent.close
        f_bus.close
        f_pol.close
    def write_test_file(self):
        """using requests to download json files from Bing API;
           print the json data directly into the class_train_file
        """
        f_ent = open(self.test_ent_file, 'w')
        f_bus = open(self.test_bus_file, 'w')
        f_pol = open(self.test_pol_file, 'w')

        request0 = 'https://user:YOUR BING API KEY=@api.datamarket.azure.com/Bing/Search/News?Query=%27'
        request1 = '%27&$format=json&$skip=0'
        request2 = '%27&$format=json&$skip=15'
        ent = '&NewsCategory=%27rt_Entertainment%27'
        bus = '&NewsCategory=%27rt_Business%27'
        pol = '&NewsCategory=%27rt_Politics%27'
        
        for query in self.test_queries:
            req1 = request0 + query + request1
            req = [req1 + ent, req1 + bus, req1 + pol]
            #print req[0]
            #print req[1]
            #print req[2]
            r1_ent = requests.get(req[0]).json()
            json.dump(r1_ent, f_ent)
            print >> f_ent, "\n"
            r1_bus = requests.get(req[1]).json()
            json.dump(r1_bus, f_bus)
            print >> f_bus, "\n"
            r1_pol = requests.get(req[2]).json()
            json.dump(r1_pol, f_pol)
            print >> f_pol, "\n"

            req2 = request0 + query + request2
            req = [req2 + ent, req2 + bus, req2 + pol]
            r2_ent = requests.get(req[0]).json()
            json.dump(r2_ent, f_ent)
            print >> f_ent, "\n"
            r2_bus = requests.get(req[1]).json()
            json.dump(r2_bus, f_bus)
            print >> f_bus, "\n"
            r2_pol = requests.get(req[2]).json()
            json.dump(r2_pol, f_pol)
            print >> f_pol, "\n"
        f_ent.close
        f_bus.close
        f_pol.close

    def read_train_file(self):
        """open the three train files; parse the json file"""
        f_ent = open(self.train_ent_file, 'r')
        f_bus = open(self.train_bus_file, 'r')
        f_pol = open(self.train_pol_file, 'r')

        lines = f_ent.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if line == '':
                pass
            else:
                self.parse_train_json(line, self.cat[0])

        lines = f_bus.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if line == '':
                pass
            else:
                self.parse_train_json(line, self.cat[1])
        lines = f_pol.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if line == '':
                pass
            else:
                self.parse_train_json(line, self.cat[2])
        f_ent.close
        f_bus.close
        f_pol.close
    def read_test_file(self):
        """open the clusterFile; parse the json file"""
        f_ent = open(self.test_ent_file, 'r')
        f_bus = open(self.test_bus_file, 'r')
        f_pol = open(self.test_pol_file, 'r')

        lines = f_ent.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if line == '':
                pass
            else:
                self.parse_test_json(line, self.cat[0])

        lines = f_bus.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if line == '':
                pass
            else:
                self.parse_test_json(line, self.cat[1])
        lines = f_pol.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if line == '':
                pass
            else:
                self.parse_test_json(line, self.cat[2])
        f_ent.close
        f_bus.close
        f_pol.close       
    def NB_train(self):
        """calculate prior and conditional probability"""
        #prior probabilities:
        self.total_train = self.num_ent_train + self.num_bus_train +self.num_pol_train
        self.prob.append(self.num_ent_train / float(self.total_train))
        self.prob.append(self.num_bus_train / float(self.total_train))
        self.prob.append(self.num_pol_train / float(self.total_train))
        #print "NB_train: self.prob", self.prob

        #ent_train:
        ent = ''
        bus = ''
        pol = ''
        
        #concatenate text of all docs in entertainment
        for docid in self.ent_train.keys():
            #print "NB_train:",docid, self.ent_train[docid]
            ent += self.ent_train[docid][0] + self.ent_train[docid][1]
        for docid in self.bus_train.keys():
            bus += self.bus_train[docid][0] + self.bus_train[docid][1]
        for docid in self.pol_train.keys():
            pol += self.pol_train[docid][0] + self.pol_train[docid][1]
        #number of words in each class
        ent_words = self.getTerms(ent)
        num_ent_words = len(ent_words)
        bus_words = self.getTerms(bus)
        num_bus_words = len(bus_words)
        pol_words = self.getTerms(pol)
        num_pol_words = len(pol_words)
        #print "NB_train: number of words in each class:",num_ent_words,num_bus_words,num_pol_words
        #print "NB_train: words in each class:",ent_words,bus_words,pol_words
        #count term frequency in the training set in thress categories
        for word in ent_words:
            if word in self.tf:
                self.tf[word][0] += 1
            else:
                self.tf[word] = [1, 0, 0]
                self.v.append(word)
        for word in bus_words:
            if word in self.tf:
                self.tf[word][1] += 1
            else:
                self.tf[word] = [0, 1, 0]
                self.v.append(word)
        for word in pol_words:
            if word in self.tf:
                self.tf[word][2] += 1
            else:
                self.tf[word] = [0, 0, 1]
                self.v.append(word)
        num_total_terms = len(self.v)     
        #compute condprob
        for word in self.v:
            self.condprob[word] = [0, 0, 0]
            self.condprob[word][0] = (self.tf[word][0] + 1) / float(num_ent_words + num_total_terms)
            self.condprob[word][1] = (self.tf[word][1] + 1) / float(num_bus_words + num_total_terms)
            self.condprob[word][2] = (self.tf[word][2] + 1) / float(num_pol_words + num_total_terms)

    def NB_test(self):
        """assign a class label to each test document"""
        for docid in self.ent_test:
            content = self.ent_test[docid][0] + self.ent_test[docid][1]
            content = self.getTerms(content)
            score = [0, 0, 0]
            for i in xrange(3):
                score[i] = log(self.prob[i])
            for word in content:
                if not word in self.features:   #only use words in the feature list
                    pass
                else:
                    for i in xrange(3):
                        score[i] += log(self.condprob[word][i])
            group = score.index(max(score))
            self.test_class[docid] = self.cat[group]
        for docid in self.bus_test:
            content = self.bus_test[docid][0] + self.bus_test[docid][1]
            content = self.getTerms(content)
            for i in xrange(3):
                score[i] = log(self.prob[i])
            for word in content:
                if not word in self.v:
                    pass
                else:
                    for i in xrange(3):
                        score[i] += log(self.condprob[word][i])
            group = score.index(max(score))
            self.test_class[docid] = self.cat[group]
        count  = 0 
        for docid in self.pol_test:
            content = self.pol_test[docid][0] + self.pol_test[docid][1]
            content = self.getTerms(content)
            for i in xrange(3):
                score[i] = log(self.prob[i])
            for word in content:
                if not word in self.v:
                    pass
                else:
                    for i in xrange(3):
                        score[i] += log(self.condprob[word][i])
            group = score.index(max(score))
            self.test_class[docid] = self.cat[group]
            if count < 10:
                pass
                #print docid, self.test_class[docid]
                #print docid, self.true_test_class[docid]
    def microF1(self):
        """calulate microF1; print the confusion table and TPs, FNs, FPs"""
        results = [[0,0,0],[0,0,0],[0,0,0]] # the confusing matrix for all categories
        #rows are actual classes; columns are predicted classes
        for docid in self.test_class:
            row = 3
            col = 3
            for i in xrange(3):
                if self.true_test_class[docid] == self.cat[i]:
                    row = i
                if self.test_class[docid] == self.cat[i]:
                    col = i
            if row < 3 and col < 3:
                results[row][col] += 1
            else:
                print "microF1: docid not found: ", docid, "row: ", "column: ", col
        TP = [0,0,0]
        FN = [0,0,0]
        FP = [0,0,0]
        for i in xrange(3):
            TP[i] = results[i][i]
            for j in xrange(3):
                if j == i:
                    pass
                else:
                    FN[i] += results[i][j]
                    FP[i] += results[j][i]
        total_TP = sum(TP)
        total_FP = sum(FP)
        total_FN = sum(FN)
        P = total_TP / float(total_TP + total_FP)
        R = total_TP / float(total_TP + total_FN)
        F1 = 2 * P * R / float(P + R)
        #print the results matrix
        print "-------------------the actual class (row) vs. the predicted class (column)---------------"
        for i in xrange(3):
            print self.cat[i],"\t",
            for j in xrange(3):
                print results[i][j],"\t",
            print "\n"
        print "----------------TP, FP, FN----------------"
        for i in xrange(3):
            print self.cat[i],"\t",TP[i], FP[i], FN[i]
        print "the mircoaveraged F1 is: ", F1
            
        return F1

    def print_test_class(self):
        print "------------------NB classification results----------------"
        print "------------------------ENTERTAINMENT:---------------------"
        for docid in self.test_class:
            if self.test_class[docid] == self.cat[0]:
                print self.true_test_class[docid],"\t",
                for i in xrange(3):
                    if self.true_test_class[docid] == self.cat[i]:
                        if i == 0:
                            print self.ent_test[docid][0]
                        elif i == 1:
                            print self.bus_test[docid][0]
                        elif i == 2:
                            print self.pol_test[docid][0]
                        break
        print "\n-------------------------------BUSINESS:-----------------------"
        for docid in self.test_class:
            if self.test_class[docid] == self.cat[1]:
                print self.true_test_class[docid],"\t",
                for i in xrange(3):
                    if self.true_test_class[docid] == self.cat[i]:
                        if i == 0:
                            print self.ent_test[docid][0]
                        elif i == 1:
                            print self.bus_test[docid][0]
                        elif i == 2:
                            print self.pol_test[docid][0]
                        break
            
        print "\n------------------------------POLITICS:---------------------------"
        for docid in self.test_class:
            if self.test_class[docid] == self.cat[2]:
                print self.true_test_class[docid],"\t",
                for i in xrange(3):
                    if self.true_test_class[docid] == self.cat[i]:
                        if i == 0:
                            print self.ent_test[docid][0]
                        elif i == 1:
                            print self.bus_test[docid][0]
                        elif i == 2:
                            print self.pol_test[docid][0]
                        break
    def calculate_df(self):
        """count how many documents a term appear in each classes;
            return df[term] = [num of docs in cat[0], num of docs in cat[1], num of docs in cat[2])
        """
        for docid in self.ent_train:            
            content = self.ent_train[docid][0] + self.ent_train[docid][1]
            content = self.getTerms(content)
            content = list(set(content))
            for word in content:
                if word in self.df:
                    self.df[word][0] += 1
                else:
                    self.df[word] = [1, 0, 0]
        for docid in self.bus_train:
            content = self.bus_train[docid][0] + self.bus_train[docid][1]
            content = self.getTerms(content)
            content = list(set(content)) #get rid of duplicate words
            for word in content:
                if word in self.df:
                    self.df[word][1] += 1
                else:
                    self.df[word] = [0, 1, 0]
        for docid in self.pol_train:
            content = self.pol_train[docid][0] + self.pol_train[docid][1]
            content = self.getTerms(content)
            content = list(set(content))
            for word in content:
                if word in self.df:
                    self.df[word][2] += 1
                else:
                    self.df[word] = [0, 0, 1]
                    
    def chi_square(self):
        """ compute chi square scores for each term in each class"""
        for docid in self.ent_train:
            content = self.ent_train[docid][0] + self.ent_train[docid][1]
            content = self.getTerms(content)
            content = list(set(content))
            for term in content:
                if not term in self.score['ent']:
                    n11 = float(self.df[term][0])
                    n10 = float(self.df[term][1] + self.df[term][2])
                    n01 = float(self.num_ent_train - n11)
                    n00 = float(self.num_bus_train + self.num_pol_train)
                    a = n11 + n10 + n01 + n00
                    b = math.pow(((n11 * n00) - (n10 * n01)), 2)
                    c = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)
                    chi = (a * b) / c
                    self.score['ent'][term] = chi
        for docid in self.bus_train:
            content = self.bus_train[docid][0] + self.bus_train[docid][1]
            content = self.getTerms(content)
            content = list(set(content))
            for term in content:
                if not term in self.score['bus']:
                    n11 = float(self.df[term][1])
                    n10 = float(self.df[term][0] + self.df[term][2])
                    n01 = float(self.num_bus_train - n11)
                    n00 = float(self.num_ent_train + self.num_pol_train)
                    a = n11 + n10 + n01 + n00
                    b = math.pow(((n11 * n00) - (n10 * n01)), 2)
                    c = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)
                    chi = (a * b) / c
                    self.score['bus'][term] = chi                
        for docid in self.pol_train:
            content = self.pol_train[docid][0] + self.pol_train[docid][1]
            content = self.getTerms(content)
            content = list(set(content))
            for term in content:
                if not term in self.score['pol']:
                    n11 = float(self.df[term][2])
                    n10 = float(self.df[term][1] + self.df[term][0])
                    n01 = float(self.num_pol_train - n11)
                    n00 = float(self.num_ent_train + self.num_pol_train)
                    a = n11 + n10 + n01 + n00
                    b = math.pow(((n11 * n00) - (n10 * n01)), 2)
                    c = (n11 + n01) * (n11 + n10) * (n10 + n00) * (n01 + n00)
                    chi = (a * b) / c
                    self.score['pol'][term] = chi            
             
    def feature_list(self, k):
        """return the feature list with top k terms in each classes; total <= 3*k"""
        #self.score['ent'], self.score['bus'], self.score['pol']; self.features = []
        count = 0
        for key, value in sorted(self.score['ent'].iteritems(), key=lambda (k,v): (v,k)):
            count += 1
            if count < k:
                if not key in self.features:
                    self.features.append(key)
        count = 0
        for key, value in sorted(self.score['bus'].iteritems(), key=lambda (k,v): (v,k)):
            count += 1
            if count < k:
                if not key in self.features:
                    self.features.append(key)        
        count = 0
        for key, value in sorted(self.score['pol'].iteritems(), key=lambda (k,v): (v,k)):
            count += 1
            if count < k:
                if not key in self.features:
                    self.features.append(key)        
        #print "length of feature_list:", len(self.features)
        print "number of features: ", k
        
def main():
    c1 = classification()
    #c1.write_train_files()
    #c1.write_test_files()
    c1.read_train_file()
    c1.read_test_file()
    #print c1.num_ent_train,c1.num_bus_train,c1.num_bus_train
    #print c1.num_ent_test,c1.num_bus_test,c1.num_bus_test
    c1.NB_train()
    c1.calculate_df()
    c1.chi_square()
    c1.feature_list(10)
    c1.NB_test()
    c1.microF1()
    c1.print_test_class()
    
    

if __name__ == '__main__':
    main()
