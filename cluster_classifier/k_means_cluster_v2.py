#-------------------------------------------------------------------------------
# Name:        TAMU csce 670 hw3 part1
# Purpose:  work with Bing API; K-means cluster
#
# Author:      Xixu Cai
#
# Created:     30/03/2013
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
import requests # Get from https://github.com/kennethreitz/requests
import string
import random

class cluster:
    def __init__(self):
        self.totalDocs = 0   # number of total documents for tf-idf calculation
        self.queries = ['texas aggies', 'texas longhorns', 'duke blue devils', 'dallas cowboys', 'dallas mavericks']
        self.totalTerms = [] #total terms in the corpus
        self.docIndex = defaultdict(list)   #docIndex[docID] = [title, description]
        self.actualCluster = defaultdict(list)   #acturalCluster[docid] = query
        self.true_cluster = defaultdict(list) #true_cluster[query] = [docid1, docid2,...]
        self.results = [] #save results for the same k but different sets of random seeds
        self.cluster = defaultdict(list)  #cluster computed by k-means (one result); cluster[int] = [docID1, docID2, ...]
        self.tfIndex = defaultdict(dict)   #tfIndex[term][docID] = int
        self.dfIndex = defaultdict(int)   #dfIndex[word] = int
        self.doc_words = defaultdict(list)  #doc_words[docID] = [term1, term2, ...]easy to look for terms in a given doc for tf lookup
        self.purity = 0.0
        self.ri = 0.0
        self.docVector = defaultdict(array) 
        self.clusterFile = 'clusterFile'   #the original json file downloaded from the internet

    def getTerms(self, line):
        """helper function; parse and get terms from a string"""
        line = line.lower()
        line = re.split(r'[\W]+', line, 0, re.UNICODE)
        return line
    def parse_json(self, line, query):
        """parse doc IDs, titles and description from one line json format string;
           store the infor into docIndex['ID'] = str('Title' + 'Descriptioin')"""
        json_data = re.sub('\r\n', '', line)
        jdata = json.loads(json_data)
        #jdata = json.loads(line)
        num_results = size(jdata['d']['results'])
        self.totalDocs += num_results
        #print "num of results: ", num_results
        for i in xrange(num_results):
            #print i,
            docid = repr(jdata['d']['results'][i]['ID']).strip('\u')
            title = repr(jdata['d']['results'][i]['Title']).strip('\u')
            self.docIndex[docid] = [title]
            description = repr(jdata['d']['results'][i]['Description']).strip('\u')
            self.docIndex[docid].append(description)
            self.true_cluster[docid] = query
            self.actualCluster[query].append(docid)
      
    def write_cluster_file(self):
        """using requests to download json files from Bing API;
           print the json data directly into the clusterFile;
           parse json data and store into dictionary docIndex:
           docIndex['ID'] = str('Title' + 'Description')"""
        f = open(self.clusterFile, 'w')
              
        request1 = 'https://user:YOUR BING API KEY=@api.datamarket.azure.com/Bing/Search/News?Query=%27'
        request2 = '%27&$format=json&$skip=0'
        request3 = '%27&$format=json&$skip=15'
 
        for i in xrange(5):
            req1 = request1 + string.replace(self.queries[i], ' ', '%20') + request2
            r1 = requests.get(req1).json()
            print req1
            json.dump(r1, f)
            print >> f, "\n"
       
            req2 = request1 + string.replace(self.queries[i], ' ', '%20') + request3
            r2 = requests.get(req2).json()
            print req2
            json.dump(r2, f)
            print >> f, "\n"
        f.close
    def read_cluster_file(self):
        """open the clusterFile; parse the json file"""
        f = open(self.clusterFile, 'r')
        lines = f.readlines()
        #jdata = json.loads(f)
        count = 0
        for line in lines:
            line = re.sub('\n', '', line)
            if line == '':
                pass #ignore empty lines
            else:
                i = count / 2
                self.parse_json(line, self.queries[i])
                count += 1
        f.close
        
    def tf_idf(self, tf, df, total):
        """given two ints: tf and df, return the tf_idf value"""
        return ((1+log2(tf))*log2(total/float(df)))
    def build_tf_idf(self):
        for docID, contents in self.docIndex.iteritems():
            text = contents[0] + contents[1]
            words = self.getTerms(text)
            for word in words:
                try:
                    self.tfIndex[word][docID] += 1
                except:
                    self.tfIndex[word][docID] = 1
                    self.dfIndex[word] += 1
                    if not word in self.totalTerms:
                        self.totalTerms.append(word)                    
                if not word in self.doc_words[docID]:
                    try:
                        self.doc_words[docID].append(word)
                    except:
                        self.doc_words[docID] = [word]
        
    def cosine(self, v1, v2):
        """given two vectors, compute their cosine similarity"""
        mag1 = 0.0
        mag2 = 0.0
        dotp = 0.0
        for i in xrange(len(v1)):
            dotp += v1[i] * v2[i]
            mag1 += math.pow(v1[i],2)
            mag2 += math.pow(v2[i], 2)
        mag1 = sqrt(mag1)
        mag2 = sqrt(mag2)
        d = dotp / (mag1 * mag2)    
        return d
    def vector_space(self):
        """construct vectors (same length) for all documents in the corpus"""
        for docID in self.doc_words:
            vector = []
            for term in self.totalTerms:
                if term in self.doc_words[docID]:
                    vector.append(self.tf_idf(self.tfIndex[term][docID], self.dfIndex[term], self.totalDocs))
                else:
                    vector.append(0)
            mag = 0
            for i in vector:
                mag += math.pow(i, 2)
            mag = sqrt(mag)
            self.docVector[docID] = array(vector)/mag
    def k_means(self, k):
        """input an integer k; compute k-means clustering method with k;
           when members in the old cluster and the new cluster are the same,converge;
           or when the iteration reaches 50, converge (maxiter = 50)
           return RSS
        """
        maxiter = 50  
        #instead of random seeds, using select_init_centroids(k)
        
        centroids = self.select_init_centroids(k)

        flag = True
        iteration = 0
        #only when the old and new cluster have the same centroids or iteration == maxiter, flag = False
        while(flag):
            #print "k-means: while(flag), iteration:",iteration
            for i in xrange(k):
                self.cluster[i] = []
            # compute clusters for the centroids        
            for docID in self.docVector:
                maxCosine = 0
                group = k + 1
                for i in xrange(k):
                
                    x = self.cosine(centroids[i], self.docVector[docID])
                    if x > maxCosine:
                        maxCosine = x
                        group = i
                self.cluster[group].append(docID)
            #recompute centroids
            old_centroids = []
            for i in xrange(k):
                old_centroids.append(centroids[i])
            for i in xrange(k):
                size = len(self.cluster[i])
                #print "k-means, while(flag): size:", size
                v = []
                for m in self.cluster[i]:
                    if v == []:
                        for cell in self.docVector[m]:
                            v.append(cell)
                    else:
                        v += self.docVector[m]
                #print "k-means:v: ", v
                
                #centroids[i] = array(v) / size
                centroids[i] = array(v)
                #nomalize new centroids as well
                mag = 0
                for j in centroids[i]:
                    mag += j * j
                mag = sqrt(mag)
                centroids[i] = centroids[i] / mag
                                
            #compute if the old and new centroids are the same
            for i in xrange(k):
                for x, y in zip(old_centroids[i], centroids[i]):
                    flag = (x != y)
                    #print "k-means: x, y: ", x, y
                    if flag:
                        break
                #flag = (old_centroids[i].all() != centroids[i].all())  #equal: flag = False; not equal: flag = True
                if flag: #any pair of centroids are not equal, break and recompute
                    break
            if iteration >= maxiter:
                flag = False
            iteration += 1
        #cluster computation finishes; time to compute RSS
        #print "k-means iteration is: ",iteration
        RSS = 0
        for i in xrange(k):
            RSSK = 0
            for docid in self.cluster[i]: #m is individual vector 
                for n in xrange(len(self.docVector[docid])):                        
                    RSSK += math.pow((centroids[i][n] - self.docVector[docid][n]),2)
            RSS += RSSK
        print "k-means k, RSS: ", k, RSS
        return RSS
    def k_means_results(self, k):
        """for a given k, run k_means for 5 times, return the cluster with the smallest RSS"""
        iteration = 5
        RSS_list = []
        results = []
        for i in xrange(iteration):
            print "K-means iteration: ", i
            rss = self.k_means(k)
            RSS_list.append(rss)
            #append the self.actualCluster to self.results
            results.append(self.cluster)
        j = RSS_list.index(min(RSS_list))
        
        self.cluster = results[j]
        print "k, RSS\t",k, RSS_list[j],
        return RSS_list[j]

    def cluster_evaluation(self):
        """compute purity and RI for self.cluster and self.actualCluster"""
        #need a matrix with k rows and 5 columns to store the results
        #for each cluster, find how many docs belong to each class, add the max's up, then /total for purity"""
        #for each cluster, find how many docs in class 1 to k, TP = sum(C(same class >2 in one cell, 2))
        #TP + FP = sum C(total in each cluaster, 2)
        #TN = sum(same class in cluster i * same class in cluster j, i != j)
        results = []
        for group in self.cluster.keys():
            #print "cluster_evaluation: group number: ", group
            if self.cluster[group] == []:
                print "the cluster is empty!"
                break
            r = []
            c = set(self.cluster[group])
            for query in self.queries:
                r.append(len(c & set(self.actualCluster[query])))
            results.append(r)
        #purity, TP, FP, TN, FN calculation
        totalMax = 0
        total = 0
        TPFP = 0
        TP = 0
        TN = 0
        FN = 0
        numRows = len(results)
        curRow = 0
        for row in results:
            #print "curRow:", curRow
            #print "row: ", row
            totalMax += max(row)
            rowTotal = 0
            for i in xrange(len(row)):
                rowTotal += row[i]
                nextRow = curRow + 1
                while nextRow < numRows:
                    FN += results[curRow][i] * results[nextRow][i]
                    nextRow += 1
                if results[curRow][i] > 2:
                    TP += results[curRow][i] * (results[curRow][i] - 1) / 2
            total += rowTotal
            TPFP += rowTotal * (rowTotal - 1) / 2
            curRow += 1
        self.purity = totalMax / float(total)
        FP = TPFP - TP
        TN = total * (total - 1) / 2 - TPFP - FN
        #print "totalMax:", totalMax, "total", total, "TP:", TP, "TPFP:", TPFP, "FP:", TPFP-TP, "FN:", FN, "TN:", TN
        self.ri = (TP + TN) / float(total * (total - 1) / 2)
        print "\tpurity:\t", self.purity,"\tri:\t", self.ri
    def print_cluster(self):
        """print the cluster results in the folloiwng format:
            cluster 1:
            "query 1": "title"
            "query 2": "title"
            ...
        """
        for x in self.cluster.keys():
            print "\nCluster ", x+1, ":"
            for docid in self.cluster[x]:
                print self.true_cluster[docid],":\t", self.docIndex[docid][0]

    def normalize(self, arr):
        """normalize a vector (list or array type); return a numpy.ndarray"""
        arr = array(arr)
        mag = 0
        for j in arr:
            mag += j * j
        mag = sqrt(mag)
        arr = arr / mag
        return arr
 
    def select_init_centroids(self, k):
        """find seed clusters with int(alpha * totalDocs / k) data points;
            compute and return their means as the initial centroids;
            see paper by Fang Yuan, etc. "A New Algorithm to Get The Initial Centroids";
            instead of using euclidine distance, I used cosine for computing distance and centroids.
        """
        print "start selecting the initial centroids..."
        alpha = 0.5
        num_vecs = int (alpha * self.totalDocs / k)
        dist = defaultdict(float) #key is a tuple (docid1, docid2)
        corpus = self.docIndex.keys()
        centroids = []
        for docid in corpus:
            corpus = [x for x in corpus if x != docid]
            for docid2 in corpus:
                dist[(docid, docid2)] = self.cosine(self.docVector[docid], self.docVector[docid2])
        total_pairs = dist.keys()
        #print "select_init_centroids: distance calculation finishes!"
        for i in xrange(k):
            #print "select_init: i:", i
            centroid = []
            seed_cluster = []
            pair = max(dist.iteritems(),key = lambda (k,v): (v, k))#pair is a tuple, the key
            #need to copy all distance involved pair[0] and pair[1] to the new distance list;
            #and remove from the old distance list;
            #and remove pair[0] and pair[1] from corpus too
            for doc in pair[0]:
                if centroid == []:
                    for dim in self.docVector[doc]:
                        centroid.append(dim)
                else:
                        centroid += self.docVector[doc]
            for x in pair[0]:
                seed_cluster.append(x)
            new_dist = defaultdict(float)

            for m in total_pairs:
                if (pair[0][0] in m) or (pair[0][1] in m):
                    if (pair[0][0] in m) and (pair[0][1] in m):
                        del dist[m]
                        total_pairs.remove(m)
                    else:
                        new_dist[m] = dist[m]
                        del dist[m]
                        total_pairs.remove(m)
            for j in xrange(num_vecs - 2):
                #print "select_init: j:", j
                new_total_pairs = new_dist.keys()
                new_pair = max(new_dist.iteritems(),key = lambda (k,v): (v, k))
                while((new_pair[0][0] in seed_cluster) and (new_pair[0][1] in seed_cluster)):
                    del new_dist[new_pair[0]]
                    new_pair = max(new_dist.iteritems(),key = lambda (k,v): (v, k))
                new = ''
                for v in new_pair[0]:
                    if not v in seed_cluster:
                        seed_cluster.append(v)
                        new = v
                        break
                centroid += self.docVector[new]
                for m in new_total_pairs:
                    if new in m:
                        del new_dist[m]
                        new_total_pairs.remove(m)
                for m in total_pairs:
                    if new in m:
                        new_dist[m] = dist[m]
                        del dist[m]
                        total_pairs.remove(m)
            #normalize centroid for new centroid
            centroids.append(self.normalize(centroid))
        #print "select_init_seeds: "
        #print centroids
        return centroids
	
            
def main():
    c1 = cluster()
#    c1.write_cluster_file()
    c1.read_cluster_file()
    
    c1.build_tf_idf()
    c1.vector_space()
    print "start k-means:"
    c1.k_means(9)
    c1.cluster_evaluation()
    c1.print_cluster()
    
    

if __name__ == '__main__':
    main()
