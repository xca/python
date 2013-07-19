#----------------------------------------------------------------
#Name: tweet ranking
#Purpose: tf-idf, Vector Space Retrieval & PageRank
#Author: Xixu Cai
#----------------------------------------------------------------
#!/usr/bin/python
import sys
import os
import re
import json
import array
from collections import defaultdict
import numpy as np
import math
import time

def comp1(x, y):
    if(y[1]>x[1]):
        return 1
    else:
        return -1
def comp0(x, y):
    return int(y[0]-x[0])
def getTerms(line):
    line = line.lower()
    line = re.split(r'[\W]+', line, 0, re.UNICODE)
    return line

def build_tf_df (filename, tf_index, d_index, df_index, doc_user, user_doc):
#    f = file("mars_tweets_medium.json", "r")
    f = file(filename, "r")
    lines = f.readlines()
    i = 0
    for line in lines:
        json_data = re.sub('\r\n', '', line)
        data = json.loads(json_data)
        words = getTerms(data["text"])
#        words = re.split(r'[\W]+', data["text"], 0, re.UNICODE)
        docid = data["id"]
        userid = data["user"]["id"]
        doc_user[docid] = userid
        try:
            user_doc[userid].append(docid)
        except:
            user_doc[userid] = [docid]

#        if i < 10:
#            print "doc_user: ", docid, doc_user[docid]
        for word in words:
            try:
                tf_index[word][docid] += 1
            except:
                tf_index[word][docid] = 1
                df_index[word] += 1
            #document index
            if not word in d_index[docid]:
                try:
                    d_index[docid].append(word)
                except:
                    d_index[docid] = [word]
        i += 1
    return

def query_doc(query, tf_index):
    """return a list with doc_ids"""
#    print "query_doc() starts: query is: ", query
    qwords = getTerms(query)
#    print "query_doc() qwords:", qwords
    if qwords == []:
        print "query_doc(): no query found!"
        return []
    results = []
    q_exist = []
    for word in qwords:
        if not word in tf_index:
            print "query_doc(): ", word, "not in the tweet corpus"
            return []
        else:
            if word in q_exist:
                print "query_doc(): q_exist: ", q_exist
                pass
            else:
                for term in tf_index[word]:
#                    print "query_doc(): term in tf_index[word]", term
                    results.append(term)
    return list(set(results))

def tf_idf(tf, df, total):
    """given two ints: tf and df, return the tf_idf value"""
    return ((1+np.log2(tf))*np.log2(total/float(df)))

def d_vector(query, docid, tf_index, d_index, df_index):
    """given a word and a document id, return a tf-idf vector"""
    total = len(d_index)
    mag = 0
#    d_vector = []
    for word in d_index[docid]:
        mag += (tf_idf(tf_index[word][docid], df_index[word], total)) * (tf_idf(tf_index[word][docid], df_index[word], total))
# normalization
    mag = math.sqrt(mag)
    results = []
    q_words = getTerms(query)
# build a vector same length as a query
    for word in q_words:
        if word in d_index[docid]:
            results.append( ( tf_idf(tf_index[word][docid], df_index[word], total) ) / mag)
        else:
            results.append(0)
#    print "d_vector() results: ", results
    return results

def q_vector(query, tf_index, df_index, d_index):
    total = len(d_index)
    qwords = getTerms(query)
#    print "q_vector", qwords
    q_vector = []
    qtf = {}
    mag = 0
    #calculate the tf for each word in query
    for word in qwords:
        try:
            qtf[word] += 1
        except:
            qtf[word] = 1
    for word in qwords:
#        print "q_vector for word in qwords:", word, tf_idf(qtf[word], df_index[word], total)
        mag += (tf_idf(qtf[word], df_index[word], total)) * (tf_idf(qtf[word], df_index[word], total))
#        q_vector.append(tf_idf(qtf[word], df_index[word], total))
#    print "q_vector:", q_vector
#    magnitude = np.linalg.norm(q_vector)
    mag = math.sqrt(mag)
    for word in qwords:
        q_vector.append((tf_idf(qtf[word], df_index[word], total))/mag)
#    print "q_vector:  ", q_vector, "q_vector finishes"
    return q_vector

def v_search(query, tf_index, df_index, d_index):
#    print "v_search(): query:", query
    doc_list = query_doc(query, tf_index)
#    print "v_search(): doc_list: ", doc_list
    if doc_list == []:
        print "v_search: No documents found!"
        return []
    query_vec = q_vector(query, tf_index, df_index, d_index)
    v_results = []
    for doc in doc_list:
        doc_vec = d_vector(query, doc, tf_index, d_index, df_index)
        score = np.dot(doc_vec, query_vec)
#        print "v_search(): score: ", score
        v_results.append([doc, score])
#    v_results.sort(cmp=comp1)
    return v_results

def print_top50_vsearch(v_results):
    if v_results == []:
        print "********************caution: vector search results**************************"
        print "vectors search found no tweets! Possibly because one of your query words is not in the corpus!"
        return
    v_results.sort(cmp=comp1)
    print "********************vector search results**************************"
    print "tweet id\tsimilarity score"
#    print "len(v_results)", len(v_results)
    if len(v_results) < 50:
#        print "inside if"
        for x in v_results:
            print  x[0], "\t",x[1]
    else:
        i = 0
        while i < 50:
            print  v_results[i][0], "\t", v_results[i][1]
            i += 1
    return

def build_graph(filename, in_dict, out_dict, count_dict, user_name):
#    f = file("mars_tweets_medium.json", "r")
    f = file(filename, "r")
    lines = f.readlines()
    i = 0
    for line in lines:
        json_data = re.sub('\r\n', '', line)
        data = json.loads(json_data)
        user_id = data["user"]["id"]
        user_name[user_id] = data["user"]["screen_name"]
        if data["entities"]["user_mentions"] == []:
            pass
        else:
            for x in data["entities"]["user_mentions"]:
                if x["id"] == user_id or x["id"] in in_dict[user_id]:
        #mention herself or the mentioned guy has already in her list
                    pass
                else:
                    if not x["id"] in user_name:
                        user_name[x["id"]] = x["screen_name"]
                    try:
                        in_dict[user_id].append(x["id"])
                    except:
                        in_dict[user_id] = x["id"]
    #visit in_dict and build out_dict and count_dict
    for term in in_dict:
        count_dict[term] = len(in_dict[term])
        for user in in_dict[term]:
            try:
                out_dict[user].append(term)
            except:
                out_dict[user] = [term]
    for term in in_dict:
#        if i < 10:
#            print "build_graph: in_dict: ", term, in_dict[term]
#            print "build_graph: out_dict: ", term, out_dict[term]
#            print "build_graph: count_dict: ", term, count_dict[term]
#            print "build_graph: user_name: ", term, user_name[term]
            i += 1
    return

def compute_PR(userPR, out_dict, count_dict, user_name):
    """computer userPR and print the the first top ranked users"""
    #first assign the same PR to everyone's old PR.
    for user, neighbors in out_dict.iteritems():
        userPR[user] = [1, 0]
        for x in neighbors:
            if not x in userPR:
                userPR[x] = [1, 0]
    alpha = 0.9
    total = len(out_dict)
    teleprob = (1 - alpha) / total
    converge = 0.0000001
    # the modified do while loop until converge
    # first calculation
    error = 0
    for user in userPR:
        for neighbor in out_dict[user]:
            userPR[user][1] += alpha * userPR[neighbor][0] / count_dict[neighbor]
        userPR[user][1] += teleprob
    #then reset the new and old PR values
    for user in userPR:
        error += math.fabs(userPR[user][0] - userPR[user][1])
        userPR[user][0] = userPR[user][1]
        userPR[user][1] = 0
#    print "comp_PR: first round error: ", error
    i = 1
    while (error > converge):
        i += 1
        error = 0
        sum = 0
        for user in userPR:
            for neighbor in out_dict[user]:
                userPR[user][1] += alpha * userPR[neighbor][0] / count_dict[neighbor]
            userPR[user][1] += teleprob

        for user in userPR:
            error += math.fabs(userPR[user][0] - userPR[user][1])
#            if error < math.fabs(userPR[user][0] - userPR[user][1]):
#                error = math.fabs(userPR[user][0] - userPR[user][1])
            userPR[user][0] = userPR[user][1]
            sum += userPR[user][1]
            userPR[user][1] = 0
    #after convergency, normalization
    for user in userPR:
        userPR[user][0] = userPR[user][0] / sum
#    print "comp_PR: userPR[316801033]: ", userPR[316801033]
#    print "comp_PR: iter round: ", i
#    print "comp_PR: userPR", userPR
    return userPR

def print_top50userPR(userPR, user_name):
    sorted_userPR = sorted(userPR.iteritems(), key = lambda val:val[1][0], reverse = True)
    print "***************************************page rank results***********************************"
    print "user screen name\tuser id\tuser PR"
    for item in sorted_userPR[:50]:
        print user_name[item[0]], item[0], item[1][0]

def PR_vector_log(userPR, v_results, doc_user):
    """combine PageRank and vector space search; return a list with [docid, score] pairs"""
#    print "PR_vector: begins"
    i = 0
    comb_scores = []
#    print v_results
    if v_results == []:
#        print "vectore search has no results!  therefore combined search with log has no results either!"
        return []
    v_results.sort(cmp=comp1)
    w = 0.5
    alpha = 10
    beta = 1000
    for pair in v_results:
#pair[0] is the docid and pair[1] is the vector cosine value
#score = w * pair[1] + (1 - w) * pair[1] / log(alpha * userPR[doc_user[docid]]+alpha)
        docid = pair[0]
        user_id = doc_user[docid]
        pr = 0
        if user_id in userPR:
            pr = userPR[user_id][0]
        score = w * pair[1] + ( (1 - w) * pair[1] / np.log2(beta * pr + alpha) )
#        if i < 10:
#            print "PR_vector:", docid, pair[1],  pr, np.log2(alpha * pr + alpha)
        try:
            comb_scores.append([docid, score])
        except:
            comb_scores = [docid, score]
#    comb_scores.sort(cmp=comp1)
        i += 1
    return comb_scores
def PR_vector_linear(userPR, v_results, doc_user):
    """combine PageRank and vector space search with linear combination;
    return a list with [docid, score] pairs"""
    i = 0
    comb_scores = []
    if v_results == []:
#        print "vectore search has no results!  therefore combined search with log has no results either!"
        return []
    v_results.sort(cmp=comp1)
    w = 0.9
    beta = 100
    for pair in v_results:
#pair[0] is the docid and pair[1] is the vector cosine value

        docid = pair[0]
        user_id = doc_user[docid]
        pr = 0
        if user_id in userPR:
            pr = userPR[user_id][0]
        score = w * pair[1] + (1 - w) * pr
#        if i < 10:
#            print "PR_vector:", docid, pair[1],  pr, np.log2(alpha * pr + alpha)
        try:
            comb_scores.append([docid, score])
        except:
            comb_scores = [docid, score]
#    comb_scores.sort(cmp=comp1)
        i += 1
    return comb_scores
def print_suggestion(userPR, user_doc):
    sorted_userPR = sorted(userPR.iteritems(), key = lambda val:val[1][0], reverse = True)
    print "*************************Here are some recommended tweet ids for you****************************"
    print "user id\t\tsuggested tweet ids"
    for item in sorted_userPR[:10]:
        print item[0], "\t",
        for x in user_doc[item[0]]:
            print x,
        print


def print_top50comb_log(comb_scores, userPR, user_doc):
    if comb_scores == []:
        print "*************************caution**************************************"
        print "Sorry, combined search with log returns no results! possibly because the query has word not in the copus!"
        print_suggestion(userPR, user_doc)
        return
    comb_scores.sort(cmp=comp1)
    print "********************combined results with log**************************"
    print "tweet id\tsimilarity score"
#    print "len(v_results)", len(v_results)
    if len(comb_scores) < 50:
#        print "inside if"
        for x in comb_scores:
            print  x[0], "\t",x[1]
    else:
        i = 0
        while i < 50:
            print  comb_scores[i][0], "\t", comb_scores[i][1]
            i += 1
    return
def print_top50comb_linear(comb_scores, userPR, user_doc):
    if comb_scores == []:
        print "*************************caution**************************************"
        print "Sorry, combined search with linear returns no results! possibly because the query has word not in the copus!"
        print_suggestion(userPR, user_doc)
        return
    comb_scores.sort(cmp=comp1)
    print "********************combined results with linear**************************"
    print "tweet id\tsimilarity score"
#    print "len(v_results)", len(v_results)
    if len(comb_scores) < 50:
#        print "inside if"
        for x in comb_scores:
            print  x[0], "\t",x[1]
    else:
        i = 0
        while i < 50:
            print  comb_scores[i][0], "\t", comb_scores[i][1]
            i += 1
    return

def main():
    filename = raw_input("filename: ")
    tf_index = defaultdict(dict)
    d_index = defaultdict(list)
    df_index = defaultdict(int)
    doc_user = {}
    user_doc = defaultdict(list)

    build_tf_df(filename, tf_index, d_index, df_index, doc_user, user_doc)

    in_dict = defaultdict(list)
    out_dict = defaultdict(list)
    count_dict = defaultdict(int)
    user_name = {}
    userPR = defaultdict(list)

    build_graph(filename, in_dict, out_dict, count_dict, user_name)
    userPR = compute_PR(userPR, out_dict, count_dict, user_name)

    while (True):
        print "\n************************New Search*******************************\n"
        query = raw_input("query: ")

        v_results = v_search(query, tf_index, df_index, d_index)
        log_scores = PR_vector_log(userPR, v_results, doc_user)
        linear_scores = PR_vector_linear(userPR, v_results, doc_user)
        print_top50_vsearch(v_results)
        print_top50userPR(userPR, user_name)
        print_top50comb_log(log_scores, userPR, user_doc)
        print_top50comb_linear(linear_scores, userPR, user_doc)

if __name__ == '__main__':
    main()
