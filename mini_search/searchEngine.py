#-------------------------------------------------------------------------------
# Name:        search engine
# Purpose: build a mini search engine for binary, phrase and wildcard search;
#          key: inverted index, permute index
#
# Author:      Xixu Cai
#
#-------------------------------------------------------------------------------
#!/usr/bin/python
import sys
import glob
import os
import re
import string
import gc
import array
import copy
from collections import defaultdict

def build_dict():
    index = defaultdict(list)
    os.chdir(".")
#    i = 1
    for file in glob.glob("*.txt"):
        f = open(file, 'r')
        lines = f.read()
        words = re.findall(r"[0-9a-zA-Z]+", lines)
#        words = re.findall(r"[\w]+", lines)      #return a list
#        words = re.sub(r'[^a-z0-9]', ' ', lines)
        file_num = int(re.sub("\D", "", file))
#        pos = 1
        pagedict={}
        for position, word in enumerate(words):
            wd = word.lower()
            try:
                pagedict[wd][1].append(position)
            except:
                pagedict[wd]=[file_num, [position]]

        #merge the current page index with the main index
        for termpage, postingpage in pagedict.iteritems():
            if termpage in index:
                index[termpage].append(postingpage)
            else:
                index[termpage] = [postingpage]
#        i += 1
#        if i == 2:
#            print dict

    return index

#the following code is partially borrowed from Arden Dertat's online code
def writeIndexToFile(index):
    '''write the inverted index to the file'''
    f=open("./indexFile", 'w')
    for term in index.iterkeys():
        postinglist=[]
        for p in index[term]:
            docID=p[0]
            positions=p[1]
            postinglist.append(':'.join([str(docID) ,','.join(map(str,positions))]))
        print >> f, ''.join((term,'|',';'.join(postinglist)))

    f.close()
def readIndex():
    index = defaultdict(list)
    f=open("./indexFile", 'r');
    for line in f:
        line=line.rstrip()
        term, postings = line.split('|')    #term='docID', postings='docID1:pos1,pos2;docID2:pos1,pos2'
        postings=postings.split(';')        #postings=['docId1:pos1,pos2','docID2:pos1,pos2']
        postings=[x.split(':') for x in postings] #postings=[['docId1', 'pos1,pos2'], ['docID2', 'pos1,pos2']]
        postings=[ [int(x[0]), map(int, x[1].split(','))] for x in postings ]   #final postings list  
        index[term]=postings
    f.close()
    return index
def intersectLists(lists):
    if len(lists) == 0:
        return []
    lists.sort(key=len)
    return list(reduce(lambda x,y: set(x) & set (y), lists))
def getTerms(line):
    line = line.lower()
    line = re.sub(r'[^a-z0-9]', ' ', line)
    line = line.split()
    return line
def getPostings(terms, index):
    return [index[term] for term in terms]
def getDocsFromPostings(postings):
    return [ [x[0] for x in p] for p in postings ]

def one_word_query(q, index):
    origin = q
    q = getTerms(q)
    if len(q) == 0:
        print 'sorry, no input query found'
        return []
    elif len(q)>1:
        binary(origin, index)
        return []
    query = q[0]
    if query not in index:
#        print 'sorry, query is not in the documents'
        return
    else:
        p = index[query]
        p = [x[0] for x in p]
        
#        print ' '.join(map(str, p))
    return p

def binary(q, index):
    """more than one word binary AND search"""
    q = getTerms(q)
    if len(q) == 0:
        print 'sorry, no input query found'
        return
    li = []
    for term in q:
        try:
            p = index[term]
            p = [x[0] for x in p]
#            print 'P:', p
            li.append(p)
#            print 'li: ', li
        except:
            print 'sorry, no match :('
            return
    li = intersectLists(li)
    if li == []:
#        print 'sorry, no match :('
        return []
    li.sort()
#    print ' '.join(map(str, li))
    return li

def phrase(q, index):
    """pharse search"""
    origin = q
    query = getTerms(q)
    if len(query) == 0:
        print 'sorry, no input query found'
        return []
    elif len(q) == 1:
        p = one_word_query(origin)
        return p

    phraseDocs = []
    length = len(query)

    for term in query:
        if term not in index:
            print 'sorry, no match :('
            return []

    postings = getPostings(query,index)
    docs = getDocsFromPostings(postings)
    docs = intersectLists(docs)
#    print xrange(len(postings))

    for i in xrange(len(postings)):
        postings[i] = [x for x in postings[i] if x[0] in docs]

    postings = copy.deepcopy(postings)

    for i in xrange(len(postings)):
        for j in xrange(len(postings[i])):
            postings[i][j][1]=[x-i for x in postings[i][j][1]]

    result = []
    for i in xrange(len(postings[0])):
        li = intersectLists([x[i][1] for x in postings])
        if li == []:
            continue
        else:
            result.append(postings[0][i][0])
    
    result.sort()
#    if result == []:
#        print "sorry, no match"
#    else:
#        print ' '.join(map(str, result))

    return result


def pindex(index):
    pdict = {}
    for term in index.iterkeys():
        new_term = term + '$'
#        print 1, new_term
        pdict[new_term] = term
        while new_term[0] != r'$':
            new_term = new_term[1:] + new_term[0]
            pdict[new_term] = term
#            print 1, new_term
    return pdict

def writePindexToFile(index):
    '''write the inverted index to the file'''
    f=open("./pindex", 'w')
    for term in index.iterkeys():
        print >> f, ''.join((term,'|',index[term]))
    f.close()

def wildcard(words, index, pdict):
    words = words.lower()
#    print words
    words = words.split()
    result = []
    for word in words:
        word_result=[]
#        print "word:", word
        new_word = word.split(r'*')
#        print "new_word",new_word
        target = new_word[1] + '$' + new_word[0]
        for term in pdict.iterkeys():
            term_result=[]
            if term[:len(target)] == target:
#                print "Term: ",term
#                print "pdict[term]", pdict[term]
#                print "index[]", index[pdict[term]]
                
                for p in index[pdict[term]]:
 
                    try: 
                        term_result.append(p[0])
                    except:
                        term_result = [p[0]]
#                print "term_result:", term_result
            try:
                word_result.extend(term_result)
            except:
                word_result = term_result
#        print "word_result: ", word_result
        if result == []:
            result = [word_result]
        else:
            result.append(word_result)
#    print "result before intersect:", result
    if result == []:
        print "sorry no match for the wildcard search"
    else:
        result = intersectLists(result)
        result.sort()
 #   print "final result: ", result
    return result

def user_input(index, pdict):
    var = raw_input("query: ")
#    print "original var:", var
    var = var.lower()
    phrase_re = re.compile('\"(.*?)\"', re.IGNORECASE)
    
    phrase_terms = phrase_re.findall(var)
#    print "phrase_terms:", phrase_terms
    if phrase_terms != []:
        for q in phrase_terms:
            try:
                phrase_result.append(phrase(q, index))
            except:
                phrase_result = [phrase(q, index)]
#        print "user_input: phrase_result: ", phrase_result
        phrase_result = intersectLists(phrase_result)
    var = re.sub('\".*?\"','', var, flags = re.IGNORECASE)
    var = var.strip(' ')
#    print "var after re.sub: ", var
    words = var.split()
    w_query = ''
    b_query = ''
    for word in words:
        if '*' in word:
            w_query = w_query + ' ' + word
        else:
            b_query = b_query + ' ' + word
#    print "b_query:", b_query
#    print "w_query:", w_query
    if w_query != '':
        w_result = wildcard(w_query, index, pdict)
    if b_query != '':
        b_result = binary(b_query, index)
    result = []
    if phrase_terms != []:
        result.append(phrase_result)
    if w_query != '':
        result.append(w_result)
    if b_query != '':
        result.append(b_result)

#    print "user_input: result:", result
    if result == [] or result == [None] or result == [[]]:
        print "sorry, no match :("
        return 
    else:
        final_result = intersectLists(result)
        if final_result == []:
            print "sorry, no match:("
        else:
            print ' '.join(map(str, final_result))
        return final_result
        
  
def main():
    index = build_dict()
    print "dictionary is done"
    writeIndexToFile(index)
#    index = readIndex()
    pdict = pindex(index)
    writePindexToFile(pdict)
    print "pindex is done"
    
    while True:
        user_input(index, pdict)





if __name__ == '__main__':
    main()
