
#   Copyright 2018 Fraunhofer IAIS
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import re
import numpy as np
import multiprocessing
from scipy import exp
from scipy.linalg import eigh
import codecs
import pickle
import distance
import termcolor
from unidecode import unidecode
import argparse
from nltk import ngrams
import matplotlib.pyplot as plt

def similarity_function_tup(tup):
    a,b = tup
    return similarity_function(a,b)

def normalization(w):
    return unidecode(re.sub(r'[^\w]', ' ', w).lower())

def ngrams_word(s, n):
    string = " "+s+" "
    return list(set([string[i:i+n] for i in range(len(string)-n+1)]))

#This function takes two lists of n-grams
def sorensen_word(s1,s2):     
    N = min(len(s1), len(s2))
    ng1 = [ngrams_word(s1,j) for j in range(2,  min( len(s1)+1 , MAX_NGRAM))  ]
    ng2 = [ngrams_word(s2,j) for j in range(2,  min( len(s2)+1 , MAX_NGRAM))  ]
    return 1 - np.sum(distance.sorensen(ng1[0][i], ng2[0][i]) for i in range(N))/ (N)
    #return 1 - np.sum(distance.sorensen(s1[i], s2[i]) for i in range(N))/ (N)

def similarity_word_ngram(s1,s2):
    ngrams1 = [ngrams_word(s1,j) for j in range(2,  min( len(s1)+1 , MAX_NGRAM))  ]  
    ngrams2 = [ngrams_word(s2,j) for j in range(2,  min( len(s2)+1 , MAX_NGRAM))  ]  
    N = min(len(ngrams1),len(ngrams2))+1        
    ngramMatches = np.array([len(np.intersect1d(ngrams1[i], ngrams2[i])) for i in range(N-1)])
    ngramLens1 = np.array([len(ngrams1[i] ) for i in range(N-1)] )
    ngramLens2 = np.array([len(ngrams2[i] ) for i in range(N-1)] )

    ngramMatchesLen = len(ngramMatches)
    if ngramMatchesLen > 0:
        return sum(2* ngramMatches / (ngramLens1 +ngramLens2) )/ ngramMatchesLen
    else:
        return 0

def init_list_of_objects(size):
    list_of_objects = list()
    for i in range(0,size):
        list_of_objects.append( list() )
    return list_of_objects
    
def similarity_sentence_ngram(s1,s2):
    ng1 = init_list_of_objects(min( len(s1.split())+1 , MAX_NGRAM)-2)
    ng2 = init_list_of_objects(min( len(s2.split())+1 , MAX_NGRAM)-2)
    for j in range(2,  min( len(s1.split())+1 , MAX_NGRAM)):
        for ngram in ngrams(s1.split(), j):
            ng1[j-2].append(ngram)
    for j in range(2,  min( len(s2.split())+1 , MAX_NGRAM)):
        for ngram in ngrams(s2.split(), j):
            ng2[j-2].append(ngram)
    summ = 0
    for j in range(min(min(len(s1.split())+1, len(s2.split())+1),MAX_NGRAM)-2):
        summ += np.sum(distance.sorensen(ng1[j][i], ng2[j][i]) for i in range(min(len(ng1[j]),len(ng2[j]))))/min(len(ng1[j]),len(ng2[j]))
    summ = summ/min(min(len(s1.split())+1, len(s2.split())+1),MAX_NGRAM)
    
    print(summ)
    
    return 1 - summ

def projectWordTup(tup):
    word    = tup[0]
    tuples  = tup[1]
    hyperparam   = tup[2]
    alphas  = tup[3]
    lambdas = tup[4]
    kernel = tup[5]
    
    pair_sim = np.array([similarity_function(word,t) for t in tuples])
    if kernel == "poly":        
        k = (np.ones(len(pair_sim)) - pair_sim)**hyperparam
    else:              
        k = np.exp(-hyperparam * (pair_sim**2))           
    #that was missing?..
    return k.dot(alphas / lambdas)



'''
Parsing user arguments
'''

argParser = argparse.ArgumentParser(description="KPCA embeddings training script", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argParser.add_argument('--repr', type=str,help="Representative words (use a subset of your vocabulary if it is too large for your memory restrictions)", action='store',required=False)
argParser.add_argument('--vocab', type=str,help="Vocabulary path", action='store',required=True)
argParser.add_argument('--kernel', type=str,help="Kernel: 'poly','rbf' (default: 'poly' )", action='store',required=False, default = "RBF")
argParser.add_argument('--hyperparam', type=float,help="Hyperparameter for the selected kernel: sigma for RBF kernel and the degree for the polynomial kernel (default 2)", action='store',required=False, default = 0.3)
argParser.add_argument('--max_ngram', type=int,help="Maximum length of the n-grams considered by the similarity function (default 2)", action='store',required=False, default = 5)
#change
argParser.add_argument('--size', type=int,help="Number of principal components of the embeddings (default 1500)", action='store',required=False, default = 2)
argParser.add_argument('--cores', type=int,help="Number of processes to be started for computation (default: number of available cores)", action='store',required=False, default= multiprocessing.cpu_count())
argParser.add_argument('--output', type=str,help="Output folder for the KPCA embeddings (default: current folder)", action='store',required=False, default= ".")
argParser.add_argument('--wordSentence', type=str,help="Choose if you want to use KPCA on words or sentences (default: words)", action='store',required=False, default= "sentence")
argParser.add_argument('--similarity', type=str,help="Choose which similarity function you'd like to use (default: similarity_ngram)", action='store',required=False, default= similarity_sentence_ngram)

args = argParser.parse_args()


MAX_NGRAM = args.max_ngram
n_components = args.size
reprPath = args.repr
vocabPath = args.vocab
kernel = args.kernel
hyperparam = args.hyperparam
cores = args.cores
outputPath = args.output
wordSentence = args.wordSentence
similarity_function = args.similarity


#Similarity function to be used as dot product for KPCA

if reprPath == None:
    reprPath = vocabPath

'''
Preprocessing
'''


with codecs.open(reprPath, "r") as fIn:
    if wordSentence == "word":
        reprVocab = [  normalization(w[:-1]) for w in fIn if len(w[:-1].split()) == 1]
    else:
        reprVocab = [  normalization(w.rstrip()) for w in fIn.readlines()]        
        reprVocab = reprVocab[:-1]     

termcolor.cprint("Generating element pairs\n", "blue")
reprVocabLen = len(reprVocab)

pairsArray = np.array([ (t1,t2) for t1 in reprVocab for t2 in reprVocab])

pool = multiprocessing.Pool(processes=cores)

'''
Similarity matrix computation: the similarity of all word pairs from the representative words is computed
'''
termcolor.cprint("Computing similarity matrix\n", "blue")

simMatrix = np.array(pool.map(similarity_function_tup, pairsArray )).reshape(reprVocabLen, reprVocabLen)

pairsArray = None


'''
Kernel Principal Component Analysis
'''
termcolor.cprint("Solving eigevector/eigenvalues problem\n", "blue")

if kernel == "rbf":    
    K = exp(-hyperparam * (simMatrix**2))
else: #poly
    distMatrix = np.ones(len(simMatrix))- simMatrix
    K = distMatrix**hyperparam


# Centering the symmetric NxN kernel matrix.
N = K.shape[0]
one_n = np.ones((N,N)) / N
K_norm = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)
# Obtaining eigenvalues in descending order with corresponding eigenvectors from the symmetric matrix.    
eigvals, eigvecs = eigh(K_norm)

alphas = np.column_stack((eigvecs[:,-i] for i in range(1,n_components+1)))
lambdas = [eigvals[-i] for i in range(1,n_components+1)]


pickle.dump( alphas, open( outputPath+"/alphas_{}_{}_{}_{}_{}.p".format(similarity_function.__name__, len(reprVocab),kernel, hyperparam, n_components), "wb" ) )
pickle.dump( lambdas, open( outputPath+"/lambdas_freq_vocab_nl_{}_{}_{}_{}_{}.p".format(similarity_function.__name__, len(reprVocab),kernel, hyperparam, n_components), "wb" ) )     


'''
Projection to KPCA embeddings of the vocabulary
'''
with codecs.open(vocabPath, "r") as fIn:
    if wordSentence == "word":
        vocab = [ normalization(w[:-1]) for w in fIn if len(w[:-1].split()) ==1]
    else:
        vocab = [  normalization(w.rstrip()) for w in fIn.readlines()]        
        vocab = vocab[:-1]     

termcolor.cprint("Projecting known vocabulary to KPCA embeddings\n", "blue")

X_train = pool.map(projectWordTup, [(word,reprVocab, hyperparam,alphas, lambdas, kernel) for word in vocab] )  

X_train = np.asarray(X_train)


fig = plt.figure(figsize=(10,10)) 
ax = fig.add_subplot(111) 

ax.plot(X_train[:,0], X_train[:,1], 'go') 

for i, label in enumerate(reprVocab): 
   plt.text (X_train[:,0][i], X_train[:,1][i], label ) 

plt.show() 

#plt.figure(figsize=(20,20))
#plt.subplots_adjust(bottom = 0.1)
#plt.scatter(
#    X_train[:,0], X_train[:,1], marker='o', cmap=plt.get_cmap('Spectral'))

#for label, x, y in zip(vocab, X_train[:,0], X_train[:,1]):
#    plt.annotate(
#        label,
#        xy=(x, y), xytext=(-10, 10),
#        textcoords='offset points', ha='left', va='bottom',
#        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
#        arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

#plt.show()

pickle.dump( X_train, open(outputPath+"/KPCA_{}_{}_{}_{}_{}.p".format(similarity_function.__name__, len(reprVocab),kernel,hyperparam, n_components), "wb" ) )

