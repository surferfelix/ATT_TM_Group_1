import pandas as pd
import numpy as np
import nltk 
import sys
import csv
from nltk.stem import WordNetLemmatizer 

get_ipython().system("pip install 'sklearn<0.24'")
get_ipython().system("pip install 'sklearn_crfsuite<0.24'")


## import CRF 
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import metrics

import utils
import feature_extraction as fe


training_data = 'data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt'
dev_data = 'data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt' # Remove /content/ when merging to py file


# Reading the files
tr_tokens, tr_gold, tr_chapters, tr_sent_id = fe.fileread(training_data)
te_tokens, te_gold, te_chapters, te_sent_id = fe.fileread(dev_data)

## Here debug = True just means that I am not yet loading embeddings, therefore embedding_model param is empty string
train_features = fe.featuretraindict(tr_tokens, tr_gold, '', baseline = False, w_embedding = True) # Dict
train_baseline = fe.featuretraindict(tr_tokens, tr_gold, '', baseline = True, w_embedding = True) # Dict

dev_features = fe.featuretraindict(te_tokens, te_gold, '', baseline = False, w_embedding = True) # Dict
dev_baseline = fe.featuretraindict(te_tokens, te_gold, '', baseline = True, w_embedding = True) # Dict


sentences = [sent for sent in zip(tr_tokens,tr_gold,tr_sent_id)]
#print(sentences) 


def token2features(sentences, i):
    '''Sentences is a list of sents, i represents the index'''

    neg_list = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non'] #Chowdhury list

    tokens = [sentences[i][0] for i, sent in enumerate(sentences)]
    gold = [sentences[i][-1] for i, sent in enumerate(sentences)]
    pos_tags = utils.POS(tokens)
    lemmas = utils.lemma_extraction(tokens, pos_tags)
    neg_word = utils.neg_word(tokens, neg_list)
    aff_neg = utils.affixal_neg(tokens)
    prev_token, next_token = utils.prev_next_tokens(tokens)
    features = {"Tokens": tokens[i], "Lemmas": lemmas[i], "POS": pos_tags[i], "Neg_Word": neg_word[i], "Affixal_Neg": aff_neg[i], "Prev_Token": prev_token[i], "Next_Token": next_token[i]}
    baseline = {"Tokens": tokens[i]}
        
    return features



def sent2features(sent):
    return [token2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    #if you added features to your input file, make sure to add them here as well.
    return [gold for token,pos_tags, lemmas, neg_word, aff_neg, prev_token, next_token, gold in sent]

def sent2tokens(sent):
    return [token for token,pos_tags, lemmas, neg_word, aff_neg, prev_token, next_token, gold in sent]


def extract_sents_from_conll(inputfile): # It gets 5 items from this func
    #Extract the tokens2features and convert them to sents
    sents = [] # List of lists
    current_sent = [] # Gets added to sent 
    tokens, gold, chapters, sent_id = fe.fileread(inputfile)
    sentcount = 0
    n = 0 
    sentences = []
    neg_list = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non'] #Chowdhury

    chapter = list(dict.fromkeys(chapters)) # This method preserves order
    sent_id = sent_id.tolist()
    chapters = chapters.tolist()

    # Our features we are using 
    tokens = tokens.tolist()
    gold = gold.tolist()
    pos_tags = utils.POS(tokens)
    lemmas = utils.lemma_extraction(tokens, pos_tags)
    neg_words = utils.neg_word(tokens, neg_list)
    aff_negs = utils.affixal_neg(tokens)
    prev_tokens, next_tokens = utils.prev_next_tokens(tokens)
    tokenholder = [] # Temporary memory for tokens

    for index,(sent, token, chap) in enumerate(zip(sent_id, tokens, chapters)):
        ### REDEFINING A ROW
        token = tokens[index]
        pos_tag = pos_tags[index]
        lemma = lemmas[index]
        neg_word = neg_words[index]
        aff_neg = aff_negs[index]
        prev_token = prev_tokens[index]
        next_token = next_tokens[index]
        gold_label = gold[index]
        ###
        if chap == chapter[n]: # Are we still on current chapter?
            if sent != sentcount: # For changing sentences
                sentcount += 1
                sentences.append(list(tokenholder))
                tokenholder = []
                row = [token, pos_tag, lemma, neg_word, aff_neg, prev_token, next_token, gold_label]
                tokenholder.append(tuple(row))
            else: 
                row = [token, pos_tag, lemma, neg_word, aff_neg, prev_token, next_token, gold_label]
                tokenholder.append(tuple(row))
        elif chap != chapter[n]:
            # If the book name is different
            n+=1 #Changing the chapter 
            # Resetting the sentence count
            sentcount = 0
    if sent == sent_id[-1]: # Condition for last sentence
        sentences.append(list(tokenholder))

    return sentences



def train_crf_model(X_train, y_train):

    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    return crf


def create_crf_model(trainingfile):

    train_sents = extract_sents_from_conll(trainingfile)
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    crf = train_crf_model(X_train, y_train)
    
    return crf



def run_crf_model(crf, evaluationfile):

    test_sents = extract_sents_from_conll(evaluationfile)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = crf.predict(X_test)
    
    return y_pred, X_test



def write_out_evaluation(eval_data, pred_labels, outputfile):

    outfile = open(outputfile, 'w')
    
    for evalsents, predsents in zip(eval_data, pred_labels):
        for data, pred in zip(evalsents, predsents):
            outfile.write(data.get('Tokens') + "\t" + pred + "\n")



def train_and_run_crf_model(trainingfile, evaluationfile, outputfile):
    crf = create_crf_model(trainingfile)
    pred_labels, eval_data = run_crf_model(crf, evaluationfile)
    write_out_evaluation(eval_data, pred_labels, outputfile)



def main():

    args = ['my_python','data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt','data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt','data/results_crf.conll']
    trainingfile = args[1]
    evaluationfile = args[2]
    outputfile = args[3]
    
    train_and_run_crf_model(trainingfile, evaluationfile, outputfile)
    

if __name__ == '__main__':
    main()

