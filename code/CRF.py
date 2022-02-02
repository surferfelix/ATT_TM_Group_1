import pandas as pd
import numpy as np
import nltk 
import sys
import csv
from nltk.stem import WordNetLemmatizer 

# get_ipython().system("pip install 'sklearn<0.24'")
# get_ipython().system("pip install 'sklearn_crfsuite<0.24'")


## import CRF 
import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import metrics

import utils
import feature_extraction as fe


def token2features(sent: tuple, i: int) -> dict:
    '''Takes a sent as input and generates featuredict for the tokens in that sent
    :param sent: the tuple with features for each token
    :param i: typically used to iterate over the features in sent, essentially an indexing object
    :return: featuredict for the sent
    '''
    neg_list = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non'] #Chowdhury list

    tokens = [sent[i][0] for i, sen in enumerate(sent)]
    gold = [sent[i][-1] for i, sen in enumerate(sent)]
    pos_tags = utils.POS(tokens)
    lemmas = utils.lemma_extraction(tokens, pos_tags)
    neg_word = utils.neg_word(tokens, neg_list)
    aff_neg = utils.affixal_neg(tokens)
    prev_token, next_token = utils.prev_next_tokens(tokens)
    features = {"Tokens": tokens[i], "Lemmas": lemmas[i], "POS": pos_tags[i], "Neg_Word": neg_word[i], "Affixal_Neg": aff_neg[i], "Prev_Token": prev_token[i], "Next_Token": next_token[i]}
    baseline = {"Tokens": tokens[i]}
        
    return features


def sent2features(sent: tuple) -> list:
    '''Will get collect the features for tokens
    :param sent: the tuple with features for each token'''
    return [token2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    '''Will get collect the labels for tokens
    :param sent: the tuple with features for each token'''
    #if you added features to your input file, make sure to add them here as well.
    return [gold for token,pos_tags, lemmas, neg_word, aff_neg, prev_token, next_token, gold in sent]

def sent2tokens(sent):
    '''Will get collect the features for tokens
    :param sent: the tuple with features for each token'''
    return [token for token,pos_tags, lemmas, neg_word, aff_neg, prev_token, next_token, gold in sent]


def extract_sents_from_conll(inputfile: str) -> list: 
    '''This function extracts the features from inputfile and returns them in a format
    suited to sklearn_CRFsuite
    :param inputfile: The filepath
    :return: a list in list of tuples, where the content of the tuple includes the features per token
    and where the content of the inner list encapsulates the sentences'''
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



def train_crf_model(X_train: list, y_train:list) -> sklearn_crfsuite.CRF:
    '''Will fit the X_train parameter; the features, with the y_train parameter; the gold labels.
    :type X_train: a list in list of tuples, where the content of the tuple includes the features per token
    and where the content of the inner list encapsulates the sentences'''
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    crf.fit(X_train, y_train)
    
    return crf


def create_crf_model(trainingfile: str) -> sklearn_crfsuite.CRF:
    '''This function returns a trained model based on extracted features from the trainingdata
    :param trainingfile: The file you wish to use to extract features from and train the model
    :return: a sklearn_crfsuite.CRF object'''
    train_sents = extract_sents_from_conll(trainingfile)
    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    crf = train_crf_model(X_train, y_train)
    
    return crf

def run_crf_model(crf, evaluationfile: str)-> tuple:
    '''This function will use a trained model to make predictions and returns a tuple of predictions and test output. 
    :param crf: a fitted model that we want to make predictions for
    :param evaluationfile: a filepath containing a file that you wish to make predictions for'''
    test_sents = extract_sents_from_conll(evaluationfile)
    X_test = [sent2features(s) for s in test_sents]
    y_pred = crf.predict(X_test)
    
    return y_pred, X_test



def write_out_evaluation(eval_data: list, pred_labels: list, outputfile: str):
    '''This function attempts to write a file with model predictions'''
    outfile = open(outputfile, 'w')
    for evalsents, predsents in zip(eval_data, pred_labels):
        for data, pred in zip(evalsents, predsents):
            outfile.write(data.get('Tokens') + "\t" + pred + "\n")



def train_and_run_crf_model(trainingfile: str, evaluationfile: str, outputfile: str):
    '''Will load the filepaths and create a crf model, fit the trainingfile, and transform the evaluationfile
    to make predictions
    :param trainingfile: file to fit data to
    :param evaluationfile: file to make predictions for
    :param outputfile: the path where you wish to store the predictions
    :return: a tsv file'''
    crf = create_crf_model(trainingfile)
    pred_labels, eval_data = run_crf_model(crf, evaluationfile)
    write_out_evaluation(eval_data, pred_labels, outputfile)


def main():
    training_data = '../data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt'
    dev_data = '../data/combined_test_set.txt' # Can also be test set
    outputfile = '../data/results_crf.conll'
    args = ['my_python',training_data,dev_data,outputfile]
    trainingfile = args[1]
    evaluationfile = args[2]
    outputfile = args[3]
    
    train_and_run_crf_model(trainingfile, evaluationfile, outputfile)
    

if __name__ == '__main__':
    
    main()

