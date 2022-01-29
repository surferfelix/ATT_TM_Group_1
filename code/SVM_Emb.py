from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import pandas as pd
import sys
import csv 
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec
import os
import numpy as np

#These are files we created, can be cloned from https://github.com/surferfelix/ATT_TM_Group_1
import utils
import feature_extraction as fe

def create_classifier(train_features, train_targets, modelname, embedded = False):
   
    if modelname == 'SVM':
        model = LinearSVC(max_iter = 10000)
        
    if embedded == True:
        vec = 0
        model.fit(train_features, train_targets)
    else:
        vec = DictVectorizer()
        features_vectorized = vec.fit_transform(train_features)
        model.fit(features_vectorized, train_targets)
   
    return model, vec

def classify_data(model,vec,inputdata,outputfile,language_model,embedded =False):
    features,gold = extract_features(inputdata, language_model= language_model, vectorizer = vec)

    predictions = model.predict(features)
    outfile = open(outputfile, 'w')
    counter = 0
    for index, line in enumerate(open(inputdata, 'r')):
        if index == 0:
            continue
        if len(line.rstrip('\n').split()) > 0:
            try:
                outfile.write(line.rstrip('\n') + '\t' + predictions[counter] + '\n')
                counter += 1
            except IndexError:
                print('Alignment seems to go wrong at line:', counter)
                break
    outfile.close()

def extract_features(inputfile, language_model, vectorizer = False):
    prev_token = ""
    data = []
    tokens, gold, chapters, sent_id = fe.fileread(inputfile)
    feature_dict = fe.featuretraindict(tokens,gold,language_model,baseline = False, w_embedding = False)
    tokens = feature_dict['Tokens']
    lemmas = feature_dict['Lemmas']
    pos_tags = feature_dict['POS']
    neg_words =  feature_dict['Neg_Word']
    aff_negs = feature_dict['Affixal_Neg']
    word_bigrams = feature_dict['Word_bigrams']
    prev_tokens = feature_dict['Prev_Token']
    next_tokens = feature_dict['Next_Token']\
        
    golds = gold.tolist()
    
    for token,lemma,pos_tag,neg_word,aff_neg,word_bigram,prev_token,next_token,gold in zip(tokens,lemmas,pos_tags,neg_words,aff_negs,word_bigrams,prev_tokens,next_tokens,golds):
        feature_dict = {"Tokens": token, 'lemmas':lemma,'pos_tags':pos_tag,'neg_word':neg_word,'aff_neg': aff_neg,'prev_token':prev_token,'next_token':next_token}
        data.append(feature_dict)

    tok_vectors = fe.featuretraindict(tokens,gold,language_model,baseline = False, w_embedding = True)["Tokens"]
    if not vectorizer:
        vectorizer = create_vectorizer_traditional_features(data)
        assignment = True
    elif vectorizer:
        assignment = False
    sparse_features = vectorizer.transform(data)
    combined_vectors = combine_sparse_and_dense_vectors(tok_vectors, sparse_features)
    if assignment == True:
        return combined_vectors, golds, vectorizer
    elif assignment == False:
        return combined_vectors, golds
    
def create_vectorizer_traditional_features(nor_features):
    """Subfunction for create_all_classifier"""
    vectorizer = DictVectorizer()
    vectorizer.fit(nor_features)
    return vectorizer

def combine_sparse_and_dense_vectors(emb_features, sparse_feature_reps):
    """Subfunction for create_all_classifier"""
    combined_vec = []
    sparse_vec = np.array(sparse_feature_reps.toarray())
    for index, vec in enumerate(sparse_vec):
        combined_vector = np.concatenate((vec, emb_features[index]))
        combined_vec.append(combined_vector)
    return combined_vec

def main(argv=None):
   
    if argv is None:
        argv = sys.argv
        
    argv = ['mypython_program','data/SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt','data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt','models/glove.42B.300d.txt']
    trainingfile = argv[1]
    inputfile = argv[2]
    model = argv[3]
    
    tmp_file = 'models/temp_glove_as_word2vec.txt'
    if not os.path.isfile(tmp_file): #Checking if it exists so it only needs to convert once, saving time on second run
        glove2word2vec(model, tmp_file) # Converting glove to w2v
    language_model = gensim.models.KeyedVectors.load_word2vec_format(tmp_file) # Loading new w2v
    
    embedded = True
    training_features, gold_labels, vectorizer = extract_features(trainingfile,language_model, vectorizer = False)
    for modelname in ['SVM']:
        ml_model, vec = create_classifier(training_features, gold_labels, modelname, embedded)
        classify_data(ml_model, vectorizer, inputfile, inputfile.replace('.txt','.' + modelname + '_all_f_emb.conll'),language_model, False)
    
if __name__ == '__main__':
    main()