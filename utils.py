import nltk
from nltk.stem import WordNetLemmatizer
import argparse
import pandas as pd

# NLTK

def wordnet_pos(pos: list) -> str:
    """
    Helper function for lemmas: converts nltk POS_tags to WordNet_tags
    :param pos: PSO_tags generated with nltk POS
    :return: WordNet_tag
    """
    # code inspiration: https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python, https://www.holisticseo.digital/python-seo/nltk/lemmatize
    if pos.startswith('J'):
        return 'a'
    elif pos.startswith('V'):
        return 'v'
    elif pos.startswith('N'):
        return 'n'
    elif pos.startswith('R'):
        return 'r'
    else:
        return 'n'

def lemma_extraction(tokens: list, pos_list: list) -> list: 
    """
    Function which creats lemmas using tokens and POS-tags(converts nltk tags to wordnet tags)
    :param tokens: list of tokens
    :param pos_list: list of POS_tags
    :return: list of lemmas 
    """
    lemmas = []
    lemmatizer = WordNetLemmatizer()

    for token, pos in zip(tokens, pos_list):
        lemma = lemmatizer.lemmatize(token, wordnet_pos(pos))
        lemmas.append(lemma)
        
    return lemmas
        
def POS(tokens:list) -> list:
    '''
    Function which creates list of POS-tags of tokens
    :param tokens: list containing the tokens to get POS for
    :return: list of POS-tags
    '''
    pos_list = []
    POS = nltk.pos_tag(tokens)
    for token, pos_tag in POS:
        pos_list.append(pos_tag)   
        
    return pos_list

def word_ngram(tokens: list, n: int) -> list:
    '''
    This function generates n word ngrams
    :param tokens: list of tokens
    :param n: number of tokens for n_gram
    :return: all possible ngrams
    '''
    # code inspiration: https://towardsdatascience.com/from-dataframe-to-n-grams-e34e29df3460

    word_grams= pd.Series(nltk.ngrams(tokens, n, pad_right = True))
    
    return word_grams
    
def neg_word(tokens: list, neg_list: list)-> list:
    '''
    This functions checks tokens against a list of negation words
    :param tokens: list of tokens
    :param neg_list: negation word list
    :return: list of negation words
    '''

    neg_word_list = []

    for token in tokens:

        # label 1 if token is in negative word list
        if token in neg_list:
            label = 1

        # label 0 if token is not in negative word list
        else:
            label = 0
        
        neg_word_list.append(label)

    return neg_word_list
    
    

def affixal_neg(tokens: list)-> list: 
    '''
    This function generates a list of negation affixes if they are found as a prefix or suffix of a token entry
    :param tokens: list of tokens to search through
    :return: list of aff_neg_labels
    '''
    label_list = []
    for token in tokens:

        # ends with -less
        if len(token) >4 and token.endswith('less'):
            label = 'less'

        # starts with
        elif len(token) >4 and token.startswith('un') and not token.startswith("under"):
            label = 'un'
        
        # negations with im followed by m or p 
        elif len(token) >4 and token.startswith('im') and (token[2] == 'm' or 'p'):
            label = 'im' 

        # looks up all tokens with 3character negation suffixes 
        ## using ill and irr (instead of il and ir) because most negated words in 
        ### english are followed by a second l or r
        elif len(token) >4 and token.startswith(('non','dis', 'ill', 'irr')):
            label = token[0:3]
        
        elif len(token) >6 and token.startswith('anti'):
            label = 'anti'
        
       # none of the above
        label_list.append(label)

    return label_list
    

