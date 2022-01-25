import nltk
import argparse
import pandas as pd
from nltk.tokenize import sent_tokenize
import utils
import os
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec # We need to convert the glove to w2v first since gensim can only load w2v

def fileread(inputfile: str) -> tuple: 
    '''Opens and reads the file
    :param inputfile: filepath to read as tsv
    :type inputfile: string
    return: lists of column content )tokens, gold, chapters, sent_id
    '''
    # Here we use iloc to select the row since there are no headers
    df = pd.read_csv(inputfile, encoding='utf-8', delimiter = "\t", quotechar = '|')
    chapters = df.iloc[:, 0]
    sent_id = df.iloc[:, 1]
    token_position = df.iloc[:, 2]
    tokens = df.iloc[:, 3]
    gold = df.iloc[:, 4]
    return tokens, gold, chapters, sent_id

def tokens_to_embeddings(tokens: list, word_embedding_model) -> list:
    '''Checks if there is embedding representation for token to use this
    as a feature, otherwise returns 300 empty dimensions.
    :param tokens: list of tokens
    :param word_embedding_model: the pre-loaded word_embedding_model
    type word_embedding_model: word2vec object
    :return: list embedding_vectors
    '''
    embedding_vectors = []
    for token in tokens:
        if token in word_embedding_model:
            vector = word_embedding_model[token]
        else:
            vector = [0]*300 # Change this number if your model does not use 300d
        embedding_vectors.append(vector)
    return embedding_vectors

def tokens_to_sentences(tokens: list, chapters: list, sent_id: list, sent_tokenizer = None) -> list:
    '''This function takes tokens as input and will join them into sentences
    :param tokens: the tokens variable from the fileread function
    :param chapters: the chapters variable from the fileread function
    :param sent_id: the sent_id variable from the fileread function
    :param sent_tokenizer: can be set to either nltk or None to select how you wish the text to be sentence tokenized'''
    list_of_tokens = [token for token in tokens]
    periods_joined = list(join_punctuation(list_of_tokens))
    complete_text = ' '.join(periods_joined)

    # Sentence Tokenization with NLTK

    if sent_tokenizer == 'nltk':  
        sentences = sent_tokenize(complete_text, language = 'english')

   # Sentence Tokenization through the sentence ID
   
    elif sent_tokenizer == None:
        sentcount = 0
        n = 0
        sentences = []
        chapter = list(dict.fromkeys(chapters)) # This method preserves order
        sent_id = sent_id.tolist()
        tokens = tokens.tolist()
        chapters = chapters.tolist()
        tokenholder = [] # Temporary memory for tokens
        for sent, token, chap in zip(sent_id, tokens, chapters):
            if chap == chapter[n]: # Are we still on current chapter?
                if not sent == sentcount: # For changing sentences
                    sentcount += 1
                    sentences.append(' '.join(tokenholder))
                    tokenholder.clear()
                elif sent == sentcount:
                    tokenholder.append(token)
            if not chap == chapter[n]:
                # If the book name is different
                n+=1 #Changing the chapter 
                # Resetting the sentence count
                sentcount = 0
        if sent == sent_id[-1]: # Condition for last sentence
            sentences.append(' '.join(tokenholder))

    return sentences, complete_text

def join_punctuation(seq: list, characters='.,;?!'):
# From https://stackoverflow.com/questions/15950672/join-split-words-and-punctuation-with-punctuation-in-the-right-place
    '''Helper function for tokens_to_sentences that joins punctuation markers to a token
    :param seq: contains list of tokens
    :type: seq: list
    :param characters: string of characters
    '''
    characters = set(characters)
    seq = iter(seq)
    current = next(seq)

    for nxt in seq:
        if nxt in characters:
            current += nxt
        else:
            yield current
            current = nxt

    yield current

def featuretraindict(tokens: list, gold: list, word_embedding_model, baseline = False, debug = False) -> dict: # All features we want should be inputparameters. 
    '''Adds the to use features in the system to a dictionary
    :param tokens: the tokens variable from fileread function
    :param gold: gold labels from fileread function
    :param word_embedding_model: A loaded word_embedding model
    :param baseline: When set to True this will only use tokens as feature representations
    :type word_embedding_model: word2vec object
    :return: dictionary with features 
    '''

    neg_list = ['nor', 'neither', 'without', 'nobody', 'none', 'nothing', 'never', 'not', 'no', 'nowhere', 'non'] #Chowdhury

    # NLTK
    pos_tags = utils.POS(tokens)
    lemmas = utils.lemma_extraction(tokens, pos_tags)
    neg_word = utils.neg_word(tokens, neg_list)
    word_bigrams = utils.word_ngram(tokens, 2)
    aff_neg = utils.affixal_neg(tokens)
    prev_token, next_token = utils.prev_next_tokens(tokens)

    # Embedding_check

    # Continue implementation when we start vectorising to combine embeddings with one-hot token dimensions
    ## For now this is just to show that we also have embedding representations of tokens ready as a feature
    if debug == False:
        emb_tokens = tokens_to_embeddings(tokens, word_embedding_model) 
    
    # Featuredict
    if baseline == False: 
        features = {"Tokens": tokens, "Lemmas": lemmas, "POS": pos_tags, "Neg_Word": neg_word, "Affixal_Neg": aff_neg, "Word_bigrams": word_bigrams, "Prev_Token": prev_token, "Next_Token": next_token, "Gold": gold}
    else:
        features = {'tokens': tokens,"lemmas": lemmas}
    # Test
    #assert all(len(f) == len(tokens) for f in features.values()), f'\n The features in the featuredict must be of identical length. Current lengths are:\n\n {print((len(f) for f in features.values()))}'
    return features


#### thats what i did for i spirat
def feature_dict_to_csv(feature_dict):
    df= pd.DataFrame(feature_dict)
    df.to_csv('feature_dict.tsv', sep='\t', header = True) 

def token_embeddings_pipe(inputfile, embeddingmodel)-> list:
    '''This can be imported if you want to fetch the token representation of embeddings alone'''
    tokens, gold, chapters, sent_id = fileread(inputfile)
    emblist = tokens_to_embeddings(tokens, embeddingmodel)
    return emblist


# Main Pipeline is in here
def main(inputpath: str, embedding_path: str, debug = False):
    print('Starting up the pipeline...\n')
    tokens, gold, chapters, sent_id = fileread(inputpath)
    print('Loading embedding model...\n')
    glove_file = embedding_path # https://radimrehurek.com/gensim/scripts/glove2word2vec.html
    tmp_file = 'models/temp_glove_as_word2vec.txt'
    if not os.path.isfile(tmp_file): #Checking if it exists so it only needs to convert once, saving time on second run
        glove2word2vec(glove_file, tmp_file)
    if debug == False:
        language_model = KeyedVectors.load_word2vec_format(tmp_file)
    elif debug == True:
        language_model= ''
    print('Converting tokens back to sentences...\n')
    sentences, complete_text = tokens_to_sentences(tokens, chapters, sent_id, sent_tokenizer = None)
    print('Adding the features to the dict\n')
    features = featuretraindict(tokens, gold, language_model, baseline = False, debug = debug)
    print('Create tsv of featuredict\n')
    feature_dict_to_csv(features)
    print('End of current implementation')
    
# Run
if __name__ == "__main__":
    # Please add the correct path here

    debug = True #Are you debugging or not?

    if debug == False:
        word_embedding_path = 'models/glove.42B.300d.txt'
    elif debug == True:
        word_embedding_path = ''
    inputfile = "data/SEM-2012-SharedTask-CD-SCO-dev-simple.txt"
    assert os.path.isfile(inputfile), 'Your path does not seem to be a file'
    main(inputfile, word_embedding_path, debug = debug)

