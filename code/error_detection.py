import pandas as pd
import argparse


def fileread(inputfile):
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
    pred = df.iloc[:, -1]
    return tokens, gold, chapters, sent_id, pred


def error_detect(tokens, gold, chapters, sent_id, pred):
    '''
    This function compares gold and predicted label.
    :param tokens, gold, chapters, sent_id and pred: lists extracted from inputfiel
    :return: dictionary with error occurrences    
    '''

    error_dict = []
        
    # for each instance if prediction not gold add token, prediction and gold to dict
    for token, gol, chap,sent, pre in zip(tokens, gold,chapters, sent_id, pred):
        if gol != pre:
            error_sent = {"chap": chap, "sent_id": sent, "Token": token, "Gold": gol, "Prediction": pre}
            error_dict.append(error_sent)

    return error_dict  


def dict_to_csv(error_dict, outputfile):
    '''
    This Function writes out error dicitonary to tsv file
    :param error_dict: dictionary with errors
    :param outputfile: name of outputfile
    '''
    df= pd.DataFrame(error_dict)
    df.to_csv(outputfile+'.tsv', sep='\t', header = True) 
    
    
def main():

    args = ['my_python','resultsfile','outputfile']
    resultsfile = args[1]
    outputfile = args[2]
    
    tokens, gold, chapters, sent_id, pred = fileread(resultsfile)
    error_dict = error_detect(tokens, gold, chapters, sent_id, pred)
    dict_to_csv(error_dict, outputfile) 
    
if __name__ == '__main__':
main() 