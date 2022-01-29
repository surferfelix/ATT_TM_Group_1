import pandas as pd
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, precision_recall_fscore_support

def extract_annotations(inputfile, annotationcolumn, delimiter='\t'):
    '''
    This function extracts annotations
    :param inputfile: the path to the file
    :param annotationcolumn: the name of the column in which the target annotation is provided
    :param delimiter: optional parameter to overwrite the default delimiter (tab)
    :returns: the annotations as a list
    '''
    header_svm = ["Chapter", "Sentence Number", "Token Number", "Token", "Gold", "Prediction"]
    header_crf = ["Token", "Prediction"] 
    header_gold = ["Chapter", "Sentence Number", "Token Number", "Token", "Gold"]
    
    if "SVM" in inputfile:
        conll_input = pd.read_csv(inputfile, sep=delimiter, names = header_svm)
    elif "crf" in inputfile:
        conll_input = pd.read_csv(inputfile, sep=delimiter, names = header_crf)
    else:
        conll_input = pd.read_csv(inputfile, sep=delimiter, quotechar = "\t", names = header_gold) 
    annotations = conll_input[annotationcolumn].tolist()
    return annotations


def gold_annotations(dev_file):
    """
    This function extracts the gold annotations by using extract_annotations.
    :param dev_file: input is the development file containing the gold annotations
    :return gold_labels: returns a list with the gold annotations
    """
    gold_labels = extract_annotations(dev_file, "Gold") 
    return gold_labels
          
def pred_annotations(extracted_file):
    """
    This function extracts the predicted annotations.
    :param extracted_file: input is the file containing the predicted annotations
    :return pred_labels: reutrns a list with the predicted annotations
    """
    pred_labels = extract_annotations(extracted_file, "Prediction")
    return pred_labels

#the next codes were inspired by this website https://coderzcolumn.com/tutorials/machine-learning/model-evaluation-scoring-metrics-scikit-learn-sklearn [29-01-2022]
def conf_matrix(gold_labels, pred_labels):
    """
    the function creates and prints a confusion matrix with gold and predicted labels
    :param gold_labels: the positional parameter consists of the gold annotations
    :param pred_labels: the positional parameter consists of the predicted annotations
    :return conf_mat: it returns a confusion matrix as a numpy.ndarray
    """
    
    conf_mat = confusion_matrix(gold_labels, pred_labels)
    print(conf_mat)
    return conf_mat

def heatmap_confusion_matrix(conf_mat):
    """
    the function creates a heatmap of the confusion matrix to visualize the distribution more clearly
    :input conf_mat: it takes a confusion matrix as an input
    :output: visualization in form of a heatmap 
    """
    with plt.style.context(('ggplot', 'seaborn')):
        fig = plt.figure(figsize=(6,6), num=1)
        plt.imshow(conf_mat, interpolation='nearest',cmap= plt.cm.Blues )
        plt.xticks([0,1],[0,1])
        plt.yticks([0,1],[0,1])
        plt.xlabel('Predicted Label')
        plt.ylabel('Actual Label')
        for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
            plt.text(j, i,conf_mat[i, j], horizontalalignment="center",color="red")
        plt.grid(None)
        plt.title('Confusion Matrix')
        plt.colorbar();

def get_evaluation(gold_labels, pred_labels):
    """
    the function creates the evaluation with precision, recall, f-score for the gold and predicted annotations
    :param gold_labels: the positional parameter consists of the gold annotations
    :param pred_labels: the positional parameter consists of the predicted annotations
    :output: it prints the evaluation of precision, recall, f-score of the chosen annotations
    """
    
    print('Precision                                   : %.3f'%precision_score(gold_labels, pred_labels,average='micro'))
    print('Recall                                      : %.3f'%recall_score(gold_labels, pred_labels,average='micro'))
    print('F1-Score                                    : %.3f'%f1_score(gold_labels,pred_labels,average='micro'))
    print(classification_report(gold_labels, pred_labels))
