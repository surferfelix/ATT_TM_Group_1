# System for Negation Cue Detection
TM_Group_1

The project was carried out by Konstantina Andronikou, Felix den Heijer, Mira Reisinger and Elena Weber during the seminar ‘Applied Text Mining 1: Methods' taught by Isa Maks.

### Annotations
The folder [**annotations**](https://github.com/surferfelix/ATT_TM_Group_1/tree/main/annotations) contains annotations of 10 articles by 4 annotators. The corpus consists of 5 articles by the CDC (Centers for Disease Control and Prevention USA) and 5 articles from different webpages. The annotations were made on the following .txt files:

* `21st-Century-Wire_20170627T181355.txt`
* `Activist-Post_20170704T090503.txt`
* `AGE-OF-AUTISM_20170620T044415.txt`
* `aids-gov_20170513T020021.txt`
* `Atlas-Monitor_20160703T084322.txt`
* `cdc-gov_20170521T155133.txt`
* `cdc-gov_20170614T145809.txt`
* `cdc-gov_20170617T024454.txt`
* `cdc-gov_20170617T195505.txt`
* `cdc-gov_20170618T093427.txt`

All annotations were made with the eHost software, and all of the annotators except for annotator 4 (Felix) used the windows distribution of the software. Annotator 4 annotated with the Mac OS X distribution. This is relevant since these versions are not cross compatible.


### Data
Development Data:
* `SEM-2012-SharedTask-CD-SCO-dev-simple.v2.txt`

Training Data:
* `SEM-2012-SharedTask-CD-SCO-training-simple.v2.txt`

Test data:

tbd for Assignment 4

### Prerequistes 
The requirements to run the provided code can be found in [**requirements.txt**](https://github.com/surferfelix/ATT_TM_Group_1/blob/main/requirements.txt).

### Word Embeddings 
We used a pre-trained GloVe word-embedding model to represent our tokens as features. When using this project, make sure you have a GloVe pre-trained embedding model downloaded (txt not bin). Alternatively, you can use a Word2Vec based embedding model, with a quick work-around.

#### Using a Word2Vec embedding model instead of GloVe
Currently, this project takes a path to a pre-trained GloVe embedding model and converts this to a Word2Vec style model. This means that with a short workaround you can load a word2vec based model (txt not bin). All you need to do is rename the name of your word2vec based model to 'temp_glove_as_word2vec.txt', and put this in the models folder of a clone of this repository. 

### Code
The folder **code** contains the following scripts and files: **NAME OF FILE AND SHORT DESCRIPTION WHAT IS HAPPENING**
* `utils.py`
* 
