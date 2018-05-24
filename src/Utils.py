import logging, warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from tqdm import tqdm
import mysql.connector
import pickle
import re
import os

def connectdb(db_name) :
    conn = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database=db_name)
    return conn 


def preprocess_text(text) :
    """Clean the text. It remove non-letter,
    convert to lowercase, omit stopwords and single-letter. No stemmer is used.

    Parameter 
    text : string
        The text that will be cleaned/preprocessed.
    
    Return
    words_preprocessed : string
        The cleaned text.
    """
    #remove non-letters
    text = re.sub("[^a-zA-Z]", " ", text)
    
    #convert to lowercase & tokenization
    #tokenizer from CountVectorizer
    #omit word that only 1 letter
    tokenizer = CountVectorizer().build_tokenizer()
    words = tokenizer(text.lower())
    
    #remove the stopwords
    words_preprocessed = [w for w in words if not w in stopwords.words("english")]
    
    #join the cleaned words
    return (" ".join(words_preprocessed))


def preprocess_text_experimental(text) :
    """Clean the text. It remove non-letter,
    convert to lowercase, omit stopwords and single-letter. No stemmer is used.

    Parameter 
    text : string
        The text that will be cleaned/preprocessed.
    
    Return
    words_preprocessed : string
        The cleaned text.
    """
    #remove non-letters
    #text = re.sub("[^a-zA-Z]", " ", text)
    
    #convert to lowercase & tokenization
    #tokenizer from CountVectorizer
    #omit word that only 1 letter
    tokenizer = CountVectorizer().build_tokenizer()
    words = tokenizer(text.lower())
    
    #remove the stopwords
    words_preprocessed = [w for w in words if not w in stopwords.words("english")]
    
    #join the cleaned words
    return (" ".join(words_preprocessed))


def preprocess_articles(articles_content) :
    logging.info("Pre-processing articles....")
    articles_tokenized = [preprocess_text(c).split(' ') for c in tqdm(articles_content, leave=False)]
    return articles_tokenized

def preprocess_clust_articles(clust_articles_content) :
    logging.info("Pre-processing cluster articles....")
    clust_articles_tokenized = []
    for clust in tqdm(clust_articles_content, leave=False) :
        c_art_tokenized = [preprocess_text(c).split(' ') for c in tqdm(clust, leave=False)]
        clust_articles_tokenized.append(c_art_tokenized)
    
    return clust_articles_tokenized

def save_to_pickle(picklename, *args) :
    with open(picklename,'wb') as f :
        pickle.dump(args, f, protocol=2)
    #logging.info("Save to pickle File : " + picklename + " [DONE]")

def load_from_pickle(picklename) :
    with open(picklename,'rb') as f :
        unpack = pickle.load(f)
    
    #logging.info("Load from pickle File : " + picklename + " [DONE]")
    return unpack

def clear_folder(folderpath,extension) :
    txtfiles = []
    for file in os.listdir(folderpath):
        if file.endswith("." + extension):
            txtfiles.append(os.path.join(folderpath, file))
    for f in txtfiles :
        os.remove(f)


if __name__ == "__main__" :
    print ("Utils.py")