import logging, warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import mysql.connector
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim import models
import scipy, numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import csv
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from tqdm import tqdm

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

def fetch_data(curA, text='all') :
    """Fetching scientific article from database 
    based on publication venue name.

    Parameter
    curA : MySql Connector Cursor
        Database Connection Cursor instance.
    text : string
        The name of publication venue.

    Return
    articles : list
        The list of title + abstratcs.
    articles_id : list
        The list of article_id
    """
    articles = []
    articles_id = []

    if (text == 'all') :
        select_article = ("SELECT article_id, article_title, article_abstract FROM fg_m_article " \
            +"WHERE article_abstract != '' ORDER BY article_id")
        curA.execute(select_article)
    else :
        select_m_article = ("SELECT article_id, article_title, article_abstract FROM fg_m_article where " \
            +"article_publicationvenuetext=%s ORDER BY article_id")
        curA.execute(select_article, (text,))
    
    for (article_id, article_title, article_abstract) in curA :
        articles.append(article_title + " " + article_abstract)
        articles_id.append(article_id)
    
    return articles_id, articles

def collecting_data(text='all') :
    '''Loading raw article text from the database.

    Return
    articles_id : list
        List of articles id
    articles : list
        List of articles raw text
    '''

    logging.info("Loading Data from database.....")
    cnx = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database='article')
    curA = cnx.cursor(buffered=True)
    articles_id, articles = fetch_data(curA, text)
    cnx.close()
    
    logging.info("Data Loaded Successfully!")
    return articles_id, articles

def save_to_pickle(picklename, *args) :
    with open(picklename,'wb') as f :
        pickle.dump(args, f)
    logging.info("Save to pickle done! File : " + picklename)

def load_from_pickle(picklename) :
    with open(picklename,'rb') as f :
        unpack = pickle.load(f)
    
    logging.info("Load from pickle done! File : " + picklename)
    return unpack

def train_word2vec(articles_tokenized, modelname="trained_model.w2v") :
    w2v_model = models.Word2Vec(articles_tokenized, min_count=1)
    w2v_model.save(modelname)

def cluster_word2vec(w2v_model, articles_tokenized, n_clusters) :
    
    #Cari ukuran dimensi word2vec
    words = list(w2v_model.wv.vocab)
    vec_size = len(w2v_model[words[0]])
    
    #Konstruksi matriks article-word2vec_mean
    #logging.info("Preparing data for Clustering....")
    vec = lambda w : w2v_model[w] if (w in w2v_model) else np.zeros(vec_size)
    article_matrix = list()
    for art in tqdm(articles_tokenized, leave=False) :
        art_vecs = [vec(w) for w in art]
        art_vecs_mean = np.mean(art_vecs, axis=0)
        article_matrix.append(art_vecs_mean)
    article_matrix = np.array(article_matrix)

    #Kmeans clustering & silhouette score
    #logging.info("Start Clustering.....")
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=500, n_init=1,
                    verbose=False) #Print progress reports inside k-means algorithm
    idx = km.fit(article_matrix)
    cluster_labels = km.labels_.tolist()

    #output
    silhouette_avg = silhouette_score(article_matrix,cluster_labels)
    #print("For n_clusters =", n_clusters,
    #          "The average silhouette_score is :", "%.4f" % silhouette_avg)
    sample_silhouette_values = silhouette_samples(article_matrix, cluster_labels)
    dim_matrix = article_matrix.shape
    
    #silhouette_plot(dim_matrix[0], n_clusters, cluster_labels, sample_silhouette_values, silhouette_avg)
    #return cluster_labels, sample_silhouette_values
    return km.inertia_

def cluster_tfidf(articles_tokenized, n_clusters) :
    logging.info("Preparing data for Clustering....")
    articles_doc = [" ".join(x) for x in articles_tokenized]
    
    #matrix tfidf construction
    tfidf_vectorizer = TfidfVectorizer()
    matrix_tfidf = tfidf_vectorizer.fit_transform(articles_doc)

    #clustering
    logging.info("Start Clustering.....")
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                    verbose=False) #Print progress reports inside k-means algorithm
    idx = km.fit(matrix_tfidf)
    cluster_labels = km.labels_.tolist()

    #output
    silhouette_avg = silhouette_score(matrix_tfidf,cluster_labels)
    print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", "%.4f" % silhouette_avg)
    sample_silhouette_values = silhouette_samples(matrix_tfidf, cluster_labels)
    dim_matrix = matrix_tfidf.shape
    print("Dimension : ",dim_matrix[0],"X",dim_matrix[1])
    with open('matrix_tfidf.txt','w') as f :
        f.write(str(matrix_tfidf))
    print("non-zero entries : ", matrix_tfidf.getnnz())

    #silhouette_plot(dim_matrix[0], n_clusters, cluster_labels, sample_silhouette_values, silhouette_avg)

    return cluster_labels, sample_silhouette_values

def silhouette_plot(num_docs, n_clusters, clusters_labels, sample_silhouette_values, silhouette_avg):
    y_lower = 10
    fig, ax1 = plt.subplots()
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 1])  # The silhouette coefficient can range from
    ax1.set_ylim([0, num_docs + (n_clusters + 1) * 5])  # inserting blank space between silhouette

    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = []
        for index in range(len(clusters_labels)):
            if (clusters_labels[index] == i): ith_cluster_silhouette_values.append(sample_silhouette_values[index])

        ith_cluster_silhouette_values.sort()
        size_cluster_i = len(ith_cluster_silhouette_values)
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 5  # 5 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.show()

def store_cluster_label(articles_id, cluster_labels, sample_silhouette_values) :
    cnx = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database='article')
    curA = cnx.cursor(buffered=True)
    
    truncate = ("truncate article.cluster_label")
    insert = ("insert into article.cluster_label values(%s,%s,%s)")
    curA.execute(truncate)
    for i in range(len(articles_id)) :
        curA.execute(insert, (str(articles_id[i]), str(cluster_labels[i]), str(round(sample_silhouette_values[i],4)) ))
    
    cnx.commit()
    cnx.close()
    logging.info('Cluster Label Stored to Database!')  

def cluster_tocsv(csvfile_name) :
    #open csv file
    csvfile = open(csvfile_name,'w',newline='')
    csvwriter = csv.writer(csvfile, delimiter=';')

    #open database connection
    cnx = mysql.connector.connect(user='root', password='admin', host='127.0.0.1', database='article')
    curA = cnx.cursor(buffered=True)

    #data query-ing
    select_article = ("select a.article_id, c.cluster_id, c.silh_score, a.article_title, a.article_abstract " \
                        + "from article.fg_m_article as a, article.cluster_label as c " \
                        + "where c.article_id = a.article_id " \
                        + "order by c.article_id")
    curA.execute(select_article)                        

    #data writing to csv
    csvwriter.writerow(["Article ID","Cluster ID","Silhouette","Title","Abstract"])
    for row in curA :
        try :
            csvwriter.writerow(row)
        except UnicodeEncodeError :
            csvwriter.writerow([row[0],row[1], row[2], row[3].encode('utf-8'),row[4].encode('utf-8')])

    #close file and connection
    cnx.close()
    csvfile.close()
    logging.info(csvfile_name + " Created Successfully!")

if __name__ == "__main__" :
    #PREPARE THE OUTPUT
    DB_NAME = 
    output_folder = 'output_' + DB_NAME + '/'
    
    CLUST_ERRORS = output_folder + 'cluster_errors.txt'
    ARTICLE = output_folder + 'article_dump.pkl'
    
    #load data directly from database
    #articles_id, articles_content = collecting_data()
    
    #pre-processing article text
    #logging.info("Pre-processing data....")
    #articles_tokenized = []
    #for content in articles_content :
    #    articles_tokenized.append(preprocess_text(content).split(" "))

    #save into pickle file
    #save_to_pickle('article_dump.pkl', articles_id, articles_tokenized)
    #load data from pickle file
    unpack = load_from_pickle(ARTICLE)
    articles_id, articles_tokenized = unpack[0], unpack[1]
    
    #training word2vec
    modelname = "w2v_model/all_articles.w2v"
    #train_word2vec(articles_tokenized, "all_articles.w2v")
    w2v_model = models.Word2Vec.load(modelname)

    #Elbow Analysis
    cluster_range = range( 2, 20 )
    cluster_errors = []
    for num_clusters in tqdm(cluster_range):
        error_rate = cluster_word2vec(w2v_model,articles_tokenized,num_clusters)
        cluster_errors.append(error_rate)
    
    with open(CLUST_ERRORS,'w') as fout :
        for i in range(len(cluster_errors)) :
            fout.write("Cluster-" + str(i+1) + " : " + str(cluster_errors[i]) + "\n")
    clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
    plt.figure(figsize=(12,6))
    plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
    plt.show()
    