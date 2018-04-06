import logging, warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import mysql.connector
from gensim import models
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Utils import *

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

def collecting_data(conn, text='all') :
    '''Loading raw article text from the database.

    Return
    articles_id : list
        List of articles id
    articles : list
        List of articles raw text
    '''

    logging.info("Loading Articles Data from database [START]")
    curA = conn.cursor(buffered=True)
    articles_id, articles = fetch_data(curA, text)
    
    logging.info("Loading Articles Data from database [DONE]")
    return articles_id, articles

def train_word2vec(articles_tokenized, modelname="trained_model.w2v") :
    w2v_model = models.Word2Vec(articles_tokenized, min_count=1)
    w2v_model.save(modelname)

def load_word2vec(modelname) :
    return models.Word2Vec.load(modelname)

def generate_article_matrix(articles_tokenized, w2v_model) :
    words = list(w2v_model.wv.vocab)
    vec_size = len(w2v_model[words[0]])

    vec = lambda w : w2v_model[w] if (w in w2v_model) else np.zeros(vec_size)
    article_matrix = list()
    for art in articles_tokenized :
        art_vecs = [vec(w) for w in art]
        art_vecs_mean = np.mean(art_vecs, axis=0)
        article_matrix.append(art_vecs_mean)
    article_matrix = np.array(article_matrix)
    return article_matrix

def calculate_silhouette(article_matrix, cluster_labels) :
    silhouette_avg = silhouette_score(article_matrix,cluster_labels)
    sample_silhouette_values = silhouette_samples(article_matrix, cluster_labels)
    return silhouette_avg, sample_silhouette_values

def cluster_word2vec(w2v_model, articles_tokenized, n_clusters, silhscorefile, plot=False) :
    
    #Konstruksi matriks article-word2vec_mean
    logging.info("Clustering [START]")
    logging.info("Preparing data for Clustering....")
    article_matrix = generate_article_matrix(articles_tokenized, w2v_model)

    #Kmeans clustering & silhouette score
    logging.info("Start Clustering....")
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                    verbose=False) #Print progress reports inside k-means algorithm
    idx = km.fit(article_matrix)
    cluster_labels = km.labels_.tolist()
    logging.info("Clustering [DONE]")

    #output
    silhouette_avg, sample_silhouette_values = calculate_silhouette(article_matrix,cluster_labels)
    silhouette_avg = round(silhouette_avg, 4)
    logging.info("For n_clusters = " + str(n_clusters) + \
              " The average silhouette_score is :" + str(silhouette_avg))
    with open(silhscorefile, 'w') as f :
        f.write('Original silhouette score : ' + str(silhouette_avg))
    if (plot) :
        dim_matrix = article_matrix.shape
        silhouette_plot(dim_matrix[0], n_clusters, cluster_labels, sample_silhouette_values, silhouette_avg)

    return cluster_labels, sample_silhouette_values

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
    silhouette_plot(dim_matrix[0], n_clusters, cluster_labels, sample_silhouette_values, silhouette_avg)

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

def store_cluster_label(conn, articles_id, cluster_labels, sample_silhouette_values) :
    curA = conn.cursor(buffered=True)
    
    truncate = ("truncate cluster_label")
    insert = ("insert into cluster_label values(%s,%s,%s)")
    curA.execute(truncate)
    for i in range(len(articles_id)) :
        curA.execute(insert, (str(articles_id[i]), str(cluster_labels[i]), str(round(sample_silhouette_values[i],4)) ))
    
    conn.commit()
    #logging.info('Storing cluster Label to Database [DONE]')  

def cluster_tocsv(conn, csvfile_name) :
    #open csv file
    csvfile = open(csvfile_name,'w',newline='')
    csvwriter = csv.writer(csvfile, delimiter=';')

    #open database connection
    curA = conn.cursor(buffered=True)

    #data query-ing
    select_article = ("select a.article_id, c.cluster_id, c.silh_score, a.article_title, a.article_abstract " \
                        + "from fg_m_article as a, cluster_label as c " \
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
    csvfile.close()
    #logging.info("Save to csv File : " + csvfile_name + " [DONE]")

def collecting_cluster_data(conn) :
    logging.info("Loading Cluster Data from database [START]")
    #open database connection 
    curA = conn.cursor(buffered=True)

    #get maximum cluster id
    max_id = ("select max(cluster_id) from cluster_label")
    curA.execute(max_id)
    for row in curA :
        n_cluster = row[0] + 1
    
    #get each cluster id & content
    clust_articles_id = []
    clust_articles_content = []
    for i in range(n_cluster) :
        select_cluster = "select a.article_id, a.article_title, a.article_abstract " + \
                         "from fg_m_article as a, cluster_label as c " + \
                         "where a.article_id = c.article_id and c.cluster_id = %s " + \
                         "order by a.article_id"
        curA.execute(select_cluster, (str(i),))
        c_art_id = []
        c_art_content = []
        for row in curA :
            c_art_id.append(row[0])
            c_art_content.append(row[1] + " " + row[2])
        clust_articles_id.append(c_art_id)
        clust_articles_content.append(c_art_content)
    
    logging.info("Loading Cluster Data from database [DONE]")
    return clust_articles_id, clust_articles_content

if __name__ == "__main__" :
    #do nothing
    print ("Clustering.py")