import logging, warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import mysql.connector
from gensim import models
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans, DBSCAN, Birch, MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
from Utils import *

def collecting_data(db_name) :
    '''Loading raw article text from the database.

    Return
    articles_id : list
        List of articles id
    articles_text : list
        List of articles raw text
    '''

    logging.info("Loading Articles Data from database [START]")
    articles_text = []
    articles_id = []

    query = ("SELECT article_id, article_title, article_abstract FROM fg_m_article " \
            +"WHERE article_abstract != '' ORDER BY article_id")

    conn = connectdb(db_name)
    c = conn.cursor(buffered=True)
    c.execute(query)

    for (article_id, article_title, article_abstract) in c :
        articles_text.append(article_title + " : " + article_abstract)
        articles_id.append(article_id)
    
    conn.close()
    logging.info("Loading Articles Data from database [DONE]")
    return articles_id, articles_text

def train_word2vec(articles_tokenized, modelname="trained_model.w2v") :
    w2v_model = models.Word2Vec(articles_tokenized, min_count=1)
    w2v_model.save(modelname)
    raise SystemExit

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


def silh_analysis(articles_tokenized, w2v_model, silhfile, silhcsv, n_max=30) :
    logging.info("Silhouette Analysis Commencing...")
    article_matrix = generate_article_matrix(articles_tokenized, w2v_model)
    cluster_range = range( 2, n_max )
    silh_scores = []
    cluster_errors = []
    for num_clusters in tqdm(cluster_range) :
        km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=100, n_init=1,
                    verbose=False) #Print progress reports inside k-means algorithm
        idx = km.fit(article_matrix)
        cluster_labels = km.labels_.tolist()
        error_rate = km.inertia_
        cluster_errors.append(error_rate)
        silhouette_avg, sample_silhouette_values = calculate_silhouette(article_matrix,cluster_labels)
        silhouette_avg = round(silhouette_avg, 4)
        silh_scores.append(silhouette_avg)
    
    #Find the largest silhouette score
    max_silh = silh_scores[0]
    max_silh_idx = 0
    #print(silh_scores)
    for i in range(len(cluster_range)) :
        if silh_scores[i] > max_silh :
            max_silh = silh_scores[i]
            max_silh_idx = cluster_range[i] 

    #clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "cluster_errors": cluster_errors } )
    clusters_df = pd.DataFrame( { "num_clusters":cluster_range, "silh_scores": silh_scores } )
    plt.figure(figsize=(12,6))
    plt.xticks(clusters_df.num_clusters)
    plt.plot( clusters_df.num_clusters, clusters_df.silh_scores, marker = "o" )
    plt.title('Analisis Silhouette (Optimal = ' + str(max_silh_idx) + ')')
    plt.xlabel('Jumlah Klaster')
    plt.ylabel('Rata - Rata Silhouette')
    plt.savefig(silhfile)
    logging.info('Optimal k = ' + str(max_silh_idx) + ', score = ' + str(max_silh))

    #output to csv
    with open(silhcsv,'w',newline='') as f :
        writer = csv.writer(f, delimiter=';')
        writer.writerow(['Jumlah Klaster','Rata-Rata Silhouette'])
        for i in range(len(cluster_range)) :
            writer.writerow([cluster_range[i],silh_scores[i]])

    #return max_silh_idx
    raise SystemExit


def cluster_word2vec(w2v_model, articles_tokenized, n_clusters, silhscorefile, plot=False) :
    
    #Konstruksi matriks article-word2vec_mean
    #Konstruksi matris docs-features
    logging.info("Clustering [START]")
    logging.info("Preparing data for Clustering....")
    article_matrix = generate_article_matrix(articles_tokenized, w2v_model)

    #Kmeans clustering & silhouette score
    logging.info("Start Clustering KMeans w2v....")
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100, n_init=1,
                    verbose=False) #Print progress reports inside k-means algorithm
    idx = km.fit(article_matrix)
    cluster_labels = km.labels_.tolist()
    cluster_centers = km.cluster_centers_
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

    return cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values


def cluster_dbscan(w2v_model, articles_tokenized, silhscorefile, plot=False) :
    
    #Konstruksi matriks article-word2vec_mean
    #Konstruksi matris docs-features
    logging.info("Clustering [START]")
    logging.info("Preparing data for Clustering....")
    article_matrix = generate_article_matrix(articles_tokenized, w2v_model)
    print('article matrix :')
    print(article_matrix[0])
    '''
    logging.info("Clustering [START]")
    logging.info("Preparing data for Clustering....")
    articles_doc = [" ".join(x) for x in articles_tokenized]
    
    #matrix tfidf construction
    tfidf_vectorizer = TfidfVectorizer()
    article_matrix = tfidf_vectorizer.fit_transform(articles_doc)
    print('article matrix :')
    print(article_matrix[0])
    '''
    #Clustering & silhouette score
    logging.info("Start Clustering DBSCAN....")
    dbs = DBSCAN(eps=0.8 ,min_samples=3, metric='euclidean')
    idx = dbs.fit(article_matrix)
    cluster_labels = dbs.labels_.tolist()
    cluster_centers = dbs.components_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    #print('nclust : ', n_clusters)
    #print('labels : ', cluster_labels)
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

    return cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values


def cluster_birch(w2v_model, articles_tokenized, n_clusters, silhscorefile, plot=False) :
    
    #Konstruksi matriks article-word2vec_mean
    #Konstruksi matris docs-features
    logging.info("Clustering [START]")
    logging.info("Preparing data for Clustering....")
    article_matrix = generate_article_matrix(articles_tokenized, w2v_model)

    #Kmeans clustering & silhouette score
    logging.info("Start Clustering Birch....")
    brc = Birch(n_clusters=n_clusters)
    idx = brc.fit(article_matrix)
    cluster_labels = brc.labels_
    cluster_centers = brc.subcluster_centers_
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
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

    return cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values


def cluster_spectral(w2v_model, articles_tokenized, n_clusters, silhscorefile, plot=False) :
    
    #Konstruksi matriks article-word2vec_mean
    #Konstruksi matris docs-features
    logging.info("Clustering [START]")
    logging.info("Preparing data for Clustering....")
    article_matrix = generate_article_matrix(articles_tokenized, w2v_model)

    #Kmeans clustering & silhouette score
    logging.info("Start Clustering SpectralClustering....")
    bandwidth = estimate_bandwidth(article_matrix)
    sc = SpectralClustering(n_clusters=n_clusters)
    sc.fit(article_matrix)
    cluster_labels = sc.labels_
    cluster_centers = []
    labels_unique = np.unique(cluster_labels)
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

    return cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values


def cluster_tfidf(articles_tokenized, n_clusters, silhscorefile, plot=False) :
    logging.info("Clustering [START]")
    logging.info("Preparing data for Clustering....")
    articles_doc = [" ".join(x) for x in articles_tokenized]
    
    #matrix tfidf construction
    tfidf_vectorizer = TfidfVectorizer()
    article_matrix = tfidf_vectorizer.fit_transform(articles_doc)

    #clustering
    logging.info("Start Clustering KMeans Tf-IDF.....")
    km = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=50, n_init=1,
                    verbose=False) #Print progress reports inside k-means algorithm
    idx = km.fit(article_matrix)
    cluster_labels = km.labels_.tolist()
    cluster_centers = km.cluster_centers_
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

    return cluster_labels, cluster_centers, silhouette_avg, sample_silhouette_values

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

def cluster_tocsv(csvfile_name, csvfile_name2,
                  clust_articles_id, clust_articles_tokenized, clust_articles_text, clust_articles_silh) :
    n_cluster = len(clust_articles_id)
    
    #open csv file
    csvfile = open(csvfile_name,'w',newline='')
    csvwriter = csv.writer(csvfile, delimiter=';')                        

    #avg silh score writting
    avg = lambda x : sum(x) / len(x)
    clust_articles_avgsilh = [avg(i) for i in clust_articles_silh]

    #data writing to csv
    csvwriter.writerow(['Cluster ID','ID','Title','Abstract','Silhouette'])
    for clust_id in range(n_cluster) :
        n_data = len(clust_articles_id[clust_id])
        for data_id in range(n_data) :
            art_id = clust_articles_id[clust_id][data_id]
            art_text = clust_articles_text[clust_id][data_id].split(' : ')
            art_title = art_text[0]
            art_abstract = ' '.join(art_text[1:len(art_text)])
            art_silh = clust_articles_silh[clust_id][data_id]

            try :
                csvwriter.writerow([clust_id, art_id, art_title, art_abstract, art_silh])
            except UnicodeEncodeError :
                csvwriter.writerow([clust_id, art_id, art_title.encode('utf-8'), art_abstract.encode('utf-8'), art_silh])

    #close file and connection
    csvfile.close()

    #writting cluster average
    csvfile = open(csvfile_name2,'w',newline='')
    csvwriter = csv.writer(csvfile, delimiter=';')
    csvwriter.writerow(['Cluster ID','Average Silhouette'])
    for i in range(n_cluster) :
        csvwriter.writerow([i, clust_articles_avgsilh[i]])
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

def postprocess_clustering(cluster_labels, articles_id, articles_tokenized, articles_text, articles_silh) :
    n_clust = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    clust_articles_id = []
    clust_articles_tokenized = []
    clust_articles_text = []
    clust_articles_silh = []

    #Initialize clusters variables
    for i in range(n_clust) :
        clust_articles_id.append([])
        clust_articles_tokenized.append([])
        clust_articles_text.append([])
        clust_articles_silh.append([])

    #Seperate old data into cluster-format variable
    for i in range(len(cluster_labels)) :
        #Define current cluster index
        clust_id = cluster_labels[i]
        
        #Anticipate noisy data on DBSCAN clustering
        if (clust_id == -1) :
            continue

        #Get data from old variable
        art_id = articles_id[i]
        art_tokenized = articles_tokenized[i]
        art_text = articles_text[i]
        art_silh = articles_silh[i]

        #Store data to cluster-format variable
        clust_articles_id[clust_id].append(art_id)
        clust_articles_tokenized[clust_id].append(art_tokenized)
        clust_articles_text[clust_id].append(art_text)
        clust_articles_silh[clust_id].append(art_silh)
    
    return clust_articles_id, clust_articles_tokenized, clust_articles_text, clust_articles_silh 

if __name__ == "__main__" :
    #do nothing
    print ("Clustering.py")