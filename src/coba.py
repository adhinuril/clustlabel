from Clustering import *

if __name__ == "__main__" :
    modelname = "output/all_articles.w2v"
    w2v_model = load_word2vec(modelname)
    dist = w2v_model.wmdistance('level edge value','level')
    print(dist)