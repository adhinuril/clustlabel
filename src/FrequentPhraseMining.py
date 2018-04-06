import logging, warnings
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def generate_act_idx_init(text_tokenized) :
    act_idx = dict()
    for i in range(len(text_tokenized)) :
        token = text_tokenized[i]
        if token not in act_idx :
            act_idx[token] = []
        act_idx[token].append(i)
    return act_idx

def filter_act_idx(act_idx, min_count) :
    act_idx_copy = dict(act_idx)
    for token in act_idx_copy :
        if (len(act_idx[token]) < min_count) :
            del act_idx[token]

def merge_token(token1, token2) :
    t2_tokens = token2.split(' ')
    last_token_idx = len(t2_tokens) - 1
    last_token = t2_tokens[last_token_idx]
    new_token = token1 + ' ' + last_token
    return new_token

def generate_next_act_idx(act_idx) :
    all_idx = []
    all_token = []
    for token in act_idx :
        idxs = act_idx[token]
        for idx in idxs :
            all_idx.append(idx)
            all_token.append(token)
    next_act_idx = dict()
    for i in range(len(all_idx)) :
        next_idx = all_idx[i] + 1
        if next_idx in all_idx :
            j = all_idx.index(next_idx)
            new_token = merge_token(all_token[i], all_token[j])
            if new_token not in next_act_idx :
                next_act_idx[new_token] = []
            next_act_idx[new_token].append(all_idx[i])
    
    return next_act_idx

def store_phrase_count(phrase_count, act_idx) :
    for token in act_idx :
        if token not in phrase_count :
            phrase_count[token] = 0
        phrase_count[token] += len(act_idx[token])

def extract_phrases(docs_tokenized, min_count) :
    '''
    Note :
    - Tidak mengekstrak 1-gram
    '''
    doc_num = len(docs_tokenized)
    doc_list = [i for i in range(doc_num)]
    doc_del_list = []
    phrase_count = dict()
    
    #Initialize index active
    docs_act_idx = []
    for i in range(doc_num) :
        docs_act_idx.append([])
        act_idx_init = generate_act_idx_init(docs_tokenized[i])
        docs_act_idx[i].append(act_idx_init)
    
    #Frequent Phrase Mining
    n = 0
    while len(doc_list) != 0 :
        for i in doc_list :
            n_act_idx = docs_act_idx[i][n]  #Current index active
            filter_act_idx(n_act_idx, min_count)
            next_act_idx = dict()
            if (len(n_act_idx) != 0) :
                next_act_idx = generate_next_act_idx(n_act_idx)
                docs_act_idx[i].append(next_act_idx)
                store_phrase_count(phrase_count, next_act_idx)
            else :
                doc_del_list.append(i)
        n += 1
        #Update the document index list
        doc_list = [i for i in doc_list if i not in doc_del_list]
    return  phrase_count

def extract_phrases_v2(docs_tokenized, min_count) :
    '''
    Note :
    - Mengekstrak 1-gram
    '''
    doc_num = len(docs_tokenized)
    doc_list = [i for i in range(doc_num)]
    doc_del_list = []
    phrase_count = dict()
    
    #Initialize index active
    docs_act_idx = []
    for i in range(doc_num) :
        docs_act_idx.append([])
        act_idx_init = generate_act_idx_init(docs_tokenized[i])
        docs_act_idx[i].append(act_idx_init)
    
    #Frequent Phrase Mining
    n = 0
    while len(doc_list) != 0 :
        for i in doc_list :
            n_act_idx = docs_act_idx[i][n]  #Current index active
            filter_act_idx(n_act_idx, min_count)
            store_phrase_count(phrase_count, n_act_idx)
            next_act_idx = dict()
            if (len(n_act_idx) != 0) :
                next_act_idx = generate_next_act_idx(n_act_idx)
                docs_act_idx[i].append(next_act_idx)
            else :
                doc_del_list.append(i)
        n += 1
        #Update the document index list
        doc_list = [i for i in doc_list if i not in doc_del_list]
    return  phrase_count

def extract_clust_phrases(clust_articles_tokenized, min_count, min_count_phrase) :
    logging.info("Extracting cluster phrases [START]")

    clust_words = []
    clust_phrases = []
    for clust in clust_articles_tokenized :
        #WORD FILTERING
        c_words = []
        c_phrases = []
        phrase_count = extract_phrases(clust,min_count)
        for phrase in phrase_count :
            if phrase_count[phrase] < min_count_phrase :
                continue
            #CEK APAKAH FRASE SUDAH ADA DI LIST GLOBAL FRASE KLASTER
            if phrase not in c_phrases :
                c_phrases.append(phrase)
                #CEK APAKAH KATA (PEMBENTUK FRASE) SUDAH ADA DI LIST GLOBAL KATA KLASTER
                token = phrase.split(' ')
                for t in token :
                    if t not in c_words :
                        c_words.append(t)
        clust_words.append(c_words)
        clust_phrases.append(c_phrases)
    
    logging.info("Extracting cluster phrases [DONE]")
    return clust_words, clust_phrases


if __name__ == "__main__" :
    doc1_tokenized = ['born','pretoria','elon','musk','taught','himself','computer',
                      'programming','age','12','queen','university','elon','musk','enter',
                      'queen','university']
    doc2_tokenized = ['elon','musk','part','commerce','program','queen','university',
                      'pretty','good','job','placing','students','very','much','same',
                      'roles','wharton','undergrads','elon','musk','wharton','undergrads']
    doc3_tokenized = ['age','17','1898','elon','musk','moved','canada','attend','queen',
                      'university','avoid','mandatory','service','south','african','military',
                      'elon','musk','enter','queen','university','south','african','military']
    docs_tokenized = [doc1_tokenized, doc2_tokenized, doc3_tokenized]

    '''
    phrase_count = dict()
    act_idx = generate_act_idx_init(doc3_tokenized)
    filter_act_idx(act_idx,2)
    next_act_idx = generate_next_act_idx(act_idx)
    store_phrase_count(phrase_count, next_act_idx)
    filter_act_idx(next_act_idx,2)
    next_act_idx2 = generate_next_act_idx(next_act_idx)
    store_phrase_count(phrase_count, next_act_idx2)
    '''
    phrase_count = extract_phrases_v2(docs_tokenized, 2)
    for token in phrase_count :
        print(token) 
        print(phrase_count[token])
    
    
