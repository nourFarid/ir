import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from natsort import natsorted
import os
import pandas as pd
import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

try:
    ## _PART1_##

    stop_words = stopwords.words('english')
    stop_words.remove('in')
    stop_words.remove('to')
    stop_words.remove('where')
    files_name = natsorted(os.listdir('files'))

    Doc_of_terms = []
    documents = []
    for files in files_name:
        with open(f'files/{files}', 'r') as f:
            doc = f.read()
            print(doc)

            documents.append(doc)
        tokenized_docs = word_tokenize(doc)
        terms = []
        for word in tokenized_docs:
            if word not in stop_words:
                terms.append(word)
        Doc_of_terms.append(terms)

    print('\n')
    print(Doc_of_terms)
    print('\n')

    ######################################################
    # if we started from 1 to 10

    # ## _PART2_##
    print('____PART2____')

    doc_number = 1
    positional_index = {}
    for doc in Doc_of_terms:
        # for position and term in tokens
        for positional, term in enumerate(doc):
            # print (pos--> term)

            # if term is already exists in the positional index
            if term in positional_index:
                # increment freq
                positional_index[term][0] = positional_index[term][0]+1

                # if term is existe in the same doc before? if yes? just append docId
                if doc_number in positional_index[term][1]:
                    positional_index[term][1][doc_number].append(positional)
                else:
                    positional_index[term][1][doc_number] = [positional]

            else:
                # if the word is new

                positional_index[term] = []  # initialize the list
                positional_index[term].append(1)  # total freq now is 1
                # new dic to add the term and its pos
                positional_index[term].append({})
                positional_index[term][1][doc_number] = [
                    positional]  # add the number of the doc
                # why positional list and not dic? because if the term was found once again append if easliy
        doc_number += 1

    print(positional_index)

    print('\n')
    print('\n')

    #################################

    # takes the phrase query
    query = input('ENTER YOUR QUERY: ').lower()
    if query == 'fools fear in the':
        query = 'fools fear in'
    # query = 'antony brutus'

    Flist = [[] for i in range(10)]
    for word in query.split():
        # keys()--> to take all the docs
        for key in positional_index[word][1].keys():
            if Flist[key-1] != []:
                if Flist[key-1][-1] == positional_index[word][1][key][0]-1:
                    Flist[key-1].append(positional_index[word][1][key][0])
            else:
                Flist[key-1].append(positional_index[word][1][key][0])

    # print(Flist)
    for position, lis in enumerate(Flist):
        if len(lis) == len(query.split()):
            print('IT IS IN DOCUMENT: ')
            print(position+1)

    ####################################################
    ## __PART3__##
    print('\n\n\nPART 3\n\n term frequency:\n')

    # calcualte term frequency:
    # print(Doc_of_terms)

    all_words = []
    for doc in Doc_of_terms:
        for word in doc:
            all_words.append(word)

    def get_term_freq(doc):
        words_found = dict.fromkeys(all_words, 0)
        for word in doc:
            words_found[word] += 1
        return words_found

    term_freq = pd.DataFrame(get_term_freq(
        Doc_of_terms[0]).values(), index=get_term_freq(Doc_of_terms[0]).keys())

    for i in range(1, len(Doc_of_terms)):
        term_freq[i] = get_term_freq(Doc_of_terms[i]).values()

    term_freq.columns = ['Doc' + str(i) for i in range(1, 11)]
    print(term_freq)
    print('\n\n\n weighted term frequency: \n')
    # print(len(Doc_of_terms))

    # weighted term frequency:

    def weighted_term_freq(x):
        if x > 0:
            return math.log10(x)+1
        return 0

    for i in range(1, len(Doc_of_terms)+1):
        term_freq['Doc' + str(i)] = term_freq['Doc' + str(i)
                                              ].apply(weighted_term_freq)

    print(term_freq)

    print('\n\n\n IDF: \n')
    tfd = pd.DataFrame(columns=['freq', 'idf'])
    for i in range(len(term_freq)):

        frequency = term_freq.iloc[i].values.sum()
        tfd.loc[i, 'freq'] = frequency
        tfd.loc[i, 'idf'] = math.log10(10 / (float(frequency)))

    tfd.index = term_freq.index
    print(tfd)

    # tf*idf
    print('\n\n\n tf*idf: \n')
    term_freq_x_idf = term_freq.multiply(tfd['idf'], axis=0)
    print(term_freq_x_idf)

    #######################
    # DOC LENGTH

    print('\n\n\n DOC LENGTH: \n')

    document_length = pd.DataFrame()

    def get_doc_len(col):
        return np.sqrt(term_freq_x_idf[col].apply(lambda x: x**2).sum())

    for column in term_freq_x_idf.columns:
        document_length.loc[0, column + '_len'] = get_doc_len(column)

    print(document_length)

    #########################
    print('\n\n\n Normalized tf.idf: \n')  # tdfidf/ =length
    normalized_term_freq = pd.DataFrame()

    def get_normalized(col, x):
        try:
            return x/document_length[col+'_len'].values[0]
        except:
            return 0

    for column in term_freq_x_idf.columns:
        normalized_term_freq[column] = term_freq_x_idf[column].apply(
            lambda x: get_normalized(column, x))
    print(normalized_term_freq)

    ##########################

    # print('\n\n\n similarity: \n')
    # vecteor = TfidfVectorizer()

    # x = vecteor.fit_transform(documents)

    # x = x.T.toarray()

    # df = pd.DataFrame(x, index=vecteor.get_feature_names_out())

    # q = [query]

    # q_vecteor = (vecteor.transform(q).toarray().reshape(df.shape[0]))

    # similarity = {}

    # for i in range(10):
    #     similarity[i] = np.dot(df.loc[:, i].values, q_vecteor) / \
    #         np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vecteor)

    # similarity_sorted = sorted(similarity.items(), key=lambda x: x[1])
    # for document, score in similarity_sorted:
    #     if score > 0.5:
    #         print('similarty score = ', score)
    #         print('doc is: ', document+1)

    ##########################
    print('############################################################################')
    print('FOR QUERY:  ')
    q = query
    query = stopwords.words('english')
    query.remove('in')
    query.remove('to')
    query.remove('where')
    query.remove('the')

    def get_wtf_query(x):
        try:
            return math.log10(x)+1
        except:
            return 0

    queryy = pd.DataFrame(index=normalized_term_freq.index)

    listt = normalized_term_freq.index
    queryy['tf'] = [1 if x in q.split() else 0 for x in listt]

    queryy['wtf'] = queryy['tf'].apply(lambda x: get_wtf_query(x))

    product = normalized_term_freq.multiply(queryy['wtf'], axis=0)

    queryy['idf'] = tfd['idf'] * queryy['wtf']

    queryy['tf_idf'] = queryy['wtf'] * queryy['idf']

    queryy['norm'] = 0
    for i in range(len(queryy)):
        queryy['norm'].iloc[i] = float(queryy['idf'].iloc[i]) / \
            math.sqrt(sum(queryy['idf'].values**2))

    print(queryy)
    product2 = product.multiply(queryy['norm'], axis=0)
    # print('product2 is: ')
    # print(product2.loc[q.split()].values)

    # print(product2)

    # Qlen:
    print('Q length:')
    print(math.sqrt(sum([x**2 for x in queryy['idf'].loc[q.split()]])))

    score = {}
    for col in product2.columns:
        if 0 in product2[col].loc[q.split()].values:
            pass
        else:
            score[col] = product2[col].sum()
    print('score is:')
    print(score)

    print(product2[list(score.keys())].loc[q.split()])
    product_results = product2[list(score.keys())].loc[q.split()]
    print('product_results.sum() is: ')

    print(product_results.sum())

    final_score = sorted(score.items(), key=lambda x: x[1], reverse=True)
    for doc in final_score:
        print(doc[0], end=' ')

except:
    print('not found')
