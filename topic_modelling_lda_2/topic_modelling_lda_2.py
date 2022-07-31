
#Topic Modeling 2:
#In this notebook I develop a topic modeling. During the implementation I focused on the methods offered by gensim pacage  to create the model.
# the dataset consists of approximately 12,000 NPR (National Public Radio) articles, obtained from their website www.npr.org

import pandas as pd 
import numpy as np
import gensim, re, spacy
from gensim import corpora 
from gensim.models import LdaModel, LdaMulticore, CoherenceModel 
from collections import Counter 

nlp = spacy.load('en_core_web_sm')

# I clean up the text by eliminating punctuation, stopwords, verbs and pronouns.
# In this way I leave only the essential in the article, so as to simplify the recognition of the topic
def makeCorpus ( text ):            
    text = text.lower()
    text = text.replace("'", " ")
    text = text.replace('\n', ' ')
    text = text.replace('\t', ' ')    
    text = re.sub( r'\S+@\S+', ' ', text)
    text = re.sub( '\w*\d\w*', ' ', text)
    text = re.sub( r'[^\w\s]', ' ', text)
    text = re.sub( r' +', ' ', text)
    if len(text) <= 5 :
        text = 'empty' 
    else:   
        doc = nlp(text)
        good = ''
        for tk in doc:        
            if tk.is_stop == False and tk.is_punct == False and tk.pos_ not in ['PRON', 'VERB']:
              good = good+tk.lemma_+' '
        text = good 
    text  = re.sub(r' +', ' ', text)                                   
    return   text 


def wordsForTopic ( wordsList ):
    dc_topics = {}    
    for i in range(0, len(wordsList)):
        myWords = []
        tp = wordsList[i]
        arguments= tp[1]
        arguments = re.sub( r'\d', '', arguments)
        arguments = re.sub( r'\.\*', '', arguments)
        arguments = arguments.replace('"', '')        
        for w in arguments.split('+'):
            myWords.append( w )
        dc_topics[i] = myWords 
    return dc_topics


if __name__ == '__main__':
    print("go")
    df = pd.read_csv('Article.csv')    
    print(" df size ",df.shape)
    print(list(df))
    
    df['dim_1'] = df['Article'].apply( lambda  x: len( x.split()))
    print(" mean words in articles ",df['dim_1'].mean())
    df['corpus'] = df['Article'].apply( makeCorpus)
    df['dim_2'] = df['corpus'].apply( lambda  x: len( x.split()))
    print(" mean words in corpus ",df['dim_2'].mean())

    # trasform the texts in a gensim format 
    g_doc = [ list(w.split()) for w in df['corpus']]

    # I make the vocavolary of all texts and filter by eliminating words with a frequency greater than 70% or that occur less than 10 times.
# So as to keep only the rare words specific to a topic.
# in a second step I  apply the countvectorize with this vocavolary 
    dc = corpora.Dictionary( g_doc)    
    print(" dim vocavolary ",len(dc))
    dc.filter_extremes(no_below=10, no_above=0.6)  
    print(" size of filter vocavolary  ",len(dc))
    
    cv = [ dc.doc2bow( line ) for line in g_doc ]

    # LDA model 
    topics = 7
    lda = LdaMulticore (corpus=cv, id2word=dc, num_topics=topics , random_state=42, passes=10)
    print(" fit LDA model ")    
    cm = CoherenceModel ( model=lda, texts=g_doc , dictionary=dc, coherence='c_v')
    val_cm = cm.get_coherence()
    print(" coerence ",val_cm)

    # with lda.print_topics I get the n most frequent words for each argument.
    wordsList = lda.print_topics(num_topics=topics , num_words=15)
    dc_topics = wordsForTopic ( wordsList)
    
    df_topic = pd.DataFrame( dc_topics)
    print(df_topic.head())

    # I apply the LDA model on  the dataset to predict the topics of the articles
    topicLabels= []
    for  y in lda[cv]:
        mx = 0
        z = -1
        for j in range(0, len(y)):
            tp = y[j]
            x = tp[1]
            if x > mx:
                mx = x
                z = j    
        topicLabels.append( z )
    print(" topics \n",Counter(topicLabels))
    df['topic_label'] =topicLabels 

    df.to_csv('articles_topic.csv', sep=';', index=False)
    df_topic .to_csv('words_for_topic.csv', sep=';', index=False)
    print("end")

 