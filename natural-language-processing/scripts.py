import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_classification_report

from nltk.tokenize.treebank import TreebankWordDetokenizer

crf = CRF(algorithm = 'lbfgs',
         c1 = 0.1,
         c2 = 0.1,
         max_iterations = 100,
         all_possible_transitions = False)

def load_data():
    df = pd.read_csv('../data/ner_dataset.csv', encoding = "ISO-8859-1")
    df.describe()
    df = df.fillna(method = 'ffill')
    
    # This is a class te get sentence. The each sentence will be list of tuples with its tag and pos.
    class sentence(object):
        def __init__(self, df):
            self.n_sent = 1
            self.df = df
            self.empty = False
            agg = lambda s : [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                           s['POS'].values.tolist(),
                                                           s['Tag'].values.tolist())]
            self.grouped = self.df.groupby("Sentence #").apply(agg)
            self.sentences = [s for s in self.grouped]

        def get_text(self):
            try:
                s = self.grouped['Sentence: {}'.format(self.n_sent)]
                self.n_sent +=1
                return s
            except:
                return None    
            
    #Displaying one full sentence
    getter = sentence(df)
    sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
    sentences[0]
    
    sent = getter.get_text()
    
    sentences = getter.sentences
    
    def word2features(sent, i):
        word = sent[i][0]
        postag = sent[i][1]

        features = {
            'bias': 1.0,
            'word.lower()': word.lower(),
            'word[-3:]': word[-3:],
            'word[-2:]': word[-2:],
            'word.isupper()': word.isupper(),
            'word.istitle()': word.istitle(),
            'word.isdigit()': word.isdigit(),
            'postag': postag,
            'postag[:2]': postag[:2],
        }
        if i > 0:
            word1 = sent[i-1][0]
            postag1 = sent[i-1][1]
            features.update({
                '-1:word.lower()': word1.lower(),
                '-1:word.istitle()': word1.istitle(),
                '-1:word.isupper()': word1.isupper(),
                '-1:postag': postag1,
                '-1:postag[:2]': postag1[:2],
            })
        else:
            features['BOS'] = True

        if i < len(sent)-1:
            word1 = sent[i+1][0]
            postag1 = sent[i+1][1]
            features.update({
                '+1:word.lower()': word1.lower(),
                '+1:word.istitle()': word1.istitle(),
                '+1:word.isupper()': word1.isupper(),
                '+1:postag': postag1,
                '+1:postag[:2]': postag1[:2],
            })
        else:
            features['EOS'] = True

        return features


    def sent2features(sent):
        return [word2features(sent, i) for i in range(len(sent))]

    def sent2labels(sent):
        return [label for token, postag, label in sent]

    def sent2tokens(sent):
        return [token for token, postag, label in sent]    
    
    reduced_tag_set = ['B-geo', 'B-gpe', 'B-org', 'B-per', 'B-tim', 'I-geo',
                       'I-gpe', 'I-org', 'I-per', 'I-tim', 'O']
    
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(s) for s in sentences]
    y = [[label if label in reduced_tag_set else 'O' for label in y_i] for y_i in y]  # reduce tag set
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.025)    
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def blur_labels(y, frac=0.67, num_blur=None):
    y_new = []
    error_array = []
    
    counter = 0
    
    for i in range(len(y)):
        error_array.append(list())
        y_new.append(list())
        change_steps = 0
        change_to = None
        skip = 0
        
        for j in range(len(y[i])):
            
            if skip > 0:
                skip -= 1
                continue
                
            current_tag = y[i][j]
            tagset = ['I-org', 'I-gpe', 'I-per', 'I-tim']
            if num_blur is None:
                num_blur = np.random.randint(1,4)
            if current_tag in tagset and  j < len(y[i]) - num_blur and not(y[i][j+1] == current_tag) and np.random.random() < frac:
                y_new[i].append(current_tag)
                error_array[i].append(False)
                
                for k in range(num_blur):
                    y_new[i].append(current_tag)
                    error_array[i].append(True)
                
                skip = num_blur
            else:
                error_array[i].append(False)
                y_new[i].append(current_tag)
        
    return y_new, error_array

def missing_labels(y, frac=0.67):
    y_new = []
    error_array = []
    
    counter = 0
    erasing = False
    
    for i in range(len(y)):
        error_array.append(list())
        y_new.append(list())
        
        
        for j in range(len(y[i])):            
            current_tag = y[i][j]
            tagset = ['B-org', 'B-per', 'B-tim', 
                   'I-org', 'I-per', 'I-tim'] 
            if current_tag in tagset and (np.random.random() < frac or erasing):
                y_new[i].append('O')
                error_array[i].append(True)                
                erasing = True
                
            else:
                error_array[i].append(False)
                y_new[i].append(current_tag)
                erasing = False
        
    return y_new, error_array


def missing_systematic_labels(X, y):
    import spacy
    from nltk.tokenize.treebank import TreebankWordDetokenizer
    nlp = spacy.load("en_core_web_sm")    
    
    y_new = []
    error_array = []
    
    counter = 0
    erasing = False
    
    for i in range(len(y)):
        if i%1000 == 0:
            print(i)
        error_array.append(list())
        y_new.append(list())
        sentence = TreebankWordDetokenizer().detokenize([t['word.lower()'] for t in X[i]])
        doc = nlp(sentence)
        ents = list(e.text for e in doc.ents)
        word_ents = list()
        for e in ents:
            word_ents.extend(e.split(' '))
        
        for j in range(len(y[i])):            
            current_tag = y[i][j]
            tagset = ['B-org', 'B-per', 'B-tim', 'B-geo', 
                   'I-org', 'I-per', 'I-tim', 'I-geo'] 
            if current_tag in tagset and not(X[i][j]['word.lower()'] in word_ents):
                y_new[i].append('O')
                error_array[i].append(True)                
                erasing = True
                
            else:
                error_array[i].append(False)
                y_new[i].append(current_tag)
                erasing = False
        
    return y_new, error_array

def print_report(y_pred, y_test, name):
    f1_score = flat_f1_score(y_test, y_pred, average = 'macro')
    print(name, f1_score)    
    report = flat_classification_report(y_test, y_pred, output_dict=True)
    clean_precision, clean_recall, clean_f1 = report['B-geo']['precision'], report['B-geo']['recall'], report['B-geo']['f1-score'] 
    print(flat_classification_report(y_test, y_pred))    

def basic_report(X_train, y_train, X_test, y_test, name='report'):
    crf.fit(X_train, y_train)
    y_pred = crf.predict(X_test)
    print_report(y_pred, y_test, name)
    
def pseudolabel_report(X_train, X_val, y_val, X_test, y_test, name='pseudolabel'):
    crf.fit(X_val, y_val)
    y_train = crf.predict(X_train)
    basic_report(X_train, y_train, X_test, y_test, name=name)

def ecn_expander(X, y, expand_x, expand_y):
    keys_prioritized = ['word.isupper()', 'word.istitle()', 'word.isdigit()', '+1:word.istitle()', 
                        '+1:word.isupper()', 'BOS', 'bias',  'word.lower()', 'word[-3:]', 'word[-2:]',
                        '+1:word.lower()', 'postag', 'postag[:2]', '+1:postag', '+1:postag[:2]', 
                        '-1:postag', '-1:postag[:2]', '-1:word.istitle()', '-1:word.lower()',
                        '-1:word.isupper()', 'EOS']
    
    expanded_y = []     
    for i in range(len(y)):
        expanded_y.append([])
        for j in range(len(y[i])):
            if not(expand_x):
                X_val_sub = X[i][j].copy()
            else:
                X_val_sub = {k: X[i][j][k] for k in keys_prioritized if k in X[i][j]}
            expanded_y[i].append(X_val_sub)
            expanded_y[i][j]['y'] = y[i][j]
            if expand_y:
                if j >= 1:
                    expanded_y[i][j]['y-1'] = y[i][j-1]
                else:
                    expanded_y[i][j]['y-1'] = 'N'
                if j >= 2:
                    expanded_y[i][j]['y-2'] = y[i][j-2]
                else:
                    expanded_y[i][j]['y-2'] = 'N'
                if j < len(y[i]) - 1:
                    expanded_y[i][j]['y+1'] = y[i][j+1]
                else:
                    expanded_y[i][j]['y+1'] = 'N'
                if j < len(y[i]) - 2:
                    expanded_y[i][j]['y+2'] = y[i][j+2]
                else:
                    expanded_y[i][j]['y+2'] = 'N'
    return expanded_y
    
def ecn_report(X_train, y_train, X_val, y_val, X_test, y_test, expand_x=False, expand_y=False, name='ecn'):
    crf_coarse = CRF(algorithm = 'lbfgs',
                     c1 = 0.1,
                     c2 = 0.1,
                     max_iterations = 100,
                     all_possible_transitions = False)

    crf_coarse.fit(X_train, y_train)
    y_val_pred = crf_coarse.predict(X_val)

    correction_network_input = ecn_expander(X_val, y_val_pred, expand_x=expand_x, expand_y=expand_y)

    crf_ecn = CRF(algorithm = 'lbfgs',
                  c1 = 0.1,
                  c2 = 0.1,
                  max_iterations = 100,
                  all_possible_transitions = False)

    crf_ecn.fit(correction_network_input, y_val)

    y_pred_coarse = crf_coarse.predict(X_test)
    ecn_input = ecn_expander(X_test, y_pred_coarse, expand_x=expand_x, expand_y=expand_y)
    y_pred_fine = crf_ecn.predict(ecn_input)

    print_report(y_pred_fine, y_test, name)