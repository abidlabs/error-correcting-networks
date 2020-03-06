import pandas as pd
import matplotlib.pyplot as plt
import copy, random

from sklearn.model_selection import train_test_split
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_f1_score
from sklearn_crfsuite.metrics import flat_classification_report


df = pd.read_csv('../data/ner_dataset.csv', encoding = "ISO-8859-1")
df.describe()

df = df.fillna(method = 'ffill')


# This is a class te get sentence. The each sentence will be list of tuples with its tag and pos.
class Sentence(object):
    def __init__(self, df):
        self.n_sent = 1
        self.df = df
        self.empty = False
        agg = lambda s: [(w, p, t) for w, p, t in zip(s['Word'].values.tolist(),
                                                      s['POS'].values.tolist(),
                                                      s['Tag'].values.tolist())]
        self.grouped = self.df.groupby("Sentence #").apply(agg)
        self.sentences = [s for s in self.grouped]

    def get_text(self):
        try:
            s = self.grouped['Sentence: {}'.format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None


getter = Sentence(df)
sentences = getter.sentences


def word2features(sent, i):
    word_ = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word_.lower(),
        'word[-3:]': word_[-3:],
        'word[-2:]': word_[-2:],
        'word.isupper()': word_.isupper(),
        'word.istitle()': word_.istitle(),
        'word.isdigit()': word_.isdigit(),
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

crf = CRF(algorithm = 'lbfgs',
          c1 = 0.1,
          c2 = 0.1,
          max_iterations = 100,
          all_possible_transitions = False)


def blur_labels(y, frac=0.25):
    y_new = []
    error_array = []

    counter = 0

    for i in range(len(y)):
        error_array.append(list())
        y_new.append(list())
        change_steps = 0
        change_to = None

        for j in range(len(y[i])):

            current_tag = y[i][j]
            if current_tag == 'B-geo' and j >= 3:
                for k in range(3):
                    y_new[i][j - k - 1] = current_tag
                    error_array[i][j - k - 1] = True

                y_new[i].append(current_tag)
                error_array[i].append(False)
            else:
                error_array[i].append(False)
                y_new[i].append(current_tag)

    return y_new, error_array


y_train_new, error_train_array = blur_labels(y_train)

t = sum([len(a) for a in error_train_array]); print('Num tags:', t)
e = sum([sum(a) for a in error_train_array]); print('Num errs:', e)
print(e/t)

# crf.fit(X_train, y_train)
# y_pred = crf.predict(X_test)
# f1_score = flat_f1_score(y_test, y_pred, average = 'macro')
# print(f1_score)
#
# report = flat_classification_report(y_test, y_pred)
# print(report)
#
# crf.fit(X_train, y_train_new)
# y_pred = crf.predict(X_test)
# f1_score = flat_f1_score(y_test, y_pred, average = 'macro')
# print(f1_score)
#
# report = flat_classification_report(y_test, y_pred)
# print(report)

# for i in range(10):
#     print("-", end="")
#     for w, word in enumerate(X_train[i]):
#         word = word['word.lower()']
#         if error_train_array[i][w]:
#             print('*', end='')
#         if y_train_new[i][w] == 'O':
#             print(word.lower(), end=' ')
#         else:
#             print(word.upper(), end=' ')
#     print()
#
# for i in range(10):
#     print("-", end="")
#     for w, word in enumerate(X_test[i]):
#         word = word['word.lower()']
#         if y_pred[i][w] != y_test[i][w]:
#             print('*', end='')
#
#         if y_pred[i][w] == 'O':
#             print(word.lower(), end=' ')
#         else:
#             print(word.upper(), end=' ')
#     print()


def measure_method(error_pred, error_array, X, y_corrected, tag=None):
    print(tag)
    # measure what percent of errors are fixed
    np = 0
    nn = 0
    tp = 0
    fp = 0
    for i in range(len(error_pred)):
        for j in range(len(error_pred[i])):
            if error_pred[i][j] and error_pred[i][j] == error_array[i][j]:
                tp += 1
            elif error_pred[i][j]:
                fp += 1
            if error_array[i][j]:
                np += 1
            else:
                nn += 1

    print("TP errors detected: {}".format(tp / np))
    print("FP errors detected: {}".format(fp / nn))

    # measure accuracy
    crf.fit(X, y_corrected)
    y_pred = crf.predict(X_test)
    f1_score = flat_f1_score(y_test, y_pred, average='macro')
    print("F1 score on trained model: {}".format(f1_score))

    report = flat_classification_report(y_test, y_pred)
    print(report)

    return tp / np, fp / nn, f1_score


def pseudolabeled_on_validation(xv, yv):
    print('Pseudolabel')
    crf.fit(xv, yv)
    y_pred = crf.predict(X_train)

    error_pred = []
    y_corrected = []

    for i in range(len(y_pred)):
        error_pred.append([])
        y_corrected.append([])

        for j in range(len(y_pred[i])):
            if not (y_pred[i][j] == y_train_new[i][j]):
                error_pred[i].append(True)
            else:
                error_pred[i].append(False)

    return measure_method(error_pred, error_train_array, X_train, y_pred, tag="pseudo")


def gtc_with_x_y_neighboring_y(xv, yv):
    print('X,Y---------------Y')
    crf.fit(X_train, y_train_new)
    y_val_pred = crf.predict(xv)

    correction_network_input = []

    for i in range(len(y_val_pred)):
        correction_network_input.append([])
        for j in range(len(y_val_pred[i])):
            correction_network_input[i].append(xv[i][j])
            correction_network_input[i][j]['y'] = y_val_pred[i][j]
            if j >= 1:
                correction_network_input[i][j]['y-1'] = y_val_pred[i][j - 1]
            else:
                correction_network_input[i][j]['y-1'] = 'N'
            if j >= 2:
                correction_network_input[i][j]['y-2'] = y_val_pred[i][j - 2]
            else:
                correction_network_input[i][j]['y-2'] = 'N'
            if j < len(y_val_pred[i]) - 1:
                correction_network_input[i][j]['y+1'] = y_val_pred[i][j + 1]
            else:
                correction_network_input[i][j]['y+1'] = 'N'
            if j < len(y_val_pred[i]) - 2:
                correction_network_input[i][j]['y+2'] = y_val_pred[i][j + 2]
            else:
                correction_network_input[i][j]['y+2'] = 'N'

    crf.fit(correction_network_input, yv)

    X_expanded = []

    for i in range(len(y_train_new)):
        X_expanded.append([])
        for j in range(len(y_train_new[i])):
            X_expanded[i].append(X_train[i][j])
            X_expanded[i][j]['y'] = y_train_new[i][j]
            if j >= 1:
                X_expanded[i][j]['y-1'] = y_train_new[i][j - 1]
            else:
                X_expanded[i][j]['y-1'] = 'N'
            if j >= 2:
                X_expanded[i][j]['y-2'] = y_train_new[i][j - 2]
            else:
                X_expanded[i][j]['y-2'] = 'N'
            if j < len(y_train_new[i]) - 1:
                X_expanded[i][j]['y+1'] = y_train_new[i][j + 1]
            else:
                X_expanded[i][j]['y+1'] = 'N'
            if j < len(y_train_new[i]) - 2:
                X_expanded[i][j]['y+2'] = y_train_new[i][j + 2]
            else:
                X_expanded[i][j]['y+2'] = 'N'

    # Go from X_expanded to X_corrected
    y_corrected = crf.predict(X_expanded)

    error_pred = []

    for i in range(len(y_corrected)):
        error_pred.append([])
        for j in range(len(y_corrected[i])):
            if not (y_corrected[i][j] == y_train_new[i][j]):
                error_pred[i].append(True)
            else:
                error_pred[i].append(False)

    return measure_method(error_pred, error_train_array, X_train, y_corrected, tag="xy,y")


def gtc_with_y_neighboring_y(xv, yv):
    print('Y---------------Y')
    crf.fit(X_train, y_train_new)
    y_val_pred = crf.predict(xv)

    correction_network_input = []

    for i in range(len(y_val_pred)):
        correction_network_input.append([])
        for j in range(len(y_val_pred[i])):
            correction_network_input[i].append(dict())
            correction_network_input[i][j]['y'] = y_val_pred[i][j]
            if j >= 1:
                correction_network_input[i][j]['y-1'] = y_val_pred[i][j - 1]
            else:
                correction_network_input[i][j]['y-1'] = 'N'
            if j >= 2:
                correction_network_input[i][j]['y-2'] = y_val_pred[i][j - 2]
            else:
                correction_network_input[i][j]['y-2'] = 'N'
            if j < len(y_val_pred[i]) - 1:
                correction_network_input[i][j]['y+1'] = y_val_pred[i][j + 1]
            else:
                correction_network_input[i][j]['y+1'] = 'N'
            if j < len(y_val_pred[i]) - 2:
                correction_network_input[i][j]['y+2'] = y_val_pred[i][j + 2]
            else:
                correction_network_input[i][j]['y+2'] = 'N'

    crf.fit(correction_network_input, yv)

    X_expanded = []

    for i in range(len(y_train_new)):
        X_expanded.append([])
        for j in range(len(y_train_new[i])):
            X_expanded[i].append(dict())
            X_expanded[i][j]['y'] = y_train_new[i][j]
            if j >= 1:
                X_expanded[i][j]['y-1'] = y_train_new[i][j - 1]
            else:
                X_expanded[i][j]['y-1'] = 'N'
            if j >= 2:
                X_expanded[i][j]['y-2'] = y_train_new[i][j - 2]
            else:
                X_expanded[i][j]['y-2'] = 'N'
            if j < len(y_train_new[i]) - 1:
                X_expanded[i][j]['y+1'] = y_train_new[i][j + 1]
            else:
                X_expanded[i][j]['y+1'] = 'N'
            if j < len(y_train_new[i]) - 2:
                X_expanded[i][j]['y+2'] = y_train_new[i][j + 2]
            else:
                X_expanded[i][j]['y+2'] = 'N'

    # Go from X_expanded to X_corrected
    y_corrected = crf.predict(X_expanded)

    error_pred = []

    for i in range(len(y_corrected)):
        error_pred.append([])
        for j in range(len(y_corrected[i])):
            if not (y_corrected[i][j] == y_train_new[i][j]):
                error_pred[i].append(True)
            else:
                error_pred[i].append(False)

    return measure_method(error_pred, error_train_array, X_train, y_corrected, tag="y,y")


yny_tps = []
yny_fps = []
yny_f1s = []

# ns = [25, 50, 100, 200, 400, 800]
ns = [100]

for n in ns:
    X_val_ = X_val[:n]
    y_val_ = y_val[:n]
    tp, fp, f1 = gtc_with_y_neighboring_y(X_val_, y_val_)
    yny_tps.append(tp)
    yny_fps.append(fp)
    yny_f1s.append(f1)

xny_tps = []
xny_fps = []
xny_f1s = []

for n in ns:
    X_val_ = X_val[:n]
    y_val_ = y_val[:n]
    tp, fp, f1 = gtc_with_x_y_neighboring_y(X_val_, y_val_)
    xny_tps.append(tp)
    xny_fps.append(fp)
    xny_f1s.append(f1)

val_tps = []
val_fps = []
val_f1s = []

for n in ns:
    X_val_ = X_val[:n]
    y_val_ = y_val[:n]
    tp, fp, f1 = pseudolabeled_on_validation(X_val_, y_val_)
    val_tps.append(tp)
    val_fps.append(fp)
    val_f1s.append(f1)


plt.semilogx(ns, yny_f1s, '-o', label='ECN_ys')
plt.semilogx(ns, xny_f1s, '-o', label='ECN_x_ys')
plt.semilogx(ns, val_f1s, '-o', label='Pseudolabel')
plt.legend()
plt.xlabel('Validation Size')
plt.ylabel('Weighted F1 score')
plt.savefig('f1s.png')

plt.semilogx(ns, yny_tps, '-o', label='ECN_ys')
plt.semilogx(ns, xny_tps, '-o', label='ECN_x_ys')
plt.semilogx(ns, val_tps, '-o', label='Pseudolabel')
plt.legend()
plt.xlabel('Validation Size')
plt.ylabel('True positive (error detected)')
plt.savefig('tps.png')

plt.semilogx(ns, yny_fps, '-o', label='ECN_ys')
plt.semilogx(ns, xny_fps, '-o', label='ECN_x_ys')
plt.semilogx(ns, val_fps, '-o', label='Pseudolabel')
plt.legend()
plt.xlabel('Validation Size')
plt.ylabel('False positive (error detected)')
plt.savefig('fps.png')


