# @Author  : Peizhao Li
# @Contact : peizhaoli05@gmail.com

from load.ingest import load_conll
from pandas import DataFrame
import pomegranate as pg
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

NUM_LABEL = 5
dict = {"ORG": 0, "PER": 1, "LOC": 2, "MISC": 3, "O": 4}


def label_func(label):
    label = label.split('-')[-1]
    if label not in dict:
        raise ValueError('unknown label')
    return label


def new_word(words, new):
    for word in words:
        word[new] = 0


def word_acc(preds, labels, acc):
    for pred, label in zip(preds, labels):
        if pred == label:
            acc += 1

    return acc


def phrase_acc(preds, labels, gt_phrase, pred_phrase, hit):
    gt_phrase_idx = [i for i in range(len(labels)) if labels[i] != 4]
    gt_phrase_start = [gt_phrase_idx[i] for i in range(0, len(gt_phrase_idx)) if
                       gt_phrase_idx[i] != gt_phrase_idx[i - 1] + 1]
    pred_phrase_idx = [i for i in range(len(preds)) if preds[i] != 4]
    pred_phrase_start = [pred_phrase_idx[i] for i in range(0, len(pred_phrase_idx)) if
                         pred_phrase_idx[i] != pred_phrase_idx[i - 1] + 1]
    gt_phrase += len(gt_phrase_start)
    pred_phrase += len(pred_phrase_start)

    for idx in gt_phrase_start:
        flag = True
        while idx in gt_phrase_idx:
            if preds[idx] != labels[idx]:
                flag = False
                break
            idx += 1

        if flag:
            hit += 1

    return gt_phrase, pred_phrase, hit


def train(data, partition):
    # initialization
    words = [{'UNK': 0} for _ in range(NUM_LABEL)]
    start, end = np.zeros([NUM_LABEL]), np.zeros([NUM_LABEL])
    trans = np.zeros([NUM_LABEL, NUM_LABEL])

    # calculate distribution, traverse every word
    for i in range(len(data.docs)):
        for j in range(len(data.docs[i].sentences)):
            for k in range(len(data.docs[i].sentences[j])):
                word = data.docs[i].sentences[j][k]
                label = dict[data.docs[i].labels[j][k]]
                # check and add new word
                if data.docs[i].sentences[j][k] not in words[label].keys():
                    # collect new word from the first subset
                    if i < int(partition * len(data.docs)):
                        new_word(words, data.docs[i].sentences[j][k])
                    # turn word to UNK from the second subset
                    else:
                        word = 'UNK'
                words[label][word] += 1

                if k == len(data.docs[i].sentences[j]) - 1:
                    end[label] += 1
                else:
                    if k == 0:
                        start[label] += 1
                    label_next = dict[data.docs[i].labels[j][k + 1]]
                    trans[label][label_next] += 1

    dists = np.array([pg.DiscreteDistribution(words[i]) for i in range(NUM_LABEL)])

    # normalization
    start, end = start / np.sum(start), end / np.sum(end)
    trans = (trans.T / np.sum(trans, axis=1)).T

    # model
    model = pg.HiddenMarkovModel.from_matrix(trans, dists, start)

    return model, words


def valid(data, model, words):
    total = acc = 0
    gt_phrase = pred_phrase = hit = 0

    for i in range(len(data.docs)):
        for j in range(len(data.docs[i].sentences)):
            sentence_labels = []
            for k in range(len(data.docs[i].sentences[j])):
                word = data.docs[i].sentences[j][k]
                label = dict[data.docs[i].labels[j][k]]
                sentence_labels.append(label)
                if word not in words[0].keys():
                    # change unknown word into 'UNK'
                    # TODO clean the mass
                    tmp_j = list(data.docs[i].sentences)
                    tmp_k = list(tmp_j[j])
                    tmp_k[k] = 'UNK'
                    tmp_k = tuple(tmp_k)
                    tmp_j[j] = tmp_k
                    data.docs[i].sentences = tuple(tmp_j)

            # prediction for every sentence
            sentence = data.docs[i].sentences[j]
            preds = model.predict(sentence, algorithm='map')
            total += len(preds)

            # word accuracy
            acc = word_acc(preds, sentence_labels, acc)

            # sentential accuracy
            gt_phrase, pred_phrase, hit = phrase_acc(preds, sentence_labels, gt_phrase, pred_phrase, hit)

    acc /= total
    precision, recall = hit / pred_phrase, hit / gt_phrase
    entityF1 = 2 * precision * recall / (precision + recall)

    return acc, entityF1


if __name__ == '__main__':
    partitions = [0.05 * i for i in range(1, 20)] + [0.99]
    acc, entityF1 = [], []

    for partition in partitions:
        conll_en = load_conll('./data/conll2003/en/train.txt', label_func=label_func)
        conll_en_valid = load_conll('./data/conll2003/en/valid.txt', label_func=label_func)
        model, words = train(conll_en, partition)
        accuracy, F1 = valid(conll_en_valid, model, words)
        acc.append(accuracy)
        entityF1.append(F1)

    results = DataFrame({"partition": partitions, "Acc": acc, "entity F1": entityF1})

    # plot
    sns.set()
    sns.lineplot(x="partition", y="Acc", data=results)
    sns.lineplot(x="partition", y="entity F1", data=results)
    plt.legend(labels=['Accuracy', 'entity F1'])
    plt.ylabel("")
    plt.show()

    results.to_csv("results.xlsx")
