"""
条件概率P(ti|ti-1)叫作隐状态之间的 转移概率 。
条件概率P(wi|ti)叫作隐状态到显状态的 发射概率 ，也叫作隐状态生成显状态的概率。

Alpha   Acc
0.2     0.7381
0.1     0.7589
0.05    0.773525
0.01    0.788609
0.005   0.790238
"""
import codecs
import pickle
import sys
import numpy as np


Alpha = 0.005


class Binary_HMM():
    def __init__(self):
        self.train_data = None

        self.launch_matrix = None
        self.transition_matrix = None
        self.wordbank = dict()
        self.tagbank = dict()

    def load_train_set(self, filename):
        with codecs.open(filename, "r", "utf-8") as fr:
            data = fr.readlines()

        self.train_data, tmp, words, tags = [], [], set(), set()
        for line in data:
            if line.strip() == "":
                self.train_data.append(tmp)
                tmp = []
            else:
                s = line.split("\t")
                tmp.append((s[1], s[3]))
                words.add(s[1])
                tags.add(s[3])
        i = 0
        for word in words:
            self.wordbank[word] = i
            i += 1
        i = 0
        for tag in tags:
            self.tagbank[tag] = i
            i += 1
        self.tagbank["*"] = len(self.tagbank)
        self.tagbank["STOP"] = len(self.tagbank)

    def train(self):
        self.launch_matrix = np.zeros((len(self.wordbank), len(self.tagbank)))
        for sent in self.train_data:
            for word, tag in sent:
                self.launch_matrix[self.wordbank[word]][self.tagbank[tag]] += 1
        for i in range(len(self.launch_matrix)):
            s = sum(self.launch_matrix[i])
            for j in range(len(self.launch_matrix[i])):
                self.launch_matrix[i][j] = (self.launch_matrix[i][j] + Alpha) / (s + Alpha * (len(self.wordbank)))

        self.transition_matrix = np.zeros((len(self.tagbank), len(self.tagbank)))
        for sent in self.train_data:
            for i in range(len(sent) + 1):
                if i == 0:
                    self.transition_matrix[self.tagbank[sent[i][1]]][self.tagbank["*"]] += 1
                elif i == len(sent):
                    self.transition_matrix[self.tagbank["STOP"]][self.tagbank[sent[i-1][1]]] += 1
                else:
                    self.transition_matrix[self.tagbank[sent[i][1]]][self.tagbank[sent[i-1][1]]] += 1
        for i in range(len(self.transition_matrix)):
            s = sum(self.transition_matrix[i])
            for j in range(len(self.transition_matrix[i])):
                self.transition_matrix[i][j] = (self.transition_matrix[i][j] + Alpha) / (
                        s + Alpha * (len(self.tagbank) - 1))

    def viterbi(self, words):
        n = len(words)
        Y, PI, BP = ["" for i in range(n+1)], [dict() for i in range(n+1)], [dict() for i in range(n+1)]
        PI[0]["*"] = 1
        words.insert(0, "")

        for k in range(1, n+1):
            for v in self.tagbank:
                maxv, max_tag, e = 0.0, "", self.launch_matrix[self.wordbank[words[k]]][self.tagbank[v]] if words[k] in self.wordbank else 1
                if k == 1:
                    for u in ("*"):
                        tmp = PI[k-1][u] * self.transition_matrix[self.tagbank[v]][self.tagbank[u]] * e
                        if tmp > maxv:
                            maxv = tmp
                            max_tag = u
                    PI[k][v] = maxv
                    BP[k][v] = max_tag
                else:
                    for u in self.tagbank:
                        tmp = PI[k-1][u] * self.transition_matrix[self.tagbank[v]][self.tagbank[u]] * e
                        if tmp > maxv:
                            maxv = tmp
                            max_tag = u
                    PI[k][v] = maxv
                    BP[k][v] = max_tag

        maxv, max_tag = 0.0, ""
        for v in self.tagbank:
            tmp = PI[n][v] * self.transition_matrix[self.tagbank["STOP"]][self.tagbank[v]]
            if tmp > maxv:
                maxv = tmp
                max_tag = v
        Y[n] = max_tag
        for k in range(n-1, 0, -1):
            Y[k] = BP[k+1][Y[n]]
        return Y[1:]

    def save_model(self):
        with codecs.open("data/model.bin", "wb") as fw:
            pickle.dump(self.launch_matrix, fw)
            pickle.dump(self.transition_matrix, fw)
            pickle.dump(self.wordbank, fw)
            pickle.dump(self.tagbank, fw)

    def evaluate(self):
        with codecs.open("data/dev.conll", "r", "utf-8") as fr:
            data = fr.readlines()

        words, tags, p, n = [], [], 0, 0
        for line in data:
            if line.strip() == "":
                pred = self.viterbi(words)
                for i in range(len(tags)):
                    if tags[i] == pred[i]:
                        p += 1
                    else:
                        n += 1
                words, tags = [], []
            else:
                s = line.split("\t")
                words.append(s[1])
                tags.append(s[3])

        print("p: %d, n: %d" % (p, n))
        print("Acc: %f" % (p / (p + n)))

    def test(self):
        with codecs.open("data/train.conll", "r", "utf-8") as fr:
            data = fr.readlines()

        tmp, words, tags = [], set(), set()
        for line in data:
            if line.strip() == "":
                p = 1
                for word, tag in tmp:
                    p *= self.launch_matrix[self.wordbank[word]][self.tagbank[tag]]
                for i in range(len(tmp) + 1):
                    if i == 0:
                        p *= self.transition_matrix[self.tagbank[tmp[i][1]]][self.tagbank["*"]]
                    elif i == len(tmp):
                        p *= self.transition_matrix[self.tagbank["STOP"]][self.tagbank[tmp[i-1][1]]]
                    else:
                        p *= self.transition_matrix[self.tagbank[tmp[i][1]]][self.tagbank[tmp[i-1][1]]]
                print(p, tmp)
                tmp = []
            else:
                s = line.split("\t")
                tmp.append((s[1], s[3]))
                words.add(s[1])
                tags.add(s[3])


def main():
    hmm = Binary_HMM()
    hmm.load_train_set("data/train.conll")
    hmm.train()
    hmm.evaluate()


if __name__ == "__main__":
    main()
