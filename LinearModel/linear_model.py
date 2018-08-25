"""
0.7977070853129519
0.8114645734352407
0.8400950216897335
0.85449287337327
0.8517041933484817

68.11372590065002
"""
import codecs
import numpy as np
import time


class LinearModel(object):
    def __init__(self, filename):
        self.train_data = []
        self.tags = dict()
        # self.vocabulary = None
        self.features = dict()
        self.w, self.v = None, None

        with codecs.open(filename, "r", "utf-8") as fr:
            data = [line.strip() for line in fr.readlines()]
        tmp, tags_set, feature_set = [("START", "START")], set(), set()
        for line in data:
            if line == "":
                tmp.append(("STOP", "STOP"))
                self.train_data.append(tmp)
                tmp = []
            else:
                s = line.split("\t")
                tmp.append((s[1], s[3]))
                tags_set.add(s[3])
        i = 0
        for tag in tags_set:
            self.tags[tag] = i
            i += 1
        for sent in self.train_data:
            for i in range(1, len(sent)-1):
                feature_set.add("02: %s" % sent[i][0])
                feature_set.add("03: %s" % sent[i-1][0])
                feature_set.add("04: %s" % sent[i+1][0])
                feature_set.add("05: %s %s" % (sent[i][0], sent[i-1][0][-1]))
                feature_set.add("06: %s %s" % (sent[i][0], sent[i+1][0][0]))
                feature_set.add("07: %s" % sent[i][0][0])
                feature_set.add("08: %s" % sent[i][0][-1])
                feature_set.add("09: %s" % sent[i][0][0:-2])
                feature_set.add("10: %s %s" % (sent[i][0][0], sent[i][0][0:-2]))
                feature_set.add("11: %s %s" % (sent[i][0][-1], sent[i][0][0:-2]))
                if len(sent[i][0]) == 1:
                    feature_set.add("12: %s %s %s" % (sent[i][0], sent[i-1][0][-1],sent[i+1][0][0]))
                if sent[i][0][0:-2] == sent[i][0][1:-1]:
                    feature_set.add("13: %s consecutive" % sent[i][0][0:-2])
                feature_set.add("14: %s" % sent[i][0][0:4])
                feature_set.add("15: %s" % sent[i][0][len(sent[i][0])-4:len(sent[i][0])])
        i = 0
        for feature in feature_set:
            self.features[feature] = i
            i += 1
        self.w, self.v = np.zeros((len(tags_set), len(feature_set))), np.zeros((len(tags_set), len(feature_set)))

    def get_f(self, sent, i):
        f = []
        f_02, f_03, f_04 = "02: %s" % sent[i][0], "03: %s" % sent[i-1][0], "04: %s" % sent[i+1][0]
        f_05, f_06 = "05: %s %s" % (sent[i][0], sent[i-1][0][-1]), "06: %s %s" % (sent[i][0], sent[i+1][0][0])
        f_07, f_08, f_09 = "07: %s" % sent[i][0][0], "08: %s" % sent[i][0][-1], "09: %s" % sent[i][0][0:-2]
        f_10, f_11 = "10: %s %s" % (sent[i][0][0], sent[i][0][0:-2]), "11: %s %s" % (sent[i][0][-1], sent[i][0][0:-2])
        f_12 = "12: %s %s %s" % (sent[i][0], sent[i-1][0][-1],sent[i+1][0][0])
        f_13 = "13: %s consecutive" % sent[i][0][0:-2]
        f_14, f_15 = "14: %s" % sent[i][0][0:4], "15: %s" % sent[i][0][len(sent[i][0])-4:len(sent[i][0])]

        if f_02 in self.features:
            f.append(self.features[f_02])
        if f_03 in self.features:
            f.append(self.features[f_03])
        if f_04 in self.features:
            f.append(self.features[f_04])
        if f_05 in self.features:
            f.append(self.features[f_05])
        if f_06 in self.features:
            f.append(self.features[f_06])
        if f_07 in self.features:
            f.append(self.features[f_07])
        if f_08 in self.features:
            f.append(self.features[f_08])
        if f_09 in self.features:
            f.append(self.features[f_09])
        if f_10 in self.features:
            f.append(self.features[f_10])
        if f_11 in self.features:
            f.append(self.features[f_11])
        if len(sent[i][0]) == 1 and f_12 in self.features:
            f.append(self.features[f_12])
        if sent[i][0][0:-2] == sent[i][0][1:-1] and f_13 in self.features:
            f.append(self.features[f_13])
        if f_14 in self.features:
            f.append(self.features[f_14])
        if f_15 in self.features:
            f.append(self.features[f_15])
        return f

    def get_argmax(self, sent, i):
        maxv, maxtag, f = -2147483648, "", self.get_f(sent, i)
        for tag in self.tags:
            dot = 0
            for i in f:
                dot += self.w[self.tags[tag]][i]
            if dot > maxv:
                maxv = dot
                maxtag = tag
        return maxtag, f

    def train(self):
        M = 5
        t1 = time.time()
        for m in range(M):
            for sent in self.train_data:
                for i in range(1, len(sent) - 1):
                    t, f = self.get_argmax(sent, i)
                    if t != sent[i][1]:
                        for ii in f:
                            self.w[self.tags[sent[i][1]]][ii] += 1
                            self.w[self.tags[t]][ii] -= 1
                            # self.v[tag][ii] += self.w[tag][ii]
                            # self.v[t][ii] += self.w[t][ii]
            # print(sum(self.w[self.tags["NN"]]))
            # 评价一次
            print(self.evaluate())
        t2 = time.time()
        print("time cost:", t2 - t1)

    def evaluate(self):
        p, n = 0, 0
        with codecs.open("data/dev.conll", "r", "utf-8") as fr:
            data = [line.strip() for line in fr.readlines()]
        tmp, dev_data = [("START", "START")], []
        for line in data:
            if line == "":
                tmp.append(("STOP", "STOP"))
                dev_data.append(tmp)
                tmp = []
            else:
                s = line.split("\t")
                tmp.append((s[1], s[3]))
        for sent in dev_data:
            for i in range(1, len(sent) - 1):
                if sent[i][1] == self.get_argmax(sent, i)[0]:
                    p += 1
                else:
                    n += 1
        return p / (p+n)


if __name__ == "__main__":
    pass
    linear_model = LinearModel("data/train.conll")
    linear_model.train()