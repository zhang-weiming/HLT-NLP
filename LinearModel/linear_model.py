import codecs
import numpy as np


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
        f.append(self.features["02: %s" % sent[i][0]])
        f.append(self.features["03: %s" % sent[i-1][0]])
        f.append(self.features["04: %s" % sent[i+1][0]])
        f.append(self.features["05: %s %s" % (sent[i][0], sent[i-1][0][-1])])
        f.append(self.features["06: %s %s" % (sent[i][0], sent[i+1][0][0])])
        f.append(self.features["07: %s" % sent[i][0][0]])
        f.append(self.features["08: %s" % sent[i][0][-1]])
        f.append(self.features["09: %s" % sent[i][0][0:-2]])
        f.append(self.features["10: %s %s" % (sent[i][0][0], sent[i][0][0:-2])])
        f.append(self.features["11: %s %s" % (sent[i][0][-1], sent[i][0][0:-2])])
        if len(sent[i][0]) == 1:
            f.append(self.features["12: %s %s %s" % (sent[i][0], sent[i-1][0][-1],sent[i+1][0][0])])
        if sent[i][0][0:-2] == sent[i][0][1:-1]:
            f.append(self.features["13: %s consecutive" % sent[i][0][0:-2]])
        f.append(self.features["14: %s" % sent[i][0][0:4]])
        f.append(self.features["15: %s" % sent[i][0][len(sent[i][0])-4:len(sent[i][0])]])
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
        M = 10
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
            print(sum(self.w[self.tags["NN"]]))
            # 评价一次
            print(self.evaluate())

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
        # for sent in self.train_data:
        #     for i in range(1, len(sent) - 1):
        #         if sent[i][1] == self.get_argmax(sent, i)[0]:
        #             p += 1
        #         else:
        #             n += 1
        return p / (p+n)


if __name__ == "__main__":
    pass
    linear_model = LinearModel("data/train.conll")
    linear_model.train()