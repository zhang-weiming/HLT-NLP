"""
条件概率P(ti|ti-1)叫作隐状态之间的 转移概率 。
条件概率P(wi|ti)叫作隐状态到显状态的 发射概率 ，也叫作隐状态生成显状态的概率。
"""
import codecs
import pickle
import sys


Alpha = 0.3


def count(tag, tags):
    c = 0
    for t in tags:
        c += t.count(tag)
    return c


def hmm(words_with_tag, tags, V, S):
    Q, E, count_tag, len_V = dict(), dict(), dict( (("*", len(tags)),) ), len(V)
    for tag in S:
        if tag not in count_tag:
            count_tag[tag] = count(tag, tags)
        a = count("*" + tag, tags) + Alpha
        b = count_tag["*"] + Alpha * len_V
        Q[tag + "*"] = a / b

        for prefix in S:
            if prefix not in count_tag:
                count_tag[prefix] = count(prefix, tags)
            a = count(prefix + tag, tags) + Alpha
            b = count_tag[prefix] + Alpha * len_V
            Q[tag + prefix] = a / b
    for word in V:
        for tag in S:
            a = words_with_tag.count("%s %s" % (word, tag)) + Alpha
            b = count_tag[tag] + Alpha * len_V
            E["%s %s" % (word, tag)] = a / b

    return Q, E


def train():
    print("train...", end="")
    sys.stdout.flush()
    with codecs.open("data/train.conll", "r", "utf-8") as fr:
        data = fr.readlines()
        words_with_tag, tags, V, S, tags_tmp = [], [], set(), set(), "*"
        for line in data:
            if line.strip() == "":
                tags.append(tags_tmp[:-1])
                tags_tmp = "*"
            else:
                s = line.split("\t")
                words_with_tag.append("%s %s" % (s[1], s[3]))
                V.add(s[1])
                S.add(s[3])
                tags_tmp += s[3] + " "

    Q, E = hmm(words_with_tag, tags, V, S)

    with codecs.open("data/model.bin", "wb") as fw:
        pickle.dump(Q, fw)
        pickle.dump(E, fw)
    print("done")
    return Q, E


def load_model():
    with codecs.open("data/model.bin", "rb") as fr:
        Q = pickle.load(fr)
        E = pickle.load(fr)
    return Q, E


def main():
    # Q, E = train()
    Q, E = load_model()
    P = []
    with codecs.open("data/train.conll", "r", "utf-8") as fr:
        data = fr.readlines()
        words_with_tag, tags, words_with_tag_tmp, tags_tmp = [], [], [], ["*"]
        for line in data:
            if line.strip() == "":
                p = 1
                for word_with_tag in words_with_tag_tmp:
                    p *= E[word_with_tag]
                for i in range(len(tags_tmp)-1):
                    p *= Q[tags_tmp[i+1] + tags_tmp[i+1]]
                print(p)
                P.append(p)
                tags_tmp = ["*"]
            else:
                s = line.split("\t")
                words_with_tag_tmp.append("%s %s" % (s[1], s[3]))
                tags_tmp.append(s[3])
    average = sum(P) / len(P)
    var = sum([(x - average)**2 for x in P]) / len(P)
    print(average, var)


if __name__ == "__main__":
    main()
