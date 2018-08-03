import codecs


def forward_maximum_matching(sentences, word_dict, max_word_len=3):
    for i, item in enumerate(sentences):
        item = item.strip()
        s, e, tmp = 0, min(max_word_len, len(item)), []
        while s < len(item):
            if item[s:e] in word_dict or s + 1 == e:
                tmp.append(item[s:e])
                s = e
                e = min(s + max_word_len, len(item))
            else:
                e -= 1
        sentences[i] = " ".join(tmp)
        print("\r%d / %d" % (i+1, len(sentences)), end="")
    print()
    return sentences


def backward_maximum_matching(sentences, word_dict, max_word_len=3):
    for i, item in enumerate(sentences):
        item = item.strip()
        s, e, tmp = max(len(item)-max_word_len, 0), len(item), []
        while s >= 0:
            if item[s:e] in word_dict or s + 1 == e:
                tmp.append(item[s:e])
                e = s
                s = max(e - max_word_len, 0) if s != 0 else -1
            else:
                s += 1
        tmp.reverse()
        sentences[i] = " ".join(tmp)
        print("\r%d / %d" % (i+1, len(sentences)), end="")
    print()
    return sentences


def main(sourcefile="data/data.txt", dictfile="data/word.dict", outfile="data/data.out"):
    with codecs.open(sourcefile, "r", "utf-8") as fr:
        sentences = fr.readlines()

    with codecs.open(dictfile, "r", "utf-8") as fr:
        word_dict = [item.strip() for item in fr.readlines()]
    # sentences = forward_maximum_matching(sentences, word_dict, max_word_len=10)
    sentences = backward_maximum_matching(sentences, word_dict, max_word_len=10)

    with codecs.open(outfile, "w", "utf-8") as fw:
        for item in sentences:
            fw.write("%s\n" % item)
    print("%d docs have been cut." % len(sentences))


if __name__ == "__main__":
    main()
