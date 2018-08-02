import codecs


def main(sourcefile="data/data.conll", dictfile="data/word.dict"):
    """
    给一个人工分好词的文件data.conll，构建一个词典，输出到一个文件中，起名为word.dict
    :param sourcefile: 人工分好词的文件路径
    :param dictfile: 构建的词典文件路径
    :return: 无
    """
    with codecs.open(sourcefile, "r", "utf-8") as fr:
        data = fr.readlines()
    dict = set()
    for line in data:
        if line.strip() != "":
            word = line.strip().split("\t")[1]
            if len(word) > 1:
                dict.add(word)

    with codecs.open(dictfile, "w", "utf-8") as fw:
        for word in dict:
            fw.write("%s\n" % word)
    print("We have %d words in the dict file saved as '%s'." % (len(dict), dictfile))


if __name__ == "__main__":
    main()
