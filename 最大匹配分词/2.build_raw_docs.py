import codecs


def main(sourcefile="data/data.conll", dictfile="data/data.txt"):
    """
    将data.conll文件中的格式修改为：每行一句话，词语之间无空格，起名为data.txt
    :param sourcefile: 人工分好词的文件路径
    :param dictfile: 由分好词的文件生成原始文本，保存文件
    :return: 无
    """
    with codecs.open(sourcefile, "r", "utf-8") as fr:
        data = fr.readlines()

    while data[-2].strip() == "":
        data.pop(index=-1)

    with codecs.open(dictfile, "w", "utf-8") as fw:
        tmp = ""
        c = 0
        for line in data:
            if line.strip() == "":
                fw.write("%s\n" % tmp)
                c += 1
                tmp = ""
            else:
                tmp += line.strip().split("\t")[1]

    print("We have %d docs in the file saved as '%s'." % (c, dictfile))


if __name__ == "__main__":
    main()
