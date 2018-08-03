import codecs


def main(datafile="data/data.conll", outfile="data/data.out"):
    with codecs.open(datafile, "r", "utf-8") as fr:
        data = [line.strip().split("\t")[1] for line in fr.readlines() if line.strip() != ""]
    with codecs.open(outfile, "r", "utf-8") as fr:
        lines, out = fr.readlines(), []
        for item in lines:
            out.extend(item.strip().split(" "))
    print(data, "\n", out)
    len_data, len_out = len(data), len(out)

    tp, i, j = 0, 0, 0
    while i < len_data and j < len_out:
        if data[i] == out[j]:
            tp += 1
        else:
            skip_data, skip_out = data[i], out[j]
            while len(skip_data) != len(skip_out):
                if len(skip_data) > len(skip_out):
                    j += 1
                    skip_out += out[j]
                else:
                    i += 1
                    skip_data += data[i]
        i += 1
        j += 1
    precision = tp / len_out
    recall = tp / len_data
    f = precision * recall * 2 / (precision + recall)
    print("Precision: %f" % precision)
    print("Recall: %f" % recall)
    print("F: %f" % f)


if __name__ == "__main__":
    main()
