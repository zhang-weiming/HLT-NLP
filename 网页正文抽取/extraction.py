import codecs
from bs4 import BeautifulSoup


def extract(filename):
    with codecs.open(filename, "r", "utf-8") as fr:
        html = fr.read()
    soup = BeautifulSoup(html, "html")

    with codecs.open("%s.data" % filename, "w", "utf-8") as fw:
        title, body, links = soup.find("title").get_text(), soup.find("body").get_text().split("\n"), soup.find_all("a")

        i = 0
        while i < len(body):
            if body[i].strip() == "":
                i += 1
                while i < len(body) and body[i].strip() == "":
                    body.pop(i)
            i += 1

        fw.write("title:\n%s\n" % title)
        fw.write("body:\n%s\n" % "\n".join(body))
        fw.write("link:\n")
        for link in links:
            fw.write("%s %s\n" % (link.get_text(), link.get("href")))


def main():
    extract("data/1.html")
    extract("data/2.html")


if __name__ == "__main__":
    main()
