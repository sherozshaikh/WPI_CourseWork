import cs547
import PorterStemmer
import re
import glob

MY_NAME: str = "Sheroz Shaikh"
MY_ANUM: int = 901014725
MY_EMAIL: str = "sshaikh@wpi.edu"
COLLABORATORS: list = list()
I_AGREE_HONOR_CODE: bool = True

student = cs547.Student(
    MY_NAME,
    MY_ANUM,
    MY_EMAIL,
    COLLABORATORS,
    I_AGREE_HONOR_CODE
)


class Index(object):

    def __init__(self):
        self._inverted_index = {}
        self._documents = []

    def index_dir(self, base_path: str) -> int:
        num_files_indexed: int = 0
        for filename in glob.glob(base_path + '*'):
            infile = open(file=filename, mode='r', encoding='utf-8')
            filename: str = filename.replace(base_path, '')
            if filename not in self._documents:
                self._documents.append(filename)
            else:
                pass
            filename_index: int = self._documents.index(filename)
            line = infile.readlines()
            for c in line:
                for a in self.stemming(tokens=self.tokenize(text=c)):
                    if a in self._inverted_index:
                        if filename_index in self._inverted_index[a]:
                            pass
                        else:
                            self._inverted_index[a].append(filename_index)
                    else:
                        self._inverted_index[a] = [filename_index]
            infile.close()
            num_files_indexed += 1
        return num_files_indexed

    def tokenize(self, text: str) -> list:
        return re.sub(pattern=r'\s+', repl=' ', string=re.sub(pattern=r'[^a-z\d]+', repl=' ',
                                                              string=str(
                                                                  text).lower())).strip().split(' ')

    def stemming(self, tokens: list) -> list:
        return [PorterStemmer.PorterStemmer().stem(x, 0, len(x) - 1) for x in tokens]

    def process_1(self, txt: str) -> set:
        results: list = self._inverted_index.get(self.stemming(tokens=self.tokenize(text=txt))[0], [])
        results: set = set([self._documents[x] for x in results]) if results else set()
        return results

    def boolean_search(self, text: str) -> list:
        text: list = str(text).lower().strip().split(' ')
        if len(text) == 1:
            results: list = list(self.process_1(txt=text[0]))
        else:
            if 'or' == text[1]:
                results: list = list((self.process_1(txt=text[0])).union((self.process_1(txt=text[2]))))
            elif 'and' == text[1]:
                results: list = list((self.process_1(txt=text[0])).intersection((self.process_1(txt=text[2]))))
            else:
                results: list = []
        return results


def main(args):
    print(student)
    index = Index()
    print("starting indexer")
    num_files = index.index_dir('./data/')
    print("indexed %d files" % num_files)
    for term in ('football', 'mike', 'sherman', 'mike OR sherman', 'mike AND sherman','histori AND montenegro'):
        results = index.boolean_search(term)
        print("searching: %s -- results: %s" % (term, ", ".join(results)))


if __name__ == "__main__":
    import sys

    main(sys.argv)
