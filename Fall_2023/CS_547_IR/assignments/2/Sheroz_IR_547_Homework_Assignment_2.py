import binarytree
import cs547
import re
from glob import glob

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

def crawl_tree(node, term):
    if not node: return set()
    if ('*' in term and node.key.startswith(term[:-1])) or term == node.key:
        x = node.data
    else: x = set()
    return x.union(crawl_tree(node.left, term)).union(crawl_tree(node.right, term))

class BetterIndex(object):
    WILDCARD: str = '*'
    def __init__(self):
        self._bt = binarytree.binary_tree()
        self._documents: list = []

    def _permute(self, term: str) -> list:
        x: str = term + "$"
        return [x[i:] + x[:i] for i in range(len(x))]

    def _rotate(self, term: str) -> str:
        x: str = term + "$"
        if self.WILDCARD not in term: return x
        n: int = x.index(self.WILDCARD) + 1
        return x[n:] + x[:n]

    def index_dir(self, base_path: str) -> int:
        num_files_indexed: int = 0
        for fn in glob("%s/*" % base_path):
            if fn not in self._documents:
                self._documents.append(fn)
            num_files_indexed += 1
            doc_idx: int = self._documents.index(fn)
            for line in open(file=fn,encoding="utf8",mode="r"):
                for t in self.tokenize(line):
                    for term in self._permute(t):
                        if term not in self._bt:
                            self._bt[term] = set()
                        if doc_idx not in self._bt[term]:
                            self._bt[term].add(doc_idx)
        return num_files_indexed

    def tokenize(self, text: str, is_search=False) -> list:
        if is_search:
            clean_string: str = re.sub('[^a-z0-9 *]', ' ', text.lower())
        else:
            clean_string: str = re.sub('[^a-z0-9 ]', ' ', text.lower())
        tokens = clean_string.split()
        return tokens

    def pre_process_text(self, txt: str) -> list:
        return [self._rotate(x) for x in self.tokenize(text=txt,is_search=True)]

    def wildcard_search_or(self, text: str) -> list:
        current_doc_ids: set = set()
        for item in self.pre_process_text(txt=text):
          current_doc_ids: set = current_doc_ids.union(crawl_tree(node=self._bt.root,term=item))
        return [self._documents[x] for x in sorted(current_doc_ids)] if current_doc_ids else []

    def wildcard_search_and(self, text: str) -> list:
        if len(text)==0:
          return [self._documents[x] for x in crawl_tree(node=self._bt.root,term=item)]
        else:
          current_tokens: list = self.pre_process_text(txt=text)
          current_doc_ids: set = crawl_tree(node=self._bt.root,term=current_tokens[0])
          for item in current_tokens[1:]:
            current_doc_ids: set = current_doc_ids.intersection(crawl_tree(node=self._bt.root,term=item))
          return [self._documents[x] for x in sorted(current_doc_ids)] if current_doc_ids else []

def main(args):
    print(student)
    index = BetterIndex()
    print("starting indexer")
    num_files = index.index_dir('data/')
    print("indexed %d files" % num_files)

    for term in ('hel*o', 'aggies', 'agg*', 'mike sherm*', 'dot cat'):
        results = index.wildcard_search_or(term)
        print("OR  searching: %s -- results: %s" % (term, ", ".join(results)))
        results = index.wildcard_search_and(term)
        print("AND searching: %s -- results: %s" % (term, ", ".join(results)))

if __name__ == "__main__":
    import sys
    main(sys.argv)
    del main,crawl_tree,BetterIndex,student,MY_NAME,MY_ANUM,MY_EMAIL,COLLABORATORS,I_AGREE_HONOR_CODE
    del binarytree,re,cs547,glob
