import cs547
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re

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

class PageRankIndex(object):

    def __init__(self):
        self.alpha = 0.1
        self.limit1 = 300
        self.absolute_path = 'http://web.cs.wpi.edu/~kmlee/cs547/new10/'
        self.visited_urls = []
        self.key_terms = {}
        self.url_graph = {}
        self.url_graph_cleaned = {}
        self.transition_matrix = {}
        self.teleporting_matrix = {}
        self.matrix_p = {}
        self.initial_x = {}
        self.new_x = {}
        self.page_rank_index = {}

    def page_rank_cal(self)->None:
        for k,v in self.url_graph.items():
            if 'index.html' in k:
                continue
            else:
                k = k.replace(self.absolute_path,'').replace('.html','')
                vs = list()
                for a in v:
                    vs.append(a.replace(self.absolute_path,'').replace('.html',''))
                self.url_graph_cleaned[k]=vs

        page_initial_positions = {k:v for v,k in enumerate(self.url_graph_cleaned.keys(),start=0,)}
        total_pages = len(page_initial_positions)
        link_matrix = np.zeros(shape=(total_pages,total_pages))

        for i,v in self.url_graph_cleaned.items():
            row_index = page_initial_positions[i]
            for j in v:
                column_index = page_initial_positions[j]
                link_matrix[row_index,column_index] = 1

        row_sum = (link_matrix.sum(axis=1))[:,np.newaxis]
        self.transition_matrix = (link_matrix / row_sum)
        self.transition_matrix = ((1-self.alpha) * self.transition_matrix)
        self.teleporting_matrix = np.full(shape=self.transition_matrix.shape,fill_value=self.alpha)
        self.matrix_p = self.transition_matrix + self.teleporting_matrix
        self.initial_x = np.zeros(shape=self.matrix_p.shape[0])
        self.initial_x[0] = 1

        i = 0
        while i < self.limit1:
            self.new_x = np.dot(self.initial_x,self.matrix_p)
            if np.allclose(a=self.initial_x,b=self.new_x,rtol=1e-05,atol=1e-05,equal_nan=True):
                break
            else:
                self.initial_x = self.new_x
            i += 1

        self.page_rank_index = {self.visited_urls.index(self.absolute_path+k+'.html'):v for k,v in zip(self.url_graph_cleaned.keys(),self.new_x)}
        return None

    def index_url(self,url: str) -> int:
        open_urls = [url]
        while True:
            if len(open_urls)==0:
                break
            current_url = open_urls[0]
            del open_urls[0]
            if current_url not in self.visited_urls:
                self.url_graph[current_url] = list()
                self.visited_urls.append(current_url)
                doc_id = self.visited_urls.index(current_url)
                parsed_content = BeautifulSoup(urlopen(current_url),'html.parser')
                for url1 in parsed_content.find_all('a'):
                    url1 = self.absolute_path + str(url1).replace('<a href="','').split('">')[0]
                    open_urls.append(url1)
                    self.url_graph[current_url].append(url1)
                for item1 in parsed_content.contents:
                    item1 = str(item1)
                    if '<a' in item1 and '</a>' in item1:
                        continue
                    else:
                        item1 = item1.replace('\n','').replace('\r','').replace('\t','').strip()
                        for item2 in set(self.tokenize(item1)):
                            if (str(item2)):
                                if item2 not in self.key_terms:
                                    self.key_terms[item2] = [doc_id]
                                else:
                                    if doc_id not in self.key_terms[item2]:
                                        self.key_terms[item2].append(doc_id)
                                    else:
                                        pass
                            else:
                                pass
            else:
                continue

        self.page_rank_cal()
        return len(self.visited_urls)

    def tokenize(self,txt: str) -> list:
        return re.sub(pattern=r'\s+',repl=' ',string=re.sub(pattern=r'[^a-z\d]+',repl=' ',string=str(txt).lower())).strip().split(' ')

    def sort_ranks(self,l1:list) -> list:
        l1.sort(key=lambda x: x[1],reverse=True)
        return l1

    def ranked_search(self,text : str) -> list:
        tokenized_txt : list = self.tokenize(txt=text)
        first_char : str = tokenized_txt[0]
        del tokenized_txt[0]
        available_items : set = set()
        if first_char in self.key_terms:
            for j in self.key_terms[first_char]:
                available_items.add(j)
        else:
            return []

        for i in tokenized_txt:
            if i in self.key_terms:
                temp_items : set = set()
                for j in self.key_terms[i]:
                    temp_items.add(j)
                temp2_items: set = available_items.intersection(temp_items)
                if temp2_items:
                    available_items: set = available_items.intersection(temp2_items)
                else:
                    return []
            else:
                return []
 
        if available_items:
            available_items = self.sort_ranks(l1=[(self.visited_urls[x],self.page_rank_index[x]) for x in available_items])

            if len(available_items)>10:
                available_items = available_items[:10]
            else:
                pass
            return available_items
        else:
            return []

def main(args):
    print(student)
    index = PageRankIndex()
    url = 'http://web.cs.wpi.edu/~kmlee/cs547/new10/index.html'
    num_files = index.index_url(url)
    print(f"indexed {num_files} files")
    search_queries = ('palatial','college ','palatial college','college supermarket','famous aggie supermarket')

    for q in search_queries:
        results = index.ranked_search(q)
        print(f"searching: {q} -- results: {results}")

if __name__ == "__main__":
    import sys
    main(sys.argv)



