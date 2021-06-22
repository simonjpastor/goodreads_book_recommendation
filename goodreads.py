#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[225]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


# In[226]:


#books = pd.read_csv("goodreads_library_export2.csv")

books = pd.read_csv("goodreads_library_export3.csv")


# In[227]:


books.columns


# In[228]:


fig = plt.figure(figsize=(18,9))
ax = fig.add_subplot(111)
ax.set_title("# of Books in Library per Author",fontsize=20)
order = books["Author"].value_counts().head(10).index
sns.countplot(x=books["Author"],order=order)
ax.set_xlabel('Author',fontsize = 15) #xlabel
ax.set_ylabel('Count', fontsize = 15)
fig.savefig('books_author.png')


# In[229]:


books["Year Published"].fillna(2001,inplace=True)
books["Year Published"] = books["Year Published"].astype(int)


# In[230]:


books.nlargest(10,"Number of Pages")[["Title","Author","Number of Pages"]]


# In[232]:


fig = plt.figure(figsize=(60,24))
ax = fig.add_subplot(111)
ax.set_title("# of Books per Year Published",fontsize=60)
sns.histplot(books["Original Publication Year"],binwidth=2,color="purple",element="poly",cumulative=False)
ax.set_xlabel('Year Published',fontsize = 20) #xlabel
ax.set_ylabel('Count', fontsize = 40)
plt.xlim(1650,2022)
plt.xticks(ticks=[1675,1700,1725, 1750,1775, 1800,1825, 1850,1875,1900,1925,1950,1975, 2000, 2015],fontsize=40)
plt.yticks(ticks=[0,5,10,15,20,25,30],fontsize=40)


ax.xaxis.label.set_size(50)
ax.yaxis.label.set_size(50)
plt.show()
fig.savefig('books_year.png')


# In[233]:


books["Original Publication Year"].fillna("0",inplace=True)


# In[234]:


for i in range(0,len(books["Title"])):
    books["Original Publication Year"][i] = int(books["Original Publication Year"][i] )


# In[235]:


year_order = books.sort_values("Original Publication Year")["Original Publication Year"].unique()


# In[236]:


fig = plt.figure(figsize=(60,24))
ax = fig.add_subplot(111)
ax.set_title("# of Books per Year Published",fontsize=20))
sns.countplot(books["Original Publication Year"], order=year_order)
ax.set_xlabel('Year Published',fontsize = 15) #xlabel
ax.set_ylabel('Count', fontsize = 15)
plt.xlim(1,137)
plt.xticks(ticks=[])


# In[237]:


books = books.drop(['Recommended For', 'Recommended By', 'Owned Copies',"Additional Authors",
       'Original Purchase Date', 'Original Purchase Location', 'Condition',
       'Condition Description', 'BCID',"Bookshelves","Bookshelves with positions","My Review","Author l-f","ISBN13","Spoiler","Private Notes"],axis="columns")


# In[238]:


id1 = []
for i in books["ISBN"]: 
    i = i[2:]
    i =(i[:-1])
    id1.append(i)


# In[239]:


books[books["Original Publication Year"] != 0].sort_values("Original Publication Year")


# # Adding Variables

# ## Google API

# In[ ]:


import requests #we import the requests library (to simulate our HTTP request)
categories = []
ratings_count = []
for i in id1: #id1 contains all of the ISBNs, we thus iterate over the list to get the information for each book
    if len(i) > 0: 
        soup = requests.get(f"https://www.googleapis.com/books/v1/volumes?q=isbn:{i}").json() 
        #this returns our result in a json format
        try:
            #because some of the books do not seem to have ratingsCount or Categorie, they return Errors. 
            #We use try and except to prevent the entire program from crashing every time one of the books does not 
            #have a ratingsCount or Categorie.
            ratings_count.append(soup["items"][0]["volumeInfo"]["ratingsCount"]) #This returns the book's RatingCount
        except KeyError:
            ratings_count.append(0) #We add 0 if there are none to keep the order of our books intact
        try:
            categories.append(soup["items"][0]["volumeInfo"]["categories"][0].lstrip().lower()) #returns the Categorie
        except KeyError:
            categories.append(0)
    else:
        ratings_count.append(0)
        categories.append(0)


# In[ ]:


books["Reviews"] = books["Publisher"]
books["Categories"] = books["Publisher"]


# In[ ]:


for i in range(0,len(books["Reviews"])):
    books["Reviews"][i] = ratings_count[i]
    books["Categories"][i] = categories[i]


# In[ ]:


books[["Title","Author","Reviews","Categories"]]


# In[ ]:


(249)/441


# In[ ]:


books[books["Categories"]==0]


# # Scrape

# Two options: Scraping as done below, or with api and the following link :) https://www.goodreads.com/book/isbn/0446694975?key=xQXvrwOTLq7xonOLcjt2A

# In[ ]:


import requests #We'll use requests to simulate our browser's HTML request
from bs4 import BeautifulSoup #We'll use BeautifulSoup to parse the HTML 
ids=[]
for i in books["Book Id"]:
    ids.append(i) #obtaining the book's individual IDs


# In[ ]:


def url_maker():#1st we have to generate the Book's URL by adding its ID
    for i in ids:
        url = f'https://www.goodreads.com/book/show/{i}'
        soup_maker(url) #we then call this function for each URL 


# In[ ]:


more_categories = []
def soup_maker(url): #this function collects the book's genres 
    prov_list = []
    response = requests.get(url) #we get the HTML code
    soup = BeautifulSoup(response.content, "html.parser") #we parse it
    for i in soup.find_all(class_="elementList"): #where the genres are-
        prov_list.append(i.find('a').text) #-located in the HTML code
    prov_list = list(dict.fromkeys(prov_list)) #remove list duplicates
    prov_list = list(filter(None, prov_list)) #remove empty values
    more_categories.append(prov_list)
    return more_categories #we return our list


# In[ ]:


url_maker() #we now simply have to call our 1st function


# In[ ]:


more_categories


# In[ ]:


response = requests.get("https://www.goodreads.com/book/show/5111")
soup = BeautifulSoup(response.content, "html.parser")
print(soup) #simply looking at one of the books that generated
#empty categories, we realize that the page's HTML code was very weird


# # API

# In[54]:


import requests #We'll use requests to simulate our browser's HTML request
from bs4 import BeautifulSoup
ids=[]
for i in books["Book Id"]:
    ids.append(i) #obtaining the book's individual IDs


# In[55]:


genreExceptions = [
'to-read', 'currently-reading', 'owned', 'default', 'favorites', 'books-i-own',
'ebook', 'kindle', 'library', 'audiobook', 'owned-books', 'audiobooks', 'my-books',
'ebooks', 'to-buy', 'english', 'calibre', 'books', 'british', 'audio', 'my-library',
'favourites', 're-read', 'general', 'e-books',"read-in-2020"
] #ignore these different bookshelves
genres = []


# In[56]:


def get_xml(): #obtaining the bookshelves in which the book is included
    for i in ids: 
        test = requests.get(f"https://www.goodreads.com/book/show/{i}?key=xQXvrwOTLq7xonOLcjt2A",allow_redirects=False)
        test = BeautifulSoup(test.content, "lxml")
        shelves = test.find("popular_shelves")
        finding_genres(shelves)
    return genres


# In[57]:


def finding_genres(shelves): #filtering the bookshelves to obtain only
    prov_genres = [] #the first 8 results
    try: 
        for i in shelves:
            if len(i) == 0:
                x = i.attrs["name"]
                if x not in genreExceptions:
                    if len(prov_genres) < 8:
                        x = x.replace("non-fiction","nonfiction").lower()
                        prov_genres.append(x)
                        prov_genres = list(dict.fromkeys(prov_genres))
        genres.append(prov_genres)
    except TypeError:
        genres.append([])
    return genres


# In[58]:


get_xml() #we now simply have to call our 1st function


# In[241]:


len(genres)


# In[242]:


books = books.reset_index()


# In[243]:


genres_df = pd.DataFrame({"Genres":genres}).reset_index()


# In[244]:


genres_df.merge(books,on="index")


# # Viz

# In[ ]:


#books["Genres"].value_counts().head(10).plot(kind="bar")


# # Encoding

# In[245]:


df = pd.get_dummies(pd.DataFrame(genres))
df.columns = df.columns.str.split("_").str[-1]


# In[246]:


df1 = pd.DataFrame({"index":np.arange(0,441)})


# In[247]:


genres_dupli = []


# In[248]:


for i in df.columns:
    if i not in genres_dupli:
        try: 
            df1[i]= pd.DataFrame(df[i].sum(axis=1))
            genres_dupli.append(i)
        except ValueError:
            df1[i] = pd.DataFrame(df[i])
            genres_dupli.append(i)


# In[249]:


df1


# ## Useless

# In[100]:


cat_count = {}
for i in df1.columns:
    cat_count[i] = df1[i].sum()


# In[101]:


cat_count["Original Publication Year"] = 0
cat_count["Number of Pages"] = 0
cat_count["Average Rating"] = 0


# In[106]:


cat_count.pop('key', None)


# In[171]:


len(keys)


# In[107]:


keys = {k: v for k, v in sorted(cat_count.items(), key=lambda item: item[1], reverse=True)}.keys()


# In[250]:


# creating initial dataframe
categories_types2 = keys
categories_types = ('biography & autobiography', 'games', 0, 'sports & recreation',
       'history', 'science', 'self-help', 'business & economics',
       'mathematics', 'computers', 'literary collections', 'families',
       'political science', 'nature', 'education', 'fiction',
       'social science', 'design', 'philosophy',
       'language arts & disciplines', 'body, mind & spirit',
       'family & relationships', 'juvenile fiction', 'equality',
       'leadership', 'medical', 'psychology', 'dictators',
       'man-woman relationships', 'existentialism', 'drama',
       'cross-country running', 'poetry', 'cocaine', 'suicide',
       'immigrants', 'short stories', 'children of holocaust survivors',
       'english language', 'france', 'psychological fiction', 'religion',
       'business', 'self-acceptance', 'civilization', 'humor',
       'buchenwald (concentration camp)', 'war',
       'pilgrims (new plymouth colony)', 'literature',
       'juvenile nonfiction', 'comics & graphic novels',
       'adventure and adventurers', 'animal rights activists',
       'human trafficking', 'communism')
books_2 = pd.DataFrame(categories_types, columns=['Categories'])
# generate binary values using get_dummies
dum_df = pd.get_dummies(books_2, columns=["Categories"], prefix=["Type_is"] )
# merge with main df bridge_df on key values


# ## Other Analysis

# In[251]:


books = books.reset_index()


# In[252]:


books = books.merge(df1,on="index")


# In[253]:


mira = books[books["Exclusive Shelf"] == "read"][["My Rating","Exclusive Shelf"]]


# In[254]:


books_read = books[books["Exclusive Shelf"] == "read"]


# In[255]:


books_read["Publisher"].fillna("0",inplace=True)
books_read["Original Publication Year"].fillna("0",inplace=True)
books_read["Number of Pages"].fillna("0",inplace=True)
books_read["Year Published"].fillna("0",inplace=True)
books_read["Original Publication Year"].fillna("0",inplace=True)


# # Model Time

# In[256]:


drop_columns = [ 'Book Id', 'Title', 'Author', 'Author l-f',
       'Additional Authors', 'ISBN', 'ISBN13', 'My Rating',
       'Publisher', 'Binding', 'Year Published', 'Date Read', 'Date Added', 'Bookshelves',
       'Bookshelves with positions', 'Exclusive Shelf', 'My Review', 'Spoiler',
       'Private Notes', 'Read Count', 'Recommended For', 'Recommended By',
       'Owned Copies', 'Original Purchase Date', 'Original Purchase Location',
       'Condition', 'Condition Description', 'BCID']


# In[257]:


books["Exclusive Shelf"].unique()


# In[258]:


books_read


# In[95]:


books_read.drop(drop_columns, axis="columns").sum().sort_values()


# In[97]:


categories_types2 = [['biography & autobiography', 'games', 0, 'sports & recreation',
       'history', 'science', 'self-help', 'business & economics',
       'mathematics', 'computers', 'literary collections', 'families',
       'political science', 'nature', 'education', 'fiction',
       'social science', 'design', 'philosophy',
       'language arts & disciplines', 'body, mind & spirit',
       'family & relationships', 'juvenile fiction', 'equality',
       'leadership', 'medical', 'psychology', 'dictators',
       'man-woman relationships', 'existentialism', 'drama',
       'cross-country running', 'poetry', 'cocaine', 'suicide',
       'immigrants', 'short stories', 'children of holocaust survivors',
       'english language', 'france', 'psychological fiction', 'religion',
       'business', 'self-acceptance', 'civilization', 'humor',
       'buchenwald (concentration camp)', 'war',
       'pilgrims (new plymouth colony)', 'literature',
       'juvenile nonfiction', 'comics & graphic novels',
       'adventure and adventurers', 'animal rights activists',
       'human trafficking', 'communism','Average Rating','Number of Pages','Original Publication Year']]


# In[374]:


from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(books_read, test_size=0.2)
X_train2 = train_set[['self-development', 'school', 'fantasy', 'economics', 'biography',
       'france', 'french-literature', 'personal-development',
       'historical-fiction', 'self-improvement', 'politics', 'science',
       'self-help', 'history', 'french', 'business', 'novels', 'psychology',
       'philosophy', 'classic', 'literature', 'classics', 'fiction',
       'nonfiction', 'Average Rating']]
X_train = train_set[['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating']]
y_train = train_set["My Rating"]
X_test = test_set[['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating']]
X_test2 = test_set[['self-development', 'school', 'fantasy', 'economics', 'biography',
       'france', 'french-literature', 'personal-development',
       'historical-fiction', 'self-improvement', 'politics', 'science',
       'self-help', 'history', 'french', 'business', 'novels', 'psychology',
       'philosophy', 'classic', 'literature', 'classics', 'fiction',
       'nonfiction', 'Average Rating']]
y_test = test_set["My Rating"]


# ## Logistic Regression

# In[375]:


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


# In[376]:


model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train,y_train)
y_pred2 = model.predict(X_test)


# In[377]:


from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
model.score(X_train, y_train)
print(mae(y_test,y_pred2),mse(y_test,y_pred2))


# In[378]:


coeffs = dict(zip(X_train.columns,model.coef_))


# In[379]:


not_read = books[books["My Rating"] == 0]
X_for_predict = not_read


# In[380]:


log_model = model


# In[381]:


df1["innovation"] = pd.to_numeric(df1["innovation"],downcast='integer')


# In[382]:


df1["innovation"].dtype


# In[360]:


for i in ['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating','Number of Pages','Original Publication Year']:
    try:    
        X_for_predict[i] = pd.to_numeric(X_for_predict[i],downcast='float').astype("float64")
    except KeyError:
        print(i)
        
        
        


# In[383]:


#predictors = log_model.predict(X_for_predict[['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating','Number of Pages','Original Publication Year']])
predictors = log_model.predict(X_for_predict[['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating']])

predictors = pd.DataFrame(predictors)
predictors = predictors.reset_index()

finale = not_read
results = finale.merge(predictors, on="index")
results["Log Reg Predicted Rating"] = results[0]
results["Log Reg Ranking"] = results[["Log Reg Predicted Rating"]].sort_values(by=0,axis="columns").rank(axis="rows",method="first",ascending=False).round(decimals=0)
results[["Title","Author","Average Rating","Log Reg Predicted Rating","Log Reg Ranking"]].groupby("Log Reg Ranking").mean()


# ## Normal Linear

# In[386]:


model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)


# In[387]:


from sklearn.metrics import mean_squared_error as mse 
from sklearn.metrics import mean_absolute_error as mae
model.score(X_train,y_train)
mae(y_test,y_pred)
mse(y_test,y_pred)


# In[388]:


coeffs = dict(zip(X_train.columns,model.coef_))
dict(sorted(coeffs.items(), key=lambda item: item[1]))


# In[365]:


#X_for_predict["Year Published"].fillna("0",inplace=True)
#X_for_predict["Number of Pages"].fillna("0",inplace=True)
#X_for_predict["Original Publication Year"].fillna("0",inplace=True)
X_for_predict = not_read


# In[366]:


lin_model = model


# In[367]:


predictors = model.predict(X_for_predict[['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating']])


# In[368]:


predictors = pd.DataFrame(predictors)


# In[369]:


predictors = predictors.reset_index()


# In[370]:


results = results.merge(predictors, on="index")


# In[371]:


results["Lin Reg Predicted Rating"] = results["0_y"] 


# In[372]:


results["Lin Reg Ranking"] = results[["Lin Reg Predicted Rating"]].sort_values(by=0,axis="columns").rank(axis="rows",method="first",ascending=False)


# In[373]:


results


# In[303]:


results["Lin Reg Ranking"] = results.groupby("Lin Reg Predicted Rating").mean().rank(axis="rows",method="first",ascending=False)


# In[304]:



omg = results[["Title","Author","Average Rating","Lin Reg Predicted Rating","Lin Reg Ranking","Log Reg Ranking","Log Reg Predicted Rating"]].sort_values("Lin Reg Predicted Rating",ascending=False)
omg


# ## KNN Regressor

# In[390]:


from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor(n_neighbors=5)


# In[391]:


knn_model.fit(X_train, y_train)


# In[392]:


y_pred = knn_model.predict(X_test)


# In[393]:


y_pred


# In[394]:


knn_model.score(X_train, y_train)


# In[395]:


mse(y_test,y_pred)


# In[396]:


mae(y_test,y_pred)


# In[397]:


knn_reg_model = knn_model


# In[399]:


predictors2 = knn_reg_model.predict(X_for_predict[['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating']])


# In[400]:


predictors2 = pd.DataFrame(predictors2)


# In[401]:


predictors2 = predictors2.reset_index()


# In[402]:


finale2 = not_read


# In[408]:


results2 = finale2.merge(predictors2, on="index").sort_values(by=0, ascending=False)


# In[409]:


results2["KNN Reg Predicted Rating"] = results2[0] 
results2["KNN Reg Ranking"] = results2[["KNN Reg Predicted Rating"]].sort_values(by=0,axis="columns").rank(axis="rows",method="first",ascending=False).round(decimals=0)


# In[412]:


omg = omg.merge(results2,on="Title")


# In[413]:


omg["Ranking Mean"] = omg[["Lin Reg Ranking","Log Reg Ranking","KNN Reg Ranking"]].mean(axis=1).round(decimals=0)


# In[414]:


omg


# In[415]:


omg[["Title","Author_x","Average Rating_x","Lin Reg Predicted Rating","Lin Reg Ranking","Log Reg Ranking","Log Reg Predicted Rating","KNN Reg Predicted Rating","KNN Reg Ranking", "Ranking Mean"]].sort_values("Ranking Mean",ascending=True)


# In[416]:


plt.figure(figsize=(15,8))
omg.plot.bar(x="Title",y="Ranking Mean")
omg.plot.bar(x="Title",y="Lin Reg Ranking")


# In[417]:


results2[["Title","Author","Average Rating","Predicted Rating"]]


# ## k Neighbours Classifier

# In[346]:


knn_reg_model.score(X_train, y_train)


# In[348]:


lin_model.score(X_train, y_train)


# In[349]:


log_model.score(X_train, y_train)


# In[419]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
k_scores=[]
for k in range(1,31):
    # 2. run KNeighborsClassifier with k neighbours
    knn = KNeighborsClassifier(n_neighbors=k)
    # 3. obtain cross_val_score for KNeighborsClassifier with k neighbours
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    # 4. append mean of scores for k neighbors to k_scores list
    k_scores.append(scores.mean())


# In[420]:


knn = KNeighborsClassifier(n_neighbors=8)
knn.fit(X_train, y_train)


# In[421]:


y_pred = knn.predict(X_test)


# In[422]:


knn.score(X_train, y_train)


# In[423]:


knn_class_model = knn


# In[424]:


predictors2 = knn.predict(X_for_predict[['self-development', 'school', 'fantasy', 'economics', 'biography',
       'france', 'french-literature', 'personal-development',
       'historical-fiction', 'self-improvement', 'politics', 'science',
       'self-help', 'history', 'french', 'business', 'novels', 'psychology',
       'philosophy', 'classic', 'literature', 'classics', 'fiction',
       'nonfiction', 'Average Rating']])

predictors2 = knn.predict(X_for_predict[['fiction', 'nonfiction', 'classics', 'psychology', 'philosophy', 'history', 'literature', 'business', 'science', 'classic', 'politics', 'self-help', 'french', 'historical-fiction', 'fantasy', 'novels', 'young-adult', 'childrens', 'childhood', 'children', 'humor', 'self-improvement', 'ya', 'adventure', 'biography', 'kids', 'economics', 'magic-tree-house', 'personal-development', 'comics', 'graphic-novels', 'book-club', 'asterix', 'memoir', 'sociology', 'bd', 'comic', 'french-literature', 'france', 'plays', 'spy', 'action', 'drama', 'school', 'humour', 'technology', 'theatre', 'mystery', '1001-books', 'contemporary', 'audible', 'cherub', 'finance', 'leadership', 'political-science', 'war', 'series', 'romance', 'children-s', 'play', 'self-development', 'essays', 'graphic-novel', 'productivity', 'historical', 'children-s-books', 'français', 'dystopia', 'math', '1001', 'mathematics', 'poetry', 'science-fiction', 'alex-rider', 'german', 'tech', 'american', 'memoirs', 'chapter-books', 'africa', 'sci-fi', 'short-stories', 'travel', 'health', 'magical-realism', 'religion', 'spirituality', 'théâtre', 'education', 'holocaust', 'sports', 'novel', 'dystopian', 'mental-health', 'american-literature', 'design', 'spanish', 'investing', 'italian', 'race', 'roman', 'writing', 'a-very-short-introduction', 'management', 'thriller', 'vsi', 'work', 'adult', 'political-philosophy', 'economy', 'german-literature', 'abandoned', 'data', 'histoire', 'career', 'japan', 'bill-gates', 'econ', 'russian', 'very-short-introductions', 'psych', 'reference', 'russia', 'biography-memoir', 'dnf', 'autobiography', 'theater', 'feminism', 'picture-books', 'running', 'bibliothèque', 'china', 'czech', 'jeunesse', 'poésie', 'blinkist', 'computer-science', 'culture', 'environment', 'ethics', 'fitness', 'japanese', 'poesie', 'read-in-2021', 'shakespeare', 'to-get', 'african-american', 'american-history', 'asia', 'filosofia', 'games', 'japanese-literature', 'learning', 'anthropology', 'austria', 'coming-of-age', 'ensayo', 'existentialism', 'milan-kundera', 'political-theory', 'world-history', 'comedy', 'communication', 'literary-fiction', 'maths', 'middle-grade', 'russian-literature', 'short-introductions', 'military', 'society', '19th-century', '2020-reads', 'chess', 'language', 'memory', 'nature', 'novela', 'architecture', 'fantastique', 'marketing','Average Rating']])


# In[333]:


predictors2 = pd.DataFrame(predictors2)


# In[334]:


predictors2 = predictors2.reset_index()


# In[335]:


finale2 = not_read


# In[336]:


results2 = finale2.merge(predictors2, on="index").sort_values(by=0, ascending=False)


# In[337]:


results2["KNN Class Predicted Rating"] = results2[0] 
results2["KNN Class Ranking"] = results2[["KNN Class Predicted Rating"]].sort_values(by=0,axis="columns").rank(axis="rows",method="first",ascending=False).round(decimals=0)


# In[338]:


results2[0]


# In[339]:


omg = results2[["Title","KNN Class Predicted Rating", "KNN Class Ranking"]].merge(omg,on="Title")


# In[340]:


omg["Ranking Mean"] = omg[["Lin Reg Ranking","Log Reg Ranking","KNN Reg Ranking"]].mean(axis=1).round(decimals=0)


# In[341]:


final_table = omg[["Title","Author_x","Average Rating_x","Lin Reg Predicted Rating","Lin Reg Ranking","Log Reg Predicted Rating","Log Reg Ranking", "KNN Reg Predicted Rating","KNN Reg Ranking","KNN Class Predicted Rating","KNN Class Ranking","Ranking Mean"]]


# In[342]:


final_table = final_table.sort_values(by="Lin Reg Ranking", ascending=True)


# In[343]:


final_table


# ## Test

# In[ ]:


final_table["Lin Reg Predicted Rating"]


# In[ ]:


my_range=range(1,115)


# In[ ]:


plt.figure(figsize=(40,24))

plt.hlines(y=range(1,115), xmin=final_table['Lin Reg Predicted Rating'].astype(object), xmax=final_table['Log Reg Predicted Rating'].astype(object), color='grey', alpha=0.4)
plt.scatter(final_table['Lin Reg Predicted Rating'], my_range, color='skyblue', alpha=1, label='Lin Reg')
plt.scatter(final_table['Log Reg Predicted Rating'], my_range, color='green', alpha=0.4 , label='Log Reg')
plt.yticks(my_range, final_table['Title'])
plt.legend(fontsize="xx-large")
plt.show()


# In[ ]:


final_table[["Lin Reg Predicted Rating","Log Reg Predicted Rating","KNN Reg Predicted Rating"]].plot(kind="bar")


# In[ ]:


plt.figure(figsize=(60,32))
sns.barplot(x="Title",y="Ranking Mean",data=final_table)
sns.barplot(x="Title",y="Lin Reg Ranking",data=final_table)


# # Obtaining All Books

# In[ ]:


all_books = pd.read_csv("books.csv",error_bad_lines=False)
all_books


# In[ ]:


all_books


# ## Getting the Genres

# In[ ]:


isbns=[]
for i in all_books["isbn"]:
    isbns.append(i)


# In[ ]:


genreExceptions = [
'to-read', 'currently-reading', 'owned', 'default', 'favorites', 'books-i-own',
'ebook', 'kindle', 'library', 'audiobook', 'owned-books', 'audiobooks', 'my-books',
'ebooks', 'to-buy', 'english', 'calibre', 'books', 'british', 'audio', 'my-library',
'favourites', 're-read', 'general', 'e-books',"read-in-2020"
] #ignore these different bookshelves
genres = []


# In[ ]:


def get_id():
    count = 0
    for i in isbns:
        test= requests.get(f"https://www.goodreads.com/search?q={i}&format=xml&key=TTL1SltyHTYVjZHafssQxw",allow_redirects=False)
        test = BeautifulSoup(test.content, "xml")
        try:
            book_id = test.work.best_book.id.text
            get_xml(book_id)
        except AttributeError:
            genres.append(0)
        count += 1
        print(count)
        print(book_id)
    return genres


# In[ ]:


def get_xml(i): #obtaining the bookshelves in which the book is included 
    if requests.get(f"https://www.goodreads.com/book/show/{i}?key=xQXvrwOTLq7xonOLcjt2A",allow_redirects=False):
        test = requests.get(f"https://www.goodreads.com/book/show/{i}?key=xQXvrwOTLq7xonOLcjt2A",allow_redirects=False)
        test = BeautifulSoup(test.content, "lxml")
        shelves = test.find("popular_shelves")
        finding_genres(shelves)
    else: 
        genres.append(0)
    return genres


# In[ ]:


def finding_genres(shelves): #filtering the bookshelves to obtain only
    prov_genres = [] #the first 8 results
    try: 
        for i in shelves:
            if len(i) == 0:
                x = i.attrs["name"]
                if x not in genreExceptions:
                    if len(prov_genres) < 8:
                        x = x.replace("non-fiction","nonfiction").lower()
                        prov_genres.append(x)
                        prov_genres = list(dict.fromkeys(prov_genres))
        genres.append(prov_genres)
    except TypeError:
        genres.append(0)
    return genres


# In[ ]:


get_id()


# In[ ]:


len(genres)


# In[ ]:


backup = genres


# In[ ]:


len(backup)


# In[ ]:


all_books = all_books.reset_index()


# In[ ]:


genres_df2 = pd.DataFrame({"Genres":genres}).reset_index()


# In[ ]:


genres_df2 = genres_df2.merge(all_books,on="index")


# In[ ]:


#Saving df as cv
genres_df2.to_csv("goodreads_all_books2.csv")


# ## Encoding Genres

# In[ ]:


#Removing Null Values
indexes = genres_df2[genres_df2["Genres"] == 0].index
indexes = indexes[::-1]
for i in indexes:
    genres.pop(i)


# In[ ]:


df = pd.get_dummies(pd.DataFrame(genres))
df.columns = df.columns.str.split("_").str[-1]


# In[ ]:


df1 = pd.DataFrame({"index":np.arange(0,11123)})


# In[ ]:


genres_dupli = []
for i in df.columns:
    if i not in genres_dupli:
        try: 
            df1[i]= pd.DataFrame(df[i].sum(axis=1))
            genres_dupli.append(i)
        except ValueError:
            df1[i] = pd.DataFrame(df[i])
            genres_dupli.append(i)


# ## Looking at the Genres

# In[ ]:


df1.sum().sort_values(ascending=False).head(10)


# # Predicting Popular Books

# In[ ]:


books = genres_df2.merge(df1,on="index")


# In[ ]:


#if necessary when average rating exists 3 times
books[["Average Rating"]] = books["average_rating_x"].mean(axis=1)
books[["Average Rating"]]


# ## Cleaning Our DF

# In[ ]:


for i in ['self-development', 'school', 'fantasy', 'economics', 'biography',
       'france', 'french-literature', 'personal-development',
       'historical-fiction', 'self-improvement', 'politics', 'science',
       'self-help', 'history', 'french', 'business', 'novels', 'psychology',
       'philosophy', 'classic', 'literature', 'classics', 'fiction',
       'nonfiction', 'average_rating_x']:
    books[i] = books[i].fillna(0)


# In[ ]:


books = books.drop(labels='average_rating_x',axis=1)


# In[ ]:


books = books.reset_index()
books["Index"] = books["index"]


# In[ ]:


#Removing Triplets
books["a"], books["b"],books["c"] = books[["authors_x"]]
books[["a","b","c"]] = books["authors_x"]
books["Author"] = books["a"]


# In[ ]:


#Removing Triplets
books["d"], books["e"],books["f"] = books[["title_x"]]
books[["d","e","f"]] = books["title_x"]
books["Title"] = books["d"]


# In[ ]:


def remove_triplets(old,new):
    books["d"], books["e"],books["f"] = books[[old]]
    books[["d","e","f"]] = books[old]
    books[new] = books["d"]


# In[ ]:


def action():
    modifications = {"  num_pages_x":"#Pages","ratings_count_x":"Ratings Count","publication_date_x":"Publication Date",    "language_code_x":"Language Code"}
    for i,j in modifications.items():
        remove_triplets(i,j)


# ## Predicting

# In[ ]:


models = {lin_model:0,log_model:1,knn_reg_model:2,knn_class_model:3}
name = ["Lin Reg","Log Reg","KNN Reg","KNN Class"]
for i,j in models.items():
    predictors = i.predict(books[['self-development', 'school', 'fantasy', 'economics', 'biography',
       'france', 'french-literature', 'personal-development',
       'historical-fiction', 'self-improvement', 'politics', 'science',
       'self-help', 'history', 'french', 'business', 'novels', 'psychology',
       'philosophy', 'classic', 'literature', 'classics', 'fiction',
       'nonfiction', 'Average Rating','Number of Pages','Original Publication Year']])
    predictors = pd.DataFrame(predictors)
    predictors[f"{name[j]} Predicted Rating"] = predictors[0]
    predictors.drop(labels=0,axis=1,inplace=True)
    predictors = predictors.reset_index()
    predictors["Index"] = predictors["index"]
    x = books.merge(predictors, on="index")
    books = x
    books[f"{name[j]} Ranking"] = books[[f"{name[j]} Predicted Rating"]].sort_values(by=0,axis="columns").rank(axis="rows",method="first",ascending=False).round(decimals=0)
    


# In[ ]:


books["Average Ranking"] = books[["Log Reg Ranking","Lin Reg Ranking","KNN Reg Ranking","KNN Class Ranking"]].mean(axis=1)
final_results = books[["Title","Author","Average Rating","#Pages","Ratings Count","Publication Date","Language Code","Average Ranking","Log Reg Predicted Rating","Log Reg Ranking",         "Lin Reg Predicted Rating","Lin Reg Ranking",        "KNN Reg Predicted Rating", "KNN Reg Ranking",        "KNN Class Predicted Rating","KNN Class Ranking"]].sort_values("Average Ranking")


# In[ ]:


final_results.to_csv("final_results.csv")


# In[ ]:


final_results.drop("Language Code", axis=1).head(15)


# # Additional Analysis

# In[ ]:


books["Publication Year"] = books["Publication Date"]


# In[ ]:


for i in range(0,len(books["Publication Date"])):
    books["Publication Year"] = books["Publication Year"][i][-4:]


# In[ ]:


books["Publication Year"][2]


# In[ ]:


sns.distplot(final_results["#Pages"])


# In[ ]:


sns.countplot(books["Year Published"])


# In[ ]:


plt.figure(figsize=(15,8))
sns.violinplot(x="#Pages",data=final_results)


# # Evaluate

# In[ ]:


test = requests.get("https://www.goodreads.com/search?q=0743564677&format=xml&key=TTL1SltyHTYVjZHafssQxw")
test = BeautifulSoup(test.content, "xml")
try:
    test.work.best_book.id
    print(1)
except AttributeError:
    print(2)


# In[ ]:


all_books["isbn"][667]


# In[ ]:


all_books["title"][488]


# ## Get Images

# In[ ]:


test = requests.get(f"https://www.goodreads.com/book/show/{ids[0]}?key=xQXvrwOTLq7xonOLcjt2A")
test = BeautifulSoup(test.content, "lxml")


# In[ ]:


test.find("small_image_url").text


# In[ ]:


response = requests. get("https://i.imgur.com/ExdKOOz.png")
file = open("sample_image.png", "wb")
file.write(response. content)
file.close()


# In[ ]:


file


# In[ ]:




