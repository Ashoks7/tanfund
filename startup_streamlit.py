
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import FreqDist
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
import streamlit as st
import numpy as np
import plotly.express as px
import time
from transformers import pipeline
import base64



def shark(user_sentence):
 df1 = pd.read_excel('SharkTankIndiaS02.xlsx')
 df=df1[["Idea","Deal"]].copy()
 df["deal/no deal"]=df["Deal"].copy()
 df['deal/no deal']=df['deal/no deal'].apply(lambda x: 'Deal' if x != 'No Deal' else x)
 df['deal in binary']=df['deal/no deal'].apply(lambda x: 1 if x != 'No Deal' else x)
 df["deal in binary"] = df["deal in binary"].replace(['No Deal'],[0]).astype(int)
# Replacing punctuations with space
 df['Idea_processed'] = df['Idea'].str.replace("[^a-zA-Z0-9]", " ")
 df['Idea_processed'] = df['Idea_processed'].astype(str)
 df['Idea_processed'] = df['Idea_processed'].apply(lambda row: ' '.join([word for word in row.split() if len(word)>2]))
# make entire text lowercase
 df['Idea_processed'] = [row.lower() for row in df['Idea_processed']]

# Removing Stopwords Begin
 import nltk
 nltk.download('punkt')
 nltk.download('stopwords')
 from nltk.corpus import stopwords
 from nltk import word_tokenize

 stop_words = stopwords.words('english') # extracting all the stop words in english language and storing it in a variable called stop_words -> set

# Making custom list of words to be removed 
 add_words = ['brand','company']

# Adding to the list of words
 stop_words.extend(add_words)

# Function to remove stop words 
 def remove_stopwords(rev):
    # iNPUT : IT WILL TAKE ROW/REVIEW AS AN INPUT
    # take the paragraph, break into words, check if the word is a stop word, remove if stop word, combine the words into a para again
    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized  if i not in stop_words])
    return rev_new

# Removing stopwords
 df['Idea_processed'] = [remove_stopwords(r) for r in df['Idea_processed']]
# Begin Lemmatization 
 nltk.download('wordnet')
 nltk.download('omw-1.4')
 nltk.download('averaged_perceptron_tagger')
 from nltk.stem import WordNetLemmatizer
 from nltk.corpus import wordnet

# function to convert nltk tag to wordnet tag
 lemmatizer = WordNetLemmatizer()

# Finds the part of speech tag
# Convert the detailed POS tag into a shallow information
 def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# lemmatize sentence using pos tag
 def lemmatize_sentence(sentence):
  # word tokenize -> pos tag (detailed) -> wordnet tag (shallow pos) -> lemmatizer -> root word
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  # output will be a list of tuples -> [(word,detailed_tag)]
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged) # output -> [(word,shallow_tag)]
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


 df['Idea_processed'] = df['Idea_processed'].apply(lambda x: lemmatize_sentence(x))



# Importing module
 from sklearn.feature_extraction.text import TfidfVectorizer

# Creating matrix of top 2500 tokens
 tfidf = TfidfVectorizer(max_features=2500)
# tmp_df = tfidf.fit_transform(df.review_processed)
# feature_names = tfidf.get_feature_names()
# pd.DataFrame(tmp_df.toarray(), columns = feature_names).head() 

 X = tfidf.fit_transform(df.Idea_processed).toarray()
 y = df['deal/no deal'].map({'Deal' : 1, 'No Deal' : 0}).values
 featureNames = tfidf.get_feature_names_out()





# # Splitting the dataset into train and test
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

 from sklearn.tree import DecisionTreeClassifier

 dt = DecisionTreeClassifier()
 dt.fit(X_train,y_train)

 y_pred = dt.predict(X_test)

 user_sentence = user_sentence.replace("[^a-zA-Z0-9]", " ")
 user_sentence = ' '.join([x for x in user_sentence.split() if len(x) > 2])
 user_sentence = user_sentence.lower()

 user_sentence = remove_stopwords(user_sentence)
 user_sentence = lemmatize_sentence(user_sentence)


 X_test = tfidf.transform([user_sentence]).toarray()


 predictdeal= dt.predict(X_test)
 a= predictdeal[0]
 if a==1:
  b="your Idea seems good. you will get a Deal in sharktank 👌"
 else:
  b="you need to work more on your Idea.you will Not get a Deal in sharktank 👎"

 return b




def data():
 df8 = pd.read_excel('SharkTankIndiaS02.xlsx')
 df9=df8[["Idea","Deal","Amit Jain","Namita","Anupam","Vineeta","Aman","Piyush"]].copy()
 df9["deal/no deal"]=df9["Deal"].copy()
 df9['deal/no deal']=df9['deal/no deal'].apply(lambda x: 'Deal' if x != 'No Deal' else x)
 df9['deal in binary']=df9['deal/no deal'].apply(lambda x: 1 if x != 'No Deal' else x)
 df9["deal in binary"] = df9["deal in binary"].replace(['No Deal'],[0])
# Replacing punctuations with space
 df9['Idea_processed'] = df9['Idea'].str.replace("[^a-zA-Z0-9]", " ")
 df9['Idea_processed'] = df9['Idea_processed'].astype(str)
 df9['Idea_processed'] = df9['Idea_processed'].apply(lambda row: ' '.join([word for word in row.split() if len(word)>2]))
# make entire text lowercase
 df9['Idea_processed'] = [row.lower() for row in df9['Idea_processed']]
 # Removing Stopwords Begin
 import nltk
 nltk.download('punkt')
 nltk.download('stopwords')
 from nltk.corpus import stopwords
 from nltk import word_tokenize

 stop_words = stopwords.words('english') # extracting all the stop words in english language and storing it in a variable called stop_words -> set

# Making custom list of words to be removed 
 add_words = ['brand','company']

# Adding to the list of words
 stop_words.extend(add_words)

# Function to remove stop words 
 def remove_stopwords(rev):
    # iNPUT : IT WILL TAKE ROW/REVIEW AS AN INPUT
    # take the paragraph, break into words, check if the word is a stop word, remove if stop word, combine the words into a para again
    review_tokenized = word_tokenize(rev)
    rev_new = " ".join([i for i in review_tokenized  if i not in stop_words])
    return rev_new

# Removing stopwords
 df9['Idea_processed'] = [remove_stopwords(r) for r in df9['Idea_processed']]
# Begin Lemmatization 
 nltk.download('wordnet')
 nltk.download('omw-1.4')
 nltk.download('averaged_perceptron_tagger')
 from nltk.stem import WordNetLemmatizer
 from nltk.corpus import wordnet

# function to convert nltk tag to wordnet tag
 lemmatizer = WordNetLemmatizer()

# Finds the part of speech tag
# Convert the detailed POS tag into a shallow information
 def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None

# lemmatize sentence using pos tag
 def lemmatize_sentence(sentence):
  # word tokenize -> pos tag (detailed) -> wordnet tag (shallow pos) -> lemmatizer -> root word
    #tokenize the sentence and find the POS tag for each token
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(sentence))  # output will be a list of tuples -> [(word,detailed_tag)]
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged) # output -> [(word,shallow_tag)]
    lemmatized_sentence = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_sentence.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
    return " ".join(lemmatized_sentence)


 df9['Idea_processed'] = df9['Idea_processed'].apply(lambda x: lemmatize_sentence(x))



# Importing module
 from sklearn.feature_extraction.text import TfidfVectorizer

# Creating matrix of top 2500 tokens
 tfidf = TfidfVectorizer(max_features=2500)
# tmp_df = tfidf.fit_transform(df.review_processed)
# feature_names = tfidf.get_feature_names()
# pd.DataFrame(tmp_df.toarray(), columns = feature_names).head() 

 X = tfidf.fit_transform(df9.Idea_processed).toarray()
 y = df9['deal/no deal'].map({'Deal' : 1, 'No Deal' : 0}).values
 featureNames = tfidf.get_feature_names_out()





# # Splitting the dataset into train and test
 from sklearn.model_selection import train_test_split
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

 from sklearn.tree import DecisionTreeClassifier

 dt = DecisionTreeClassifier()
 dt.fit(X_train,y_train)

 y_pred = dt.predict(X_test)

 from sklearn.metrics import confusion_matrix, accuracy_score
 accuracy = accuracy_score(y_test, y_pred)

 from sklearn.metrics import roc_auc_score
 ra = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])

 ##featureImportance.sort_values(by='Importance')
 featureImportance = pd.DataFrame({i : j for i,j in zip(dt.feature_importances_,featureNames)}.items(),columns = ['Importance','word'])
 fi=featureImportance.sort_values(by='Importance',ascending=False)
 
 return df9, accuracy, ra, fi


st.set_page_config(layout="wide", initial_sidebar_state='expanded')
# Add back ground image




def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://startuptn.in/wp-content/uploads/2022/10/startuptn.jpg");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

add_bg_from_url()
 
# GUI.py
cls1,cls2,cls3=st.columns((0.5,3,0.5))
with cls2:
  st.title("Will you get :orange[Funding] for your Idea :orange[?]")

cl1,cl2,cl3=st.columns((1,2,1))
 
with cl2:
# Get input from user for hashtag
  paragraph = st.text_input("Enter the idea:")

  if st.button("predict"):
   summarizer = pipeline("summarization")
   summary = summarizer(paragraph, max_length=150, min_length=30, do_sample=False)[0]["summary_text"]
   user_sentence = summary
   progress_bar = cl2.progress(0)
   for perc_completed in range(100):
    time.sleep(0.05)
    progress_bar.progress(perc_completed+1)
   st.button("idea summary:")
   st.button(summary)
   b = shark(user_sentence)
   st.button(b)

# pls download sharktank data from kaggle so you can see the idea is good or not
# in vscode pls run in terminal in below format
# python -m streamlit run tanfund.py




        
     
 
