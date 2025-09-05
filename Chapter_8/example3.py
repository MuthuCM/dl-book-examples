#Example 8.3 Sentiment Analysis of IMDB Movie Reviews using RNN
import numpy as np
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

data_path = "https://raw.githubusercontent.com/LawrenceDuan/IMDb-Review-Analysis/master/IMDb_Reviews.csv"
movie_dataset = pd.read_csv(data_path, engine='python')
movie_dataset.head()

X = movie_dataset["review"]
y = movie_dataset["sentiment"]

#print(X[0])
#print(y[0])

def clean_text(doc):

    document = re.sub('[^a-zA-Z]', ' ', doc)
    document = re.sub(r"\s+[a-zA-Z]\s+", ' ', document)
    document = re.sub(r'\s+', ' ', document)
    return document

X_sentences = []
reviews = list(X)
for rev in reviews:
    X_sentences.append(clean_text(rev))

#from nltk.corpus import stopwords
from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer (max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
#vectorizer = TfidfVectorizer (max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords)
X= vectorizer.fit_transform(X_sentences).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#LSTM Model
model = Sequential([
    LSTM(50, input_shape=(49,1)),
    Dense(1),
    Activation('sigmoid')

])

#Model compilation
adam = Adam(learning_rate=0.001)
model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
model.fit(X_train,y_train, epochs = 20, validation_split=0.3)

y_pred = model.predict(X_test) # Use X_test instead of x_test
y_pred = (y_pred > 0.5).astype(int) # Convert probabilities to binary predictions
print(accuracy_score(y_pred, y_test))

