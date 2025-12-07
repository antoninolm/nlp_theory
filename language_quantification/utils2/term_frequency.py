import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils2.preprocessing1 import preprocess_text
from utils2.poems import poems

# preprocess text
processed_poems = [preprocess_text(poem) for poem in poems]

# initialize and fit CountVectorizer
vectorizer1 = CountVectorizer()
term_frequencies = vectorizer1.fit_transform(processed_poems)

# get vocabulary of terms
feature_names = vectorizer1.get_feature_names_out()

# get corpus index
corpus_index = [f"Poem {i+1}" for i in range(len(poems))]

# create pandas DataFrame with term frequencies
df_term_frequencies = pd.DataFrame(term_frequencies.T.todense(), index=feature_names, columns=corpus_index)

# initialize and fit CountVectorizer
vectorizer2 = CountVectorizer()
bow_matrix = vectorizer2.fit_transform(processed_poems)

# get vocabulary of terms
feature_names = vectorizer2.get_feature_names_out()

# get corpus index
corpus_index = [f"Poem {i+1}" for i in range(len(poems))]

# create pandas DataFrame with term frequencies
df_bag_of_words = pd.DataFrame(bow_matrix.T.todense(), index=feature_names, columns=corpus_index)