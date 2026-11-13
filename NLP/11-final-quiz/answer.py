import os
import numpy as np
import pandas as pd
import pickle
import spacy

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier, accuracy
from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATASET_PATH = './jobpostingdata.csv'
MODEL_PATH = './naive_bayes_exercise.pkl'

stop_words = stopwords.words('english')
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer()

def load_dataset():
  df = pd.read_csv(DATASET_PATH)
  return df['title'], df['fraudulent'], df['text']

def get_tag(tag):
  if tag.startswith('J'):
    return 'a'
  elif tag.startswith('R'):
    return 'r'
  elif tag.startswith('V'):
    return 'v'
  else:
    return 'n'
  
def preprocess(text):
  text = text.lower()
  text = word_tokenize(text)
  text = [w for w in text if w not in stop_words]
  text = [w for w in text if w.isalpha()]

  tagged = pos_tag(text)
  text = [lemmatizer.lemmatize(w, get_tag(t)) for (w, t)in tagged]
  return ' '.join(text)

def extract_feature(freq_dist, text):
  words = text.split()
  return {word: (word in words) for word in freq_dist}

def train(text, labels):
  texts = text.apply(preprocess)
  # print(texts)
  all_words = ' '.join(texts).split()
  # print(all_words)

  freq_dist = FreqDist(all_words)
  feature_set = [(extract_feature(freq_dist, texts), label) for (texts, label) in zip(text, labels)]
  print(feature_set)
  split = int(0.8 * len(feature_set))

  train_set = feature_set[:split]
  test_set = feature_set[split:]

  classifier = NaiveBayesClassifier.train(train_set)
  print(f"Trained Naive Bayes with accuracy: {accuracy(classifier, test_set)}")

  vectors = vectorizer.fit_transform(texts)

  return classifier, freq_dist, vectorizer, vectors

def train_or_load_model():
  if not os.path.exists(MODEL_PATH):
    titles, labels, texts = load_dataset()
    model = train(texts, labels)

    with open(MODEL_PATH, 'wb') as f:
      pickle.dump((*model, titles, texts), f)
    return (*model, titles, texts)
  else:
    with open(MODEL_PATH, 'rb') as f:
      return pickle.load(f)
    
def menu():
  text = ''
  text_category = ''

  classifier, freq_dist, vectorizer, vectors, titles, texts = train_or_load_model()

  while True:
    print('Job Posting Classification')
    print('Your Text:', text if text else 'None')
    print('Your Text Category:', text_category if text_category else 'None')  # FIX: now prints the category value, not just the label
    print('1. Write your texts')  # FIX: typo 'yout' -> 'your'
    print('2. View Recommendation')
    print('3. View NER')
    print('4. Exit')

    choice = input('>> ')
    if choice == '1':
      text = input('Write your text: ')
      if len(text) < 20 or len(text.split()) < 3:
        text = ''
        continue
      clean_text = preprocess(text)
      feature = extract_feature(freq_dist, clean_text)
      result = classifier.classify(feature)

      if result == 1:
        text_category = 'Fake Job Posting'
      else:
        text_category = 'Real Job Posting'

    elif choice == '2':
      if len(text) == 0:
        print('No Text')
        continue
      clean_text = preprocess(text)
      vector = vectorizer.transform([clean_text])
      similar = cosine_similarity(vector, vectors)[0]
      top = np.argsort(similar)[-5:][::-1]

      for i in top:
        print('Title:', titles.iloc[i])
        print('Similarity:', similar[i])

    elif choice == '3':
      if len(text) == 0:
        print('No Text')
        continue
      nlp = spacy.load('en_core_web_sm')
      doc = nlp(text)

      categories = {}
      for ent in doc.ents:
        categories.setdefault(ent.label_, []).append(ent.text)
      if not categories:
        print('No named entities found')
      for label, ents in categories.items():
        print(f"{label}: {', '.join(ents)}")
    
    elif choice == '4':
      break

    else:
      print('Invalid choice!')

menu()