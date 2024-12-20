import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import string
import numpy as np
import subprocess as sub

nltk.download('punkt')
nltk.download('stopwords')

def preproccess_words(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [word for word in words if word not in stop_words and word not in string.punctuation]
    return words

def score_sentence(text, word_freq):
    sentences = sent_tokenize(text)
    sentence_score = {}
    sentence_vectors = []

    for sentence in sentences:
        sentence_word = word_tokenize(sentence.lower())
        sentence_vector = np.array([word_freq.get(word,0) for word in sentence_word])
        sentence_vectors.append(sentence_vector)

    attenntion_weights = []
    for vector in sentence_vector:
        attention_weight = np.array(vector) / [np.sum(np.sum(v) for v in sentence_vectors) + 1e-9]
        attenntion_weights.append(attention_weight)

    for idx, sentence in enumerate(sentences):
        sentence_score[sentence] = attenntion_weights[idx]

    return sentence_score

def summrize(text, num_sentence):
    words = preproccess_words(text)
    word_freq = Counter(words)

    sentence_score = score_sentence(text , word_freq=word_freq)

    soreted_Sentence =  sorted(sentence_score.items(), key=lambda x :x[1] , reverse=True)
    top_sentence = soreted_Sentence[:num_sentence]
    summary = ''.join([sentence for sentence, score in top_sentence])

    return summary

def Run():
    input_text = input("Enter your text : ")
    summary = summrize(input_text, num_sentence=2)
    sub.run('cls',shell=True)
    print('===============================================================')
    print("Summary:")
    print('===============================================================')
    print(summary)
    print('===============================================================')
