import re
from nltk.corpus import stopwords
import spacy
import joblib
import streamlit as st

stop_words_en = stopwords.words('english')
stop_words_fr = stopwords.words('french')

nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")


def cleaner_lemmatizer(message):
    message = message.lower() # on transforme en minuscule tous les lettres
    message = re.sub(r"['|-]",' ', message) # on remplace [' | -] par les espaces
    message = re.sub(r'http\S+|www\S+', '<url>', message) # on remplace tous les url par un token <url>
    message = re.sub(r'\w+.@\w+.\w+', '<email>', message) # on remplace aussi les email par <email>
    message = re.sub(r'\+\s?\d+', '<numero mobile>', message) # on remplace aussi les numeros par <numero mobile>


    # montants avec devises ($, €, £, etc.)
    message = re.sub(r'[\$€£¥]\s*[\d\s,\.]+', '<montant>', message)
    message = re.sub(r'[\d\s,\.]+\s*[\$€£¥]', '<montant>', message)
    message = re.sub(r'\b\d[\d\s,\.]*\s*(dollars?|euros?|pounds?|yen|francs?|usd|eur|gbp)\b', '<montant>', message, flags=re.IGNORECASE)

    message = re.sub(r'\b\d+\s*(st|nd|rd|th|ème|eme|er|ère|ere)\b', '<ordinal>', message, flags=re.IGNORECASE) # ordre

    message = re.sub(r'\b\d[\d\s,\.]*\s*%', '<pourcentage>', message) # pourcentage

    message = re.sub(r'\b\w*\d+\w*\b', '<chiffre>', message) # chiffres restants sans devise
    message = re.sub(r'[^\w\s]', '', message) # on supprime les ponctuations

    list_words = message.split()
    
    score_langue = {'en' : 0, 'fr' : 0}

    for word in list_words:
        for stop_word in stop_words_en:
            if word == stop_word:
                score_langue['en'] += 1
        for stop_word in stop_words_fr:
            if word == stop_word:
                score_langue['fr'] += 1
            
    if score_langue['en'] > score_langue['fr']:
        langue_detecte = 'en'
        list_words = [word for word in list_words if word not in stop_words_en]
        list_words = ' '.join(list_words)

    else:
        langue_detecte = 'fr'
        list_words = [word for word in list_words if word not in stop_words_fr]
        list_words = ' '.join(list_words)

    nlp = nlp_en if langue_detecte == 'en' else nlp_fr
    doc = nlp(list_words)
    list_lemma = [token.lemma_ for token in doc]

    return ' '.join(list_lemma)

preprocessor = joblib.load('./preprocessing_v1.pkl')
model = joblib.load('./model_v1.pkl')

while True:
    message = input("Entrez votre message : ")

    if message != '':
        text_processed = preprocessor.transform([cleaner_lemmatizer(message)]).toarray()
        y_pred = model.predict(text_processed)

        result = 'Spam' if y_pred[0] == 1 else 'Ham'
    else:
        message = input("Entrez votre message : ")

    print(result)

