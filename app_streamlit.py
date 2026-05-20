import re
from nltk.corpus import stopwords
import spacy
import joblib
import streamlit as st
from datetime import datetime

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

st.header(" :blue[Spam] or  :green[Ham]", text_alignment="center")

st.markdown("**Testez vos SMS ici / Test your SMS here:**", text_alignment="center")
message = st.text_area("", height=150)

if st.button("Predire / Predict"):

    if message != '':
        text_processed = preprocessor.transform([cleaner_lemmatizer(message)]).toarray()
        y_pred = model.predict(text_processed)

        result = 'Spam' if y_pred[0] == 1 else 'Ham'
        if result == "Spam":
            st.success(f":blue[{result}]")
        else:
            st.success(f":green[{result}]")

    else:
        st.error("Veuillez entrer un SMS a predire!")

st.markdown("""
    ---
""")
st.markdown("""
    Bienvenue.
    Cette plateforme détecte si un SMS est un **spam** ou un message légitime (**ham**).
    Entrez votre SMS ci-dessus et cliquez sur **Prédire** pour obtenir le résultat.
    Elle supporte les langues suivantes : Français · Anglais

    Welcome.
    This platform detects whether an SMS is **spam** or a legitimate message (**ham**).
    Enter your SMS above and click **Predict** to get the result.
    Supported languages : French · English
""", unsafe_allow_html=True, text_alignment='center')


st.markdown(f"""
    <div style="text-align: center; opacity: 0.5; font-size: 12px;">
        © {datetime.now().year} ny-tendry — Tous droits réservés
    </div>
""", unsafe_allow_html=True)
