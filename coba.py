import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
# from nltk.corpus import names
 
def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'bagus', 'enak', 'mantap', 'baik', 'cocok', 'spesial', 'murah', 'menyenangkan', 'cocok', 'cepat', 'sempurna', 'keren', 'indah', 'ramah', 'puas']
negative_vocab = [ 'jelek', 'buruk', 'benci', 'tidak', 'kurang', 'kecewa', 'mahal', 'membosankan', 'lambat', 'kurang']
neutral_vocab = [ 'makanan', 'minimuman', 'ini', 'melihat', 'bandung', 'dari', 'atas']
 
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
 
train_set = negative_features + positive_features + neutral_features
 
classifier = NaiveBayesClassifier.train(train_set) 
 
# Predict
neg = 0
pos = 0
neu = 0
sentence = "Menyenangkan dengan view yg bagus dan cuaca Bandung yang sejuk,untuk makanannya enak dengan penyajian berkelas.keseluruhan puas dengan makanan,penyajian,dan tempat dengan pemandangan yang bagus"
sentence = sentence.lower()

## preprocessing stemmning
#nltk_tokens = nltk.word_tokenize(sentence)b
#words = [word for word in nltk_tokens if word.isalpha()]
#
#porter_stemmer = PorterStemmer()
#stemmed = [porter_stemmer.stem(word) for word in nltk_tokens]
#for word in words:
#       print ("Asal: %s  Stem: %s"  % (word,porter_stemmer.stem(word)))
#
#for word in sentence:
#    classResult = classifier.classify(word_feats(word))
#    if classResult == 'neg':
#        neg = neg + 1
#    if classResult == 'pos':
#        pos = pos + 1
#
## print(words)
#print('Positive: ' + str(float(pos)/len(sentence)))
#print('Negative: ' + str(float(neg)/len(sentence)))

##preprocessing Lemmatizer
#nltk_tokens = nltk.word_tokenize(sentence)
#words = [word for word in nltk_tokens if word.isalpha()]
#
#wordnet_lemmatizer = WordNetLemmatizer()
#for w in words:
#    print ("Asal: %s  Lemma: %s"  % (w,wordnet_lemmatizer.lemmatize(w)))
#
#for word in sentence:
#    classResult = classifier.classify( word_feats(word))
#    if classResult == 'neg':
#        neg = neg + 1
#    if classResult == 'pos':
#        pos = pos + 1
#
## print(words)
#print('Positive: ' + str(float(pos)/len(sentence)))
#print('Negative: ' + str(float(neg)/len(sentence)))

#nltk_tokens = nltk.word_tokenize(sentence)
words1 = sentence.split(' ')
words = [word for word in words1 if word.isalpha()]

for word in words:
    classResult = classifier.classify(word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    elif classResult == 'pos':
        pos = pos + 1
print('Prediksi')
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))
#print('Neutral: ' + str(float(neu)/len(words)))
