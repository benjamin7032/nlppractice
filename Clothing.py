import json
import spacy

class Sentiment:
    NEGATIVE = "NEGATIVE"
    POSITIVE = "POSITIVE"  
       
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        else:
            return Sentiment.POSITIVE
#đọc dữ liệu từ các file     
train_file = 'train_Clothing.json'
reviews = []
with open(train_file) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))        
test_file = 'test_Clothing.json'
tests = []     
with open(test_file) as f:
    for line in f:
        test = json.loads(line)
        tests.append(Review(test['reviewText'], test['overall']))
        
#đưa các thuộc tính và nhãn của mẫu vào các mảng    
train_X = [x.text for x in reviews]
train_y = [x.sentiment for x in reviews]
print('Số lượng mẫu training: ' + str(len(train_X)))
test_X = [x.text for x in tests]
test_y = [x.sentiment for x in tests]
print('Số lượng mẫu test: ' + str(len(test_X)))

#chuẩn hóa dữ liệu bằng stemming/lemmatizing/stopwords-removal
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
stemmer = PorterStemmer() 
lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english')
train_X_edited = []
test_X_edited = []
for text in train_X:
    words = word_tokenize(text)
    phrase = []
    for word in words:
        if word not in stop_words:
            word = stemmer.stem(word)
            word = lemmatizer.lemmatize(word)
            phrase.append(word)
    train_X_edited.append(" ".join(phrase))           
for text in test_X:
    words = word_tokenize(text)
    phrase = []
    for word in words:
        if word not in stop_words:
            word = stemmer.stem(word)
            word = lemmatizer.lemmatize(word)
            phrase.append(word)
    test_X_edited.append(" ".join(phrase))
    
#đưa các câu vào word vectors
nlp = spacy.load("en_core_web_md")
train_docs = [nlp(text) for text in train_X_edited]
train_X_wv = [x.vector for x in train_docs]
test_docs = [nlp(text) for text in test_X_edited]
test_X_wv = [x.vector for x in test_docs]
#huấn luyện tập training bằng SVM
from sklearn import svm 
clf_svm = svm.SVC(kernel='linear')
clf_svm.fit(train_X_wv, train_y) 
#tính toán tỉ lệ đoán đúng của AI  
test_predict = clf_svm.predict(test_X_wv)
count = 0
for i in range(0, len(tests)):
    if test_predict[i] == test_y[i]:
        count += 1      
print("Tỉ lệ đoán đúng: " + str(count) + "/500 test (" +str(count/5) +"%)")