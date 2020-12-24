import pandas as pd
import numpy as np
import string
import re
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
#Loading data 
print("Loading the data.....")
f=open('../input/quora/train.csv', encoding = "ISO-8859-1")
data=pd.read_csv(f)
f_test=open('../input/quora/test.csv', encoding = "ISO-8859-1")
d_test=pd.read_csv(f_test)
print("Data loading ended.....")


#These are preprocessin functions that are used below
contra={"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have" }

def remove_newline(text):
    """
    remove \n and  \t
    """
    text = re.sub('\n', ' ', text)
    text = re.sub('\t', ' ', text)
    text = re.sub('\b', ' ', text)
    text = re.sub('\r', ' ', text)
    return text


def spacing_punctuation(text):
    """
    add space before and after punctuation and symbols
    """
    re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤$&#‘’])')
    return re_tok.sub(r' \1 ', text)


def remove_punctuation(text):
    """
    remove punctuation from text
    """
    re_tok = re.compile(f'([{string.punctuation}])')
    return re_tok.sub(' ', text)


def spacing_number(text):
    """
    add space before and after numbers
    """
    re_tok = re.compile('([0-9]{1,})')
    return re_tok.sub(r' \1 ', text)

def decontracted(text, mapping):
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

def clean_number(text):
    """
    replace number with hash
    """
    text = re.sub('[0-9]{5,}', '#####', text)
    text = re.sub('[0-9]{4}', '####', text)
    text = re.sub('[0-9]{3}', '###', text)
    text = re.sub('[0-9]{2}', '##', text)
    return text


def remove_number(text):
    """
    numbers are not toxic
    """
    return re.sub('\d+', ' ', text)


def remove_space(text):
    """
    remove extra spaces and ending space if any
    """
    text = re.sub('\s+', ' ', text)
    text = re.sub('\s+$', '', text)
    return text
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


import nltk
from nltk.corpus import stopwords
from  nltk.stem import SnowballStemmer

TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")
def preprocess(text):
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    text = spacing_punctuation(text)
    text = spacing_number(text)
    text = decontracted(text,contra)
    text = remove_number(text)
    text=cleanHtml(text)
    text = remove_space(text)
    text = remove_punctuation(text)
    return text

#Apply above preprocessing techniques
print("Applying preprocessing.........")
lst=data['question_text'].apply(lambda x:preprocess(x))
dtest=d_test['question_text'].apply(lambda x:preprocess(x))
print("Preprocessing Ended")

traind=lst.to_list()
testd=dtest.to_list()

god=traind+testd
# print(god[0])
print("Constructing TF-IDF matrix......")
def feature_extraction(data):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_vectorizer =TfidfVectorizer(ngram_range=(1,4),min_df=3,max_df=0.9,strip_accents='unicode',use_idf=True,smooth_idf=True,sublinear_tf=True)
    features=tfidf_vectorizer.fit_transform(data)
    return features,tfidf_vectorizer

tf_feat,tfv=feature_extraction(god)

tftrain=tfv.transform(traind)
tftest=tfv.transform(testd)
print("Construction of TF-IDF matrix ended......")


print("NB log count ratio multiplication started")
y=data['target'].values
p = 1 + tftrain[y==1].sum(0)
q = 1 + tftrain[y==0].sum(0)
csr = csr_matrix(np.log(
    (p / (1 + (y==1).sum())) /
    (q / (1 + (y==0).sum()))
))


def multiply(X,csr):
    return X.multiply(csr)

# nb_transformer = NBTransformer(alpha=1).fit(tftrain, y)
nb_train = multiply(tftrain,csr)
nb_test = multiply(tftest,csr)
print("NB log count ratio multiplication ended")


print("Model training started...")
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

#This is just used to calculate the best threshold on the basis of predictions on validation_sets during cross validation f1_sccore is used to get the best threshold 
def estimate_threshold(y_true, y_proba):
    thresh = 0
    best_score = 0
    #This range is decided by lookin at the predictions so we wanted try out these values and see what is best threshold
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            thresh = threshold
            best_score = score
    result = {'threshold_value': thresh, 'f1_score': best_score}
    return result

y=data['target'].values

kf = StratifiedKFold(n_splits=20,shuffle=True,random_state=42)
predictions = np.zeros(nb_test.shape[0])
trpredictions=np.zeros(nb_train.shape[0])
cv_score =[]
i=1
for train_index,test_index in kf.split(nb_train,y):
    X_train,y_train=nb_train[train_index],y[train_index]
    X_test,y_test=nb_train[test_index],y[test_index]
    print(" Training for fold :=",i)
    clf=LogisticRegression(C=0.5,max_iter=40,class_weight='balanced')
    clf.fit(X_train,y_train)
    pred=clf.predict_proba(X_test)
    trpredictions[test_index] = pred[:,1]
    predictions += clf.predict_proba(nb_test)[:,1]/20
    print(" Training ended for fold :=",i)
    i+=1

#Submission file generation
result=estimate_threshold(y,trpredictions)
print(result)
sub = pd.read_csv('../input/quora/sample_submission.csv')
sub.target = (predictions> result['threshold_value']).astype('int')
sub.to_csv("best_score.csv", index=False)
print("Model Training ended")



    
