import pandas as pd
import numpy as np
import re
import string

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt 
from matplotlib.pyplot import figure
# %matplotlib inline

import seaborn as sns
from sklearn import feature_extraction, model_selection, naive_bayes, pipeline, manifold, preprocessing, feature_selection, metrics

train = pd.read_csv("../input/quora/train.csv")
test = pd.read_csv("../input/quora/test.csv")

train.columns

test.columns

print("Number of train data points:",train.shape[0])
print("Number of test data points:",test.shape[0])
print("Shape of Train Data:", train.shape)
print("Shape of Test Data:", test.shape)
train.head()

import seaborn as sns

# Total sincere ans insincere questions
fig_dims = (12, 7)
fig, ax = plt.subplots(figsize=fig_dims)
sns.countplot(train['target'])

print("Percetage of insincere comments: ", round(train["target"].mean()*100, 2))
print("Percetage of sincere comments: ", 100-round(train["target"].mean()*100, 2))

import matplotlib.pyplot as plt
figure(num=None, figsize=(10, 8), dpi=80, facecolor='w', edgecolor='k')
ax=train['target'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%', shadow=True)
ax.set_title('Target')
plt.show()

from wordcloud import STOPWORDS,WordCloud

def wcloud(text,title=None,figure_size=(24.0,16.0)):
    stopwords = set(STOPWORDS)
    
    wordcloud = WordCloud(stopwords=stopwords,random_state = 42,width=800, height=400,).generate(str(text))
    
    plt.figure(figsize=figure_size)
    plt.title(title,fontdict={'size': 40,})
    plt.imshow(wordcloud)

train1_df = train[train["target"]==1]
train0_df = train[train["target"]==0]

wcloud(train1_df['question_text'],'Insincere Questions Cloud')

wcloud(train0_df['question_text'],'Insincere Questions Cloud')

train['word_count'] =  train['question_text'].apply(lambda x: len(str(x).split()))
test['word_count'] =  test['question_text'].apply(lambda x: len(str(x).split()))

train['char_count'] =  train['question_text'].apply(lambda x: len(str(x)))
test['char_count'] =  test['question_text'].apply(lambda x: len(str(x)))

train['stopwords'] =  train['question_text'].apply(lambda x: len([word for word in str(x).lower().split() if word in STOPWORDS]))
test['stopwords'] =  test['question_text'].apply(lambda x: len([word for word in str(x).lower().split() if word in STOPWORDS]))

fig_dims = (12, 7)
fig, ax = plt.subplots(figsize=fig_dims)
sns.boxplot(x='target', y='word_count', data=train)

fig_dims = (12, 7)
fig, ax = plt.subplots(figsize=fig_dims)
sns.boxplot(x='target', y='char_count', data=train)

fig_dims = (12, 7)
fig, ax = plt.subplots(figsize=fig_dims)
sns.boxplot(x='target', y='stopwords', data=train)

from collections import defaultdict
from wordcloud import STOPWORDS
stopwords = set(STOPWORDS)

def ngram_extractor(text, n_gram):
    token = [token for token in text.lower().split(" ") if token != "" if token not in STOPWORDS]
    ngrams = zip(*[token[i:] for i in range(n_gram)])
    return [" ".join(ngram) for ngram in ngrams]

def generate_ngrams(df, col, n_gram, max_row):
    temp_dict = defaultdict(int)
    for question in df[col]:
        for word in ngram_extractor(question, n_gram):
            temp_dict[word] += 1
    temp_df = pd.DataFrame(sorted(temp_dict.items(), key=lambda x: x[1])[::-1]).head(max_row)
    temp_df.columns = ["word", "wordcount"]
    return temp_df

def comparison_plot(df_1,df_2,col_1,col_2, space):
    fig, ax = plt.subplots(1, 2, figsize=(20,10))
    
    sns.barplot(x=col_2, y=col_1, data=df_1, ax=ax[0], color="blue")
    sns.barplot(x=col_2, y=col_1, data=df_2, ax=ax[1], color="blue")

    ax[0].set_title('Top words in sincere questions', size=18, color="blue")
    ax[1].set_title('Top words in insincere questions', size=18, color="blue")

    fig.subplots_adjust(wspace=space)

    plt.show()

sincere_2gram = generate_ngrams(train0_df, 'question_text', 1, 20)
insincere_2gram = generate_ngrams(train1_df, 'question_text', 1, 20)
comparison_plot(sincere_2gram,insincere_2gram,'word','wordcount', .50)

sincere_2gram = generate_ngrams(train0_df, 'question_text', 2, 20)
insincere_2gram = generate_ngrams(train1_df, 'question_text', 2, 20)
comparison_plot(sincere_2gram,insincere_2gram,'word','wordcount', .50)

sincere_2gram = generate_ngrams(train0_df, 'question_text', 3, 20)
insincere_2gram = generate_ngrams(train1_df, 'question_text', 3, 20)
comparison_plot(sincere_2gram,insincere_2gram,'word','wordcount', .50)
