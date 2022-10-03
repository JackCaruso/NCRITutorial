# Sentiment Analysis
# Import dependencies and required magics
wordDoc = open('file.txt')
# Load text from web-page, save to local file
!pip install textblob
# Load from saved file, review it, 
# drop lines as needed, perform necessary processing.
text = ''
for line in wordDoc:
    text += line
# Perform sentiment analysis
from textblob import TextBlob
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')


blob = TextBlob(text)
sentences = list()
polarity = list()
subjectivity = list()

for sentence in blob.sentences:
    print(sentence)
    print(sentence.sentiment.polarity)
    sentences.append(sentence.raw)
    polarity.append(sentence.sentiment.polarity)
    subjectivity.append(sentence.sentiment.subjectivity)
print(sentences)
# Save sentiment data to dataframe
import pandas
df = pandas.DataFrame()
df['sentences'] = sentences
df['polarity'] = polarity
df['subjectivity'] = subjectivity
df = df.sort_values('polarity')
df
# Output key sentiment analysis results including:
#   Overall sentiment analysis scores for the document
#   Correlation of polarity and subjectivity scores across sentences
df['polarity'].mean()
print(df['polarity'].mean())
df.corr()
# Print out 20 sentences and their scores including:
#    5 most negative sentences including polarity and subjectivity
#    5 most positive sentences including polarity and subjectivity
#    5 most subjective sentences including polarity and subjectivity
#    5 most objective sentences including polarity and subjectivity
df = df.sort_values('polarity')
print(df.head())
print(df.tail())
df = df.sort_values('subjectivity')
print(df.head())
print(df.tail())
