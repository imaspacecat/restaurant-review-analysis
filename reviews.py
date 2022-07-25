import nltk, re, string
from nltk.tokenize import word_tokenize

text = input("enter some text: ")

def clean_text_1(text):
	text = text.lower()
	text = re.sub("\[.*?\]", '', text)
	text = re.sub("[%s]" % re.escape(string.punctuation), '', text)
	text = re.sub("\w*\d\w*", '', text)
	text = re.sub("\n", '', text)
	return text
	
text = clean_text_1(text)
print(text)
words = word_tokenize(text)
print(words)