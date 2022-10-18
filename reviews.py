import nltk, re, string
from nltk.tokenize import word_tokenize, sent_tokenize
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from textblob import TextBlob, Word
from sklearn.feature_extraction.text import CountVectorizer
from nltk.probability import FreqDist

import pkg_resources
from symspellpy import SymSpell


sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt"
)
bigram_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
)
# term_index is the column of the term and count_index is the
# column of the term frequency
sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)


# run only one initial spell check on dataset since it's very slow

def clean_text_1(text):	
	text = text.lower()
	text = re.sub("\[.*?\]", '', text)
	text = re.sub("[%s]" % re.escape(string.punctuation), '', text)
	text = re.sub("\w*\d\w*", '', text)
	text = re.sub("\n", '', text)
	return text

count = 0
def spell_check(text):
	# global count
	# count += 1
	# print(text)
	# text = TextBlob(text).correct()
	# print(text)
	# print(count)
	suggestions = sym_spell.lookup_compound(text, max_edit_distance=2)
# display suggestion term, edit distance, and term frequency
	for suggestion in suggestions:
		print(suggestion)
	print("Original:", text)
	
	
pd.options.display.max_colwidth = 100
pd.set_option('display.max_rows', None)
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t')
print(dataset["Review"][:20])

data_clean = pd.DataFrame(dataset["Review"].apply(clean_text_1))
data_clean = pd.DataFrame(data_clean["Review"].apply(spell_check))
print("\n\n", data_clean["Review"][:20])

clean_text = []

sentences = []

words = []

ps = PorterStemmer()

for i in range(0, 100):
	review = clean_text_1(dataset["Review"][i])
	review_sentences = sent_tokenize(review)
			
	for word in (word_tokenize(review)):
		if not word in set(stopwords.words("english")):
			words.append(word)
		
	sentences.append(review_sentences)
	for review in review_sentences:
		review = word_tokenize(review)
		
		#alternatively use ps.stem(word)
				
		# Word(word).correct()
		review = [ps.stem(word) for word in review
			if not word in set(stopwords.words('english'))]
		review = ' '.join(review)
		clean_text.append(review)

print(clean_text)
print("\n\n")
print(sentences)
print("\n\n")
print(words)

word_freq = nltk.FreqDist(words)
word_freq_df = pd.DataFrame(list(word_freq.items()), columns = ["Word","Frequency"]) 
print(word_freq_df)

cv = CountVectorizer(stop_words="english")
data_cv = cv.fit_transform(data_clean["Review"])
data_dtm = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names_out())
data_dtm.index = data_clean.index
# print(data_dtm)





	


	
	