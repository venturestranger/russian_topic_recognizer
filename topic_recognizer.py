from pymystem3 import Mystem
from nltk import sent_tokenize, word_tokenize, regexp_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.pipeline import Pipeline
import pickle
import numpy as np
from config import Config

class Normalizer:
	def __init__(self, path=Config.PATH, stopwords_loc='stopwords-ru.txt'):
		try:
			self.stopwords = []

			with open(path + stopwords_loc, 'r') as file:
				for line in file:
					self.stopwords.append(line.strip())
		except:
			self.stopwords = []
	
	def query(self, text, stopwords=None, normalize=True, min_length=4, as_str=False):
		if not stopwords:
			stopwords = self.stopwords

		if normalize:
			stem = Mystem()
			words = [lemma['analysis'][0]['lex'] for lemma in stem.analyze(text) if lemma.get('analysis', None) != None and len(lemma.get('analysis', [])) != 0 and not lemma['text'][0].isupper()]
		else:
			regexp=r'(?u)\b\w{4,}\b'
			words = [w for sent in sent_tokenize(text)
				for w in regexp_tokenize(sent, regexp)]

		if stopwords:
			stopwords = set(stopwords)
			words = [tok for tok in words if tok not in stopwords]

		words = [word for word in words if len(word) >= min_length]
		if as_str:
			words = ' '.join(words)

		return words

class TopicRecognizer:
	def __init__(self, path=Config.PATH, docs='', stopwords_loc='stopwords-ru.txt'):
		try:
			with open(path + 'tfidf.pkl', 'rb') as file:
				vectorizer = pickle.load(file)
			with open(path + 'lda.pkl', 'rb') as file:
				lda = pickle.load(file)
			self.normalizer = Normalizer(stopwords_loc=stopwords_loc, path=path)
			self.kernel = Pipeline([
				('vectorizer', vectorizer),
				('latent_dirichlet_allocation', lda),
			])
		except Exception as e:
			print('TopicRecognizer: not initialized ', e)
	
	def fit(self, text, n_topics=100, path=Config.PATH, stopwords_loc='stopwords-ru.txt', tokens=None):
		self.normalizer = Normalizer(stopwords_loc=stopwords_loc, path=path)

		if tokens == None:
			for idx in range(len(text)):
				text[idx] = ' '.join(normalizer.query(text[idx]))
				print(f'Fit on: {idx} / {len(text)}')
		else:
			with open(path + tokens) as file:
				text = pickle.load(file)

		vectorizer = TfidfVectorizer(max_features=10000)
		tfidf_matrix = vectorizer.fit_transform(text)
		
		lda = LatentDirichletAllocation(n_components=n_topics, max_iter=5, random_state=0)
		lda.fit(tfidf_matrix)

		with open(path + 'tfidf.pkl', 'wb') as file:
			pickle.dump(vectorizer, file)
		with open(path + 'lda.pkl', 'wb') as file:
			pickle.dump(lda, file)

		self.__init__(path=path, stopwords_loc=stopwords_loc)

	def query(self, text, n_returns=1):
		try:
			data = self.kernel.transform([self.normalizer.query(text, as_str=True)])
			return self.kernel['vectorizer'].get_feature_names_out()[self.kernel['latent_dirichlet_allocation'].components_[np.argmax(data[0])].argsort()[-n_returns:]]
		except Exception as e:
			print('TopicRecognizer: ', e)
			return -1

if __name__=='__main__':
	model = TopicRecognizer()
	print(model.query('Казахстанских водителей предупредили об ухудшении погодных условий и сложной ситуации на дорогах в регионах, передает Tengrinews.kz со ссылкой на "КазАвтоЖол"', n_returns=3))
