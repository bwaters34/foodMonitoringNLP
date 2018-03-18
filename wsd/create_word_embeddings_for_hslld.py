# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
import pickle 
import os 
from gensim.models import Word2Vec 
from sklearn.decomposition import PCA 
from matplotlib import pyplot 


class wordEmbeddings:
	def __init__(self):
		self.database_from_HSLLD = [] 
		self.food_words = self.load("data/food_desc_files/food_names.pickle")

		self.HSLLD_file_Loc = self.load('food_files.pickle')
		self.HSLLD_file_Loc = self.correct_file_location(self.HSLLD_file_Loc)
		self.Read_files_from_HSLLD = self.read_files_HSLLD(self.HSLLD_file_Loc)
		
		print "Lenght of sentences HSLLD:", len(self.database_from_HSLLD)
		self.sentences_to_wordEmbeddings(self.database_from_HSLLD)

	def load(self, fileLocaiton):
		fileLocaiton = "../"+fileLocaiton
		with open(fileLocaiton, 'r') as f:
			return pickle.load(f)

	def correct_file_location(self, fileLocaiton):
		for index, fileLoc in enumerate(fileLocaiton):
			fileLocaiton[index] = "../"+fileLoc
			if not os.path.exists(fileLocaiton[index]):
				print "Path of the file doesn't exist", fileLocaiton[index]
		return fileLocaiton

	def read_files_HSLLD(self, fileLoc):
		for file in fileLoc:
			f = open(file, 'r')
			for sentences in f:
				if '*' in sentences:
					sentences = sentences[6:]
					self.database_from_HSLLD.append(sentences.split())

	def sentences_to_wordEmbeddings(self, sentence, min_word_count = 1):
		model = Word2Vec(sentence, size = 100, min_count = min_word_count)
		model.save('word_embeddings_HSLLD.bin')
		# new_model = Word2Vec.load('word_embeddings_HSLLD.bin')
		print(model)
		words = list(model.wv.vocab)
		# print(words)
		# print(model['apple'])
		X = model[model.wv.vocab]
		pca = PCA(n_components = 2)
		result = pca.fit_transform(X)

		# pyplot.scatter(result[:, 0], result[:, 1])
		for i, word in enumerate(words):
			if word in self.food_words:
				pyplot.scatter(result[i, 0], result[i, 1])
				pyplot.annotate(word, xy =(result[i, 0], result[i, 1]))
			
		pyplot.show()


if __name__ == '__main__':
	Embeddings = wordEmbeddings()
