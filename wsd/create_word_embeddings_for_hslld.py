# https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
import pickle 
import os 
from gensim.models import Word2Vec 
from sklearn.decomposition import PCA 
from matplotlib import pyplot 
import time 
import gensim 

class wordEmbeddings:
	def __init__(self):
		self.database_from_HSLLD = [] 
		self.dimension_size = 300
		self.food_words = self.load("data/food_desc_files/food_names.pickle")
		self.food_words_twitter = self.load("./data/food_desc_files/all_food_words_by_Yelena_Mejova.pickle")
		self.food_words.update(self.food_words_twitter)

		self.HSLLD_file_Loc = self.load('food_files.pickle')
		# self.names_of_file_with_hand_labels()

		self.HSLLD_file_Loc = self.correct_file_location(self.HSLLD_file_Loc)
		self.Read_files_from_HSLLD = self.read_files_HSLLD(self.HSLLD_file_Loc)
		
		print "Lenght of sentences HSLLD:", len(self.database_from_HSLLD)
		# self.sentences_to_wordEmbeddings_practice(self.database_from_HSLLD)
		# self.sentences_to_Google_wordEmbeddings_practice(None)

	def all_sentences(self):
		return self.database_from_HSLLD

	def names_of_file_with_hand_labels(self):
		loc = "../solutions/HSLLD/HV1/MT/"
		list_of_files = os.listdir(loc)
		return list_of_files

	def food_words_database(self):
		return self.food_words

	def load(self, fileLocaiton):
		fileLocaiton = "../"+fileLocaiton
		with open(fileLocaiton, 'r') as f:
			return pickle.load(f)

	def from_this_folder_load(self, fileLocaiton):
		with open(fileLocaiton, 'r') as f:
			return pickle.load(f)

	def save(self, fileLocaiton, variable):
		with open(fileLocaiton, 'w') as f:
			pickle.dump(variable, f)

	def correct_file_location(self, fileLocaiton):
		for index, fileLoc in enumerate(fileLocaiton):
			fileLocaiton[index] = "../"+fileLoc
			if not os.path.exists(fileLocaiton[index]):
				print "Path of the file doesn't exist", fileLocaiton[index]
		return fileLocaiton

	def read_files_HSLLD(self, fileLoc):
		files_already_annotated = self.names_of_file_with_hand_labels()

		for file in fileLoc:
			temp_file_name = file.split('/')[-1]
			#To prevent Overfitting 
			#Ignoring files already annotated
			if temp_file_name in files_already_annotated: 
				# print "Skipping file ", temp_file_name
				continue 
				pass
			f = open(file, 'r')
			for sentences in f:
				if '*' in sentences:
					sentences = sentences[6:]
					self.database_from_HSLLD.append(sentences.split())

	def sentences_to_wordEmbeddings(self, google_word_embeddings = 0):
		if google_word_embeddings:
			start = time.time()
			model = gensim.models.KeyedVectors.load_word2vec_format('/home/pritish/CCPP/wordEmbeddings/GoogleNews-vectors-negative300.bin.gz', binary=True)
			print "Time to load data ", time.time() - start
			return model
		return Word2Vec.load('word_embeddings_HSLLD_300.bin')

	# Not Working
	def sentences_to_Google_wordEmbeddings_practice(self, sentence, min_word_count = 1):
		# model = Word2Vec(sentence, size = self.dimension_size, min_count = min_word_count)
		# model.save('word_embeddings_HSLLD_300.bin')
		# new_model = Word2Vec.load('word_embeddings_HSLLD.bin')
		# print(model)
		model = self.sentences_to_wordEmbeddings(1)
		words = list(model.vocab)
		
		# print(words)
		# print(model['apple'])
		# X = model[words]
		X = []
		for food_name in self.food_words:
			if food_name in words:
				X.append(model.word_vec(food_name))
			
		print "....Starting PCA......"
		pca = PCA(n_components = 2)
		result = pca.fit_transform(X)
		print "Done with PCA......"
		# pyplot.scatter(result[:, 0], result[:, 1])
		for i, word in enumerate(words):
			if word in self.food_words:
				pyplot.scatter(result[i, 0], result[i, 1])
				pyplot.annotate(word, xy =(result[i, 0], result[i, 1]))
			
		pyplot.show()


	def sentences_to_wordEmbeddings_practice(self, sentence, min_word_count = 1):
		model = Word2Vec(sentence, size = self.dimension_size, min_count = min_word_count)
		model.save('word_embeddings_HSLLD_300.bin')
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
