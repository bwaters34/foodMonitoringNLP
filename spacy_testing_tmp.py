import en_core_web_sm
nlp = en_core_web_sm.load()


# def find_root_token_of_sentence(sentence, nlp):
# 	doc = nlp(sentence)
# 	for token in doc:
# 	    if token.dep_ == 'ROOT':
# 	    	return token
# 	raise ValueError("No root found in sentence")


# def is_sentence_present_tense(sentence, nlp):
# 	root = find_root_token_of_sentence(sentence.decode(), nlp)
# 	return root.tag_ == 'VB' or root.tag_ == 'VBG' or root.tag_ == 'VBP' or root.tag_ == 'VBZ' 

# a = is_sentence_present_tense("I'm going to the store", nlp)
# print(a)

# b = 'I went to the store'
# a = is_sentence_present_tense(b, nlp)
# print(a)

# c = 'I will go to the store'
# print(is_sentence_present_tense(c, nlp))

from spacy.lemmatizer import Lemmatizer
from spacy.lang.en import LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES
lemmatizer = Lemmatizer(LEMMA_INDEX, LEMMA_EXC, LEMMA_RULES)
lemmas = lemmatizer(u'ducks', u'NOUN')
assert lemmas == [u'duck']
print(lemmas)


for token in nlp(u"i go to the store"):
	print(lemmatizer(token.text, token.tag_))

