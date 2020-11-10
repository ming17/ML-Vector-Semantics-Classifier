Michael Ing
Intro to NLP Homework 3
10-29-2020

I wrote my code in python3.7, and used a few off-the-shelf packages including:
	-sklearn.model_selection.train_test_split
	-sklearn.feature_extraction.CountVectorizer
	-sklearn.feature_extraction.TfidfTransformer
	-sklearn.linear_model.LogisticRegression
	-xlrd
	-nltk.corpus.stopwords
	-nltk.stem.porter.PorterStemmer
	-GoogleNews-vectors-negative300 for pre-trained embeddings

The code can be executed by running python LogisticClassifier.py, but also requires that "file_location"
be changed to the location of the corpus file and "filename" be changed to the location of the 
GoogleNews-vectors-negative300 word embeddings binary file.
	
I didn't discuss the assignment with anyone else, but did spend a lot of time
reading through documentation to understand how to interact with the packages.
Originally, I tried used Stanford's CoreNLP LogisticClassifier in Java, but the 
documentation was terrible and really confusing to the point that I had to switch
part way through to python and scikit-learn.

I had some trouble determining how to find the co-occurence matrix for sparse vectors.
Originally I used large matrix multiplication, but then decided to switch to scrubbing 
the data and keeping track of counts to find the term-term matrix