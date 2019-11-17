from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB

path = r'G:\Study\4-1\AI Lab work'

dataset = load_files(path, shuffle= False, decode_error='ignore', random_state=None)
x_train= dataset.data
y_train=dataset.target
vectorizer=TfidfVectorizer( use_idf=True)
x_train=vectorizer.fit_transform(x_train)
x_train=x_train.toarray()
clf= MultinomialNB()
clf.fit(x_train,y_train)

test_path= r'G:\Study\4-1\AI text'

dataTest=load_files(test_path, shuffle=False, decode_error='ignore',random_state=None)
x_Test=dataTest.data
y_Test=dataTest.target
names = dataset.target_names
x_Test= vectorizer.transform(x_Test)
x_Test=x_Test.toarray()
pr = clf.predict(x_Test)
print("Prediction is :",names[pr])
#print('Prediction', clf.predict(x_Test))
#acuracy= clf.score(x_Test,y_Test)
#print("Acuracy is",acuracy)

#for item in document:
#	with io.open(item,'r',encoding='utf-8') as f:
#		text=f.read()
#	with io.open('test2.txt','w',encoding='utf-8') as f1:
#		 f1.write(text)
        
	     	
