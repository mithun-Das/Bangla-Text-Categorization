from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files


path = r'G:\Study\4-1\AI Lab work'

dataset = load_files(path, shuffle=True, decode_error='ignore', random_state=42)
x_train= dataset.data
y_train=dataset.target

vectorizer=TfidfVectorizer()
x_train=vectorizer.fit_transform(x_train)
x_train=x_train.toarray()
print("total unique words found: ", x_train.shape[1])

#for item in document:
#	with io.open(item,'r',encoding='utf-8') as f:
#		text=f.read()
#	with io.open('test2.txt','w',encoding='utf-8') as f1:
#		 f1.write(text)
        
	     	
