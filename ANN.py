from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import load_files
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import svm

trainPath = r'F:\\Rafi\\My_Study\\Thesis\\Mithun_corpus'
testPath = r'F:\\Rafi\\My_Study\\Thesis\\Mithun_corpus_Test'

dataset = load_files(trainPath, shuffle= False, decode_error='ignore', random_state=None,load_content=True)
trainData,testData,trainTarget,testTarget = train_test_split(dataset.data,dataset.target,train_size  = 0.7, test_size=0.3,random_state=42);
vectorizer=TfidfVectorizer( max_features=3000,token_pattern='[^ ]+', use_idf=True)
trainData=vectorizer.fit_transform(trainData)
clf = MLPClassifier(hidden_layer_sizes=(50, ), random_state=1, activation='relu')
clf.fit(trainData,trainTarget)
testData= vectorizer.transform(testData)
testData=testData.toarray()
pr = clf.predict(testData)
acuracy= clf.score(testData,testTarget)
print("Acuracy is",acuracy)