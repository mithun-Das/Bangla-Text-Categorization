import io
import  math
from textblob import TextBlob as tb
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.datasets

print("At line 7, done with all the imports...")

"""def unique_WordInFinalList(list,final_list):
	i=0
	while(i<len(list)):
		  word=list.pop()
		  if word not in list:
		  	 final_list.append(word)

    
     
def unique_wordlist(input_file,list):
    with io.open(input_file,'r',encoding='utf-8') as f:
          text=f.read()
          f.close()
          wordlist=text.split()
 #         print(wordlist)
    
          i=0       
          while(i<len(wordlist)):
                word=wordlist.pop()
                if word not in wordlist:
                   str_new=str(word).replace('।',"")
                   str_new=str(str_new).replace('?',"")
                   str_new=str(str_new).replace('!',"")
                   str_new=str(str_new).replace(',',"")
                   list.append(str_new)     
    
      
    return   

document=['আসল(Come).txt','আসল(Real).txt','উত্তর(answer).txt',
           'উত্তর(north).txt','কর(Do).txt','কর(tax).txt',
           'কাল(time).txt','কাল(black).txt','গ্রাম(weight).txt',
           'গ্রাম(village).txt','চাল(rice).txt','চাল(strategy).txt',
           'চিনি(sugar).txt','চিনি(recognize).txt','জাল(Fraud).txt',
           'জাল(net).txt','জোড়া(connecting).txt','জোড়া(pair).txt',
           'ডাক(call).txt','ডাক(letterrelated).txt','ডাল(branch).txt',
           'ডাল(pulm).txt','দল(petal).txt','দল(Team).txt','পত্র(Leaf).txt',
           'পত্র(Letter).txt','পাত্র(bowl).txt','পাত্র(groom).txt',
           'পান(Drink).txt','পান(leaf).txt','পাল(herd).txt',
           'পাল(snail).txt','ফল(fruit).txt','ফল(result).txt',
           'বল(Ball).txt','বল(say).txt','মেলা(fair).txt',
           'মেলা(opening).txt','সিদ্ধ(boil).txt','সিদ্ধ(satisfy).txt',
           'হার(lose).txt','হার(neckles).txt']


i=0
list=[]
final_list=[]
while(i<len(document)):

      unique_wordlist(document[i], list)
      i=i+1
"""
#unique_WordInFinalList(list,final_list)

path =r'G:\Study\4-1\AI Lab work'
print("at line 63 before loding the dataset....")
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
        
	     	
