#Import Dependentias
from crypt import methods
from flask import Flask, render_template, request, redirect
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import CountVectorizer

#Import Pickle file
file_name = "Restaurant-reviews-model.pkl"
classifier = pickle.load(open(file_name, 'rb'))

# Creating a corpus
file_name = "corpus.pkl"
corpus = pickle.load(open(file_name, 'rb'))

reviews = [];

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()

def predict_review(sample_message):
    # Removing the special character from the reviews and replacing it with space character
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
     # Converting the review into lower case character
    sample_message = sample_message.lower()
    # Tokenizing the review by words
    sample_message_words = sample_message.split()
     # Removing the stop words using nltk stopwords
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    # Stemming the words
    ps = PorterStemmer()
    # Joining the stemmed words
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)

def test(sample_message):
    sample_message = re.sub(pattern='[^a-zA-Z]',repl=' ', string = sample_message)
    sample_message = sample_message.lower()
    sample_message_words = sample_message.split()
    sample_message_words = [word for word in sample_message_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    final_message = [ps.stem(word) for word in sample_message_words]
    final_message = ' '.join(final_message)
    temp = cv.transform([final_message]).toarray()
    return classifier.predict(temp)

def calculate_rating(reviews):
    negative = 0
    positive = 0
    rating = 0.00
    
    #count number of positive and negative reviews
    for i in range(0,len(reviews)):
         if reviews[i]['result'] == 1:
            positive += 1
         elif reviews[i]['result'] == 0: 
             negative += 1
     
    #Get the percentage of positive reviews        
    if positive+negative > 0:
         rating = "{:.2f}".format((positive/(positive+negative)) * 5)
         
    else: 
         rating = 1.00 
    
    return rating
    
#Flask application cunstructor
app = Flask(__name__)

#Route Decorator used to bind a particular url to a function
@app.route('/')
def home():
	return render_template('index.html')

#Render_template used for generate output from a template file based on the Jinja2 engine

@app.route('/result', methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        if not message == "":
            if predict_review(message):
             temp = {"message":message,"result":1};
             reviews.append(temp);
             return render_template('index.html', result = 1, message = message) #positive
            
            else:
             temp = {"message":message,"result":0};
             reviews.append(temp);
             return render_template('index.html', result = 0, message = message) #negative
        else:
            return render_template('index.html')
    else:
        return render_template('index.html')
    
@app.route('/reviews',methods=['POST','GET'])
def showresults():
    return render_template('index2.html',reviews = reviews)

@app.route('/rating',methods=['POST','GET'])
def showRating():
    temp = calculate_rating(reviews)
    return render_template('index3.html',rating= temp,styling = int(float(temp)))

@app.route('/test', methods=['POST','GET'])
def test():
    if request.method == 'POST':
        message = request.form['message']
        if not message == "":
            if predict_review(message):
             return render_template('index4.html', result = 1, message = message) #positive
            
            else:
             return render_template('index4.html', result = 0, message = message) #negative
        else:
            return render_template('index4.html')
    else:
        return render_template('index4.html')

if __name__ == '__main__':
    #Run the app with errors and warning printed to the console
    app.run(debug=True)