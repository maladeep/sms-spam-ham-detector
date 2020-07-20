from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
# from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df= pd.read_csv("spam.csv", encoding="latin-1")
	df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
	# Features and Labels
	df['label'] = df['type'].map({'ham': 0, 'spam': 1})
	X = df['text']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	# Extract Feature With CountVectorizer :cleaning involved converting all of our data to lower case and removing all punctuation marks. 
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
#particular classifier is suitable for classification with discrete features ( word counts for text classification). It takes in integer word counts as its input. 
	clf = MultinomialNB() #NAIVE BAYES
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	





	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)