import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Read the text file into a pandas DataFrame
data = pd.read_csv("C:\\Users\\hp\\OneDrive\\Desktop\\dialogs.txt", sep='\t', header=None, names=['text', 'label'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)

# Remove stop words using NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
X_train = X_train.apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word not in stop_words]))
X_test = X_test.apply(lambda x: ' '.join([word for word in word_tokenize(x.lower()) if word not in stop_words]))

# Convert texts to numerical vectors using Count Vectorizer
vectorizer = CountVectorizer(max_df=0.5, stop_words='english')
X_train_count = vectorizer.fit_transform(X_train)
X_test_count = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_count, y_train)

# Test the chatbot
print("Chatbot: Hi there! How can I assist you today?")

while True:
    user_input = input("User: ")

    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    # Tokenize the user input
    tokens = word_tokenize(user_input.lower())
    
    # Remove stop words
    tokens = [word for word in tokens if word not in stop_words]
    
    # Convert the user input to a numerical vector
    vector = vectorizer.transform([' '.join(tokens)])
    
    # Predict the label
    prediction = clf.predict(vector)
    predicted_label = prediction[0]

    print(f"Chatbot: {predicted_label}")
