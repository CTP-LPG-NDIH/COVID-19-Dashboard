import pandas as pd



df = pd.read_csv('https://raw.githubusercontent.com/CUNYTechPrep/2020-fall-data-science/master/Week5-RandomForest/data/titanic.csv')

# Define our X
X = df['Tweet']

# Define our y
y = df['Party']


# Initalize our vectorizer
vectorizer = TfidfVectorizer()

# Fit our vectorizer
vectorizer.fit(X)

# Transform your X data using your fitted vectorizer. 
X = vectorizer.transform(X)

model = MultinomialNB(alpha=.025)

# Fit our model with our training data.
model.fit(X, y)

# Save our vectorizer and model.
pickle.dump(vectorizer, open('models/vectorizer.pkl', 'wb') )

pickle.dump(model, open('models/text-classifier.pkl', 'wb') )


