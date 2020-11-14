import flask
import pickle
import pandas as pd

# PIL is a library for handling images in python



app = flask.Flask(__name__, template_folder='templates')

path_to_vectorizer = 'models/vectorizer.pkl'
path_to_text_classifier = 'models/text-classifier.pkl'

with open(path_to_vectorizer, 'rb') as f:
    vectorizer = pickle.load(f)

with open(path_to_text_classifier, 'rb') as f:
    model = pickle.load(f)


@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        # Just render the initial form, to get input
        return(flask.render_template('main.html'))


    if flask.request.method == 'POST':
        # Get the input from the user.
        user_input_text = flask.request.form['user_input_text']
        
        # Turn the text into numbers using our vectorizer
        X = vectorizer.transform([user_input_text])
        
        # Make a prediction 
        predictions = model.predict(X)
        
        # Get the first prediction.
        prediction = predictions[0]

        predicted_probas = model.predict_proba(X)

        predicted_proba = model.predict_proba(X)[0]

        precent_democrat = predicted_proba[0]

        precent_republican = predicted_proba[1]

        return flask.render_template('main.html', 
            result=prediction,
            precent_democrat=precent_democrat,
            precent_republican=precent_republican)


@app.route('/titanic/')
def titanic():
    return flask.render_template('titanic.html')


@app.route('/images/')
def images():
    return flask.render_template('images.html')

@app.route('/bootstrap/')
def bootstrap():
    return flask.render_template('bootstrap.html')


if __name__ == '__main__':
    app.run()