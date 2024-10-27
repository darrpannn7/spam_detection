from flask import Flask, render_template, request
import joblib
import traceback

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form['email_text']
    print(f"Received email text: {email_text}")  # Debugging print

    try:
        # Transform the input text using the loaded vectorizer
        email_vectorized = vectorizer.transform([email_text])
        prediction = model.predict(email_vectorized)  # This expects a 2D array
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
        print(f"Prediction result: {result}")  # Debugging print
    except Exception as e:
        print(f"Error during prediction: {e}")  # Print basic error message
        print(traceback.format_exc())  # Print full traceback
        result = "Error occurred during prediction."

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
