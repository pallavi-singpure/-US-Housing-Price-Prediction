from flask import Flask, render_template, request
import numpy as np
import pickle

app= Flask(__name__)

# Load the trained model (update the filename as needed)
model = pickle.load(open('model-1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form inputs
        area = float(request.form['area'])
        bedrooms = int(request.form['bedrooms'])
        bathrooms = int(request.form['bathrooms'])
        stories = int(request.form['stories'])
        mainroad = 1 if request.form['mainroad'] == 'yes' else 0
        guestroom = 1 if request.form['guestroom'] == 'yes' else 0
        basement = 1 if request.form['basement'] == 'yes' else 0
        hotwaterheating = 1 if request.form['hotwaterheating'] == 'yes' else 0
        airconditioning = 1 if request.form['airconditioning'] == 'yes' else 0
        parking = int(request.form['parking'])
        prefarea = 1 if request.form['prefarea'] == 'yes' else 0

        # Furnishing status (One-hot encoded)
        furnishing = request.form['furnishingstatus']
        furnishing_furnished = 1 if furnishing == 'furnished' else 0
        furnishing_semifurnished = 1 if furnishing == 'semi-furnished' else 0
        # furnishing_unfurnished is the base case (0,0)

        # Prepare input array in correct order
        final_features = np.array([[area, bedrooms, bathrooms, stories,
                                    mainroad, guestroom, basement,
                                    hotwaterheating, airconditioning,
                                    parking, prefarea,
                                    furnishing_semifurnished, furnishing_furnished]])

        # Make prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text=f'Estimated House Price: ₹{output:,}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(debug=True)
