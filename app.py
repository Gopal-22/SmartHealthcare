from flask import Flask, render_template, request
import numpy as np
import pickle
from PIL import Image
import cv2
import imutils
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

# ================== Flask Routes ==================
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/heart_disease')
def heart_disease():
    return render_template('heart_disease.html')

@app.route('/brain_tumor')
def brain_tumor():
    return render_template('brain_tumor.html')

@app.route('/cancer')
def cancer():
    return render_template('lung_cancer.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/dpf')
def dpf():
    return render_template('dpf-calculator.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/bmi')
def bmi():
    return render_template('bmi-calculator.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

# ================== Model Loading with Error Handling ==================
def load_pickle_model(filename):
    """Safely load a pickle model with error handling."""
    try:
        with open(filename, 'rb') as file:
            model = pickle.load(file)
        print(f"Loaded model: {filename}")
        return model
    except Exception as e:
        print(f"Error loading model {filename}: {str(e)}")
        return None  # Return None if model loading fails

# Load models safely
db_model = load_pickle_model('diabetes_pred.sav')

# Load heart disease model using joblib
try:
    hd_model = joblib.load('heart_disease_random_forest.sav')
    print("Heart disease model loaded successfully.")
except Exception as e:
    print(f"Error loading heart disease model: {str(e)}")
    hd_model = None

# Load lung cancer model using joblib
# Load lung cancer model using joblib
try:
    lc_model = joblib.load('lung_cancer_random_forest.sav')  # Update the path if necessary
    print("Lung cancer model loaded successfully.")
except FileNotFoundError:
    print("🚨 Error: 'lung_cancer_random_forest.sav' file not found. Please ensure the file is in the correct directory.")
    lc_model = None
except Exception as e:
    print(f"🚨 Error loading lung cancer model: {str(e)}")
    lc_model = None

# Load brain tumor model
try:
    print("Loading brain tumor model...")
    model = load_model('brain_tumor_final_model.h5')  # Ensure this file exists in the correct directory
    print("Brain tumor model loaded successfully.")
except Exception as e:
    print(f"Error loading brain tumor model: {str(e)}")
    model = None

# ================== Diabetes Prediction ==================
@app.route('/db', methods=['POST'])
def db():
    if db_model is None:
        return "Error: Diabetes prediction model failed to load.", 500

    try:
        features = [request.form[key] for key in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                                                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
        arr = np.array(features, dtype=np.float32).reshape(1, -1)
        pred = db_model.predict(arr)

        x = "Our model predicts that this is a positive case for Diabetes." if pred[0] == 1 else \
            "Our model predicts that this is a negative case for Diabetes."

        return render_template('diabetes.html', title=x)
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

# ================== Heart Disease Prediction ==================
@app.route('/hd', methods=['POST'])
def hd():
    if hd_model is None:
        return "Error: Heart disease prediction model failed to load.", 500

    try:
        # Ensure all 13 features are provided
        required_features = ['age', 'sex', 'chestpain', 'restingbp', 'chol', 'fastingbs', 
                             'rcg', 'maxhr', 'ea', 'oldpeak', 'slope', 'ca', 'thal']
        features = []
        for key in required_features:
            if key in request.form:
                features.append(request.form[key])
            else:
                # Handle missing features (e.g., set to 0 or a default value)
                features.append(0)  # Replace with appropriate default value if needed

        # Convert features to a NumPy array
        arr = np.array(features, dtype=np.float32).reshape(1, -1)

        # Make prediction
        pred = hd_model.predict(arr)

        # Interpret the result
        x = "Our model predicts that this is a positive case for heart disease." if pred[0] == 1 else \
            "Our model predicts that this is a negative case for heart disease."

        return render_template('heart_disease.html', title=x)
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

# ================== Lung Cancer Prediction ==================
@app.route('/lc', methods=['POST'])
def lc():
    if lc_model is None:
        return "Error: Lung cancer prediction model failed to load.", 500

    try:
        features = [request.form[key] for key in ['gender', 'age', 'smoking', 'yellowfing', 'anxiety', 'pp', 
                                                  'chronic', 'fatigue', 'allergy', 'wheezing', 'alcohol', 
                                                  'coughing', 'shortness', 'swalloing', 'chestpain']]
        arr = np.array(features, dtype=np.float32).reshape(1, -1)
        pred = lc_model.predict(arr)

        x = "Our model predicts that this is a positive case for Lung Cancer." if pred[0] >= 0.5 else \
            "Our model predicts that this is a negative case for Lung Cancer."

        return render_template('lung_cancer.html', title=x)
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

# ================== Brain Tumor Prediction ==================
def crop_brain_contour(image):
    """Preprocess brain scan image for prediction."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    extTop, extBot, extLeft, extRight = tuple(c[c[:, :, 1].argmin()][0]), tuple(c[c[:, :, 1].argmax()][0]), \
                                        tuple(c[c[:, :, 0].argmin()][0]), tuple(c[c[:, :, 0].argmax()][0])
    return image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

@app.route('/bt', methods=['POST'])
def bt():
    if model is None:
        return "Error: Brain tumor model failed to load.", 500

    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    try:
        # Load and preprocess the image
        img_pil = Image.open(file).convert('RGB')
        img = np.array(img_pil)

        # Check if the image is empty or invalid
        if img is None or img.size == 0:
            return "Error: Uploaded image is invalid or empty.", 400

        img = crop_brain_contour(img)
        img = cv2.resize(img, (150, 150))  # Resize to match the model's input shape

        # Normalize the image
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)

        # Interpret the result
        x = "The model predicts this is a positive case for brain tumor." if np.round(prediction[0]) == 1 else \
            "The model predicts that this is a negative case for brain tumor."

        return render_template('brain_tumor.html', title=x)
    except Exception as e:
        return f"Prediction error: {str(e)}", 500

# ================== Run the Flask App ==================
if __name__ == "__main__":
    app.run(debug=True)