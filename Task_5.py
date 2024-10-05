import streamlit as st
import sklearn
import pickle

# Manually opening and closing the file
with open('/Users/macvision/PycharmProjects/Ecode_Code_and_GUIS/EcodeCamp_Internships/failure (1).pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully.")



# Title
st.title('Heart Failure Prediction App')

# Input fields for new data based on the features you provided
age = st.number_input('Age', min_value=0, max_value=100, value=52)
sex = st.selectbox('Sex (1 = Male, 0 = Female)', [0, 1])
cp = st.selectbox('Chest Pain Type (0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure (in mm Hg)', min_value=50, max_value=200, value=125)
chol = st.number_input('Serum Cholesterol (mg/dl)', min_value=100, max_value=600, value=212)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)', [0, 1])
restecg = st.selectbox('Resting ECG (0 = Normal, 1 = Abnormality, 2 = Probable/Definite)', [0, 1, 2])
thalach = st.number_input('Maximum Heart Rate Achieved', min_value=50, max_value=250, value=168)
exang = st.selectbox('Exercise Induced Angina (1 = Yes, 0 = No)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment (0 = Upsloping, 1 = Flat, 2 = Downsloping)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (0-3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)', [1, 2, 3])

# Predict button
if st.button('Predict'):
    # Input data as a 2D array
    new_data = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]

    # Prediction
    prediction = model.predict(new_data)

    # Display result
    if prediction[0] == 1:
        st.write("The model predicts: **Heart Failure Detected**.")
    else:
        st.write("The model predicts: **No Heart Failure Detected**.")
