import streamlit as st
import pandas as pd
import pickle 

def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Load the model, scaler, and feature names
def load_model():
    with open('titanic_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('feature_names.pkl', 'rb') as feature_file:
        feature_names = pickle.load(feature_file)
    return model, scaler, feature_names


# Preprocess user input for prediction
def preprocess_input(Age, Sex, Fare, Pclass, Sibsp, Parch, Embarked, feature_names):
    # Create a dataframe with the same structure as the training data
    data = {'Pclass': Pclass,
            'Age': Age,
            'Fare': Fare,
            'SibSp': Sibsp,
            'Parch': Parch,
            'Sex_male': 1 if Sex == 'Male' else 0,
            'Embarked_Q': 1 if Embarked == 'Q' else 0,
            'Embarked_S': 1 if Embarked == 'S' else 0}

    df = pd.DataFrame(data, index=[0])

    # Reorder the dataframe columns to match the feature names used during training
    df = df[feature_names]
    return df


# Streamlit UI
def main(): 
     # Set a background image
    image_path = '/Users/macvision/PycharmProjects/Ecode_Code_and_GUIS/EcodeCamp_Deployments/787014.jpg'  # Replace with the path to your local image
    img_base64 = get_base64_image(image_path)

    if img_base64:  # Check if image was loaded successfully
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{img_base64}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Background image could not be loaded.")
    st.title("Titanic Survival Prediction")

    st.write("Enter the following information to predict survival on the Titanic:")

    # Input fields for user
    Pclass = st.selectbox("Passenger Class", [1, 2, 3], index=0)
    Sex = st.selectbox("Sex", ['Male', 'Female'], index=0)
    Age = st.slider("Age", 0, 100, 30)
    Sibsp = st.number_input("Number of Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)
    Parch = st.number_input("Number of Parents/Children Aboard", min_value=0, max_value=10, value=0)
    Fare = st.slider("Ticket Fare", 0, 500, 30)
    Embarked = st.selectbox("Port of Embarkation", ['C', 'Q', 'S'], index=2)

    # Button to trigger prediction
    if st.button("Predict"):
        # Load the trained model, scaler, and feature names
        model, scaler, feature_names = load_model()

        # Preprocess the input
        input_data = preprocess_input(Age, Sex, Fare, Pclass, Sibsp, Parch, Embarked, feature_names)

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Make a prediction
        prediction = model.predict(input_data_scaled)

        # Display the result
        if prediction[0] == 1:
            st.success("The passenger would have survived!")
        else:
            st.error("The passenger would not have survived.")


if __name__ == "__main__":
    main()
