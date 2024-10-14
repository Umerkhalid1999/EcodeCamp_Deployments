import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os
import base64


def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        return encoded_string
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Path to the saved model (ensure it's a .h5 file)
MODEL_PATH = "saved_model.h5"


# Streamlit App
def main():
    # Set a background image
    image_path = 'img.jpeg'  # Replace with the path to your local image
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
    st.title("Stock Price Prediction App")

    # Load the saved .h5 model
    try:
        model = load_model(MODEL_PATH)
        st.success("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return

    # Allow the user to make a single prediction by providing custom input data
    st.subheader("Make a Prediction")

    # Create input fields for custom stock data (Open, High, Low, Volume)
    open_price = st.number_input("Open Price", value=0.0)
    high_price = st.number_input("High Price", value=0.0)
    low_price = st.number_input("Low Price", value=0.0)
    volume = st.number_input("Volume", value=0.0)

    if st.button("Predict"):
        # Prepare custom input for prediction (reshape as needed for your model)
        custom_input = np.array([[open_price, high_price, low_price, volume]])

        # Ensure the input shape matches what the model expects
        time_step = 2  # adjust if your model needs a specific time step
        custom_input_seq = np.tile(custom_input, (time_step, 1)).reshape((1, time_step, -1))

        # Make prediction using the loaded model
        try:
            custom_prediction = model.predict(custom_input_seq)
            predicted_price = custom_prediction[0][0]
            st.write(f"Predicted Stock Price: {predicted_price}")

            # Plotting the prediction along with previous prices
            st.subheader("Stock Price Predictions vs Time")

            # Example of time-series data (You can replace this with actual data if available)
            time_points = np.arange(10)  # Example: last 10 days
            actual_prices = np.random.uniform(100, 200, size=10)  # Random actual prices as example

            # Add the predicted price to the graph
            extended_time_points = np.append(time_points, time_points[-1] + 1)
            extended_prices = np.append(actual_prices, predicted_price)

            fig, ax = plt.subplots()
            ax.plot(time_points, actual_prices, label="Actual Prices", color="blue")
            ax.plot(extended_time_points, extended_prices, label="Predicted Price", color="red", marker='o')
            ax.set_xlabel("Time")
            ax.set_ylabel("Stock Price")
            ax.legend()

            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")


if __name__ == "__main__":
    main()
