INTERFACE:-
![Screenshot (26)](https://github.com/Eswar1254/project_chatbot_bert/assets/116758728/1c35b89e-25a8-4776-ba04-6c8cdcc9d9f0)

SYSTEM ARCHITECTURE:-
![Screenshot (21)](https://github.com/Eswar1254/project_chatbot_bert/assets/116758728/aadac449-905e-474d-96ad-a06646bf8ea5)



Chatbot Application:-

This is a simple chatbot application built using Streamlit, NLTK, and Keras. The chatbot is trained on a dataset of intents and can respond to user messages based on the predicted intent.

Features:-
User-friendly interface for interacting with the chatbot
Trained model for predicting intents of user messages
Responses generated based on the predicted intent

Requirements:-

Python 3.x
Streamlit
NLTK
Keras
Other dependencies specified in requirements.txt
Installation
Clone this repository to your local machine:


git clone https://github.com/Eswar1254/project_chatbot_bert.git
Navigate to the project directory:

Install the required dependencies using pip:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py

Open your web browser and go to http://localhost:8501 to interact with the chatbot.

Usage:-

Enter a message in the text input field labeled "You:".
Click the "Send" button to send the message to the chatbot.
The chatbot will respond with a message in the text area labeled "Bot:".
File Structure:-
app.py: Main Python script containing the Streamlit application code.
chatbot_model.h5: Pre-trained Keras model for predicting intents.
intents.json: JSON file containing the dataset of intents used for training the chatbot.
words.pkl: Pickle file containing the vocabulary of words used in the training data.
classes.pkl: Pickle file containing the list of intents/classes used in the training data.

Contributing:-
Contributions are welcome! If you find any issues or have suggestions for improvement, feel free to open an issue or create a pull request.



