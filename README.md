# AI Assistant with Streamlit and ChromaDB

This is a simple Streamlit web application that uses OpenAI's GPT-3.5-turbo model to simulate a conversational AI assistant. It also integrates with ChromaDB to store the conversation histories.

## Installation

Before running the application, you need to clone this repository to your local machine. You can do this by running the following command:

```bash
git clone https://github.com/your_username/your_repository.git
```

Replace your_username with your GitHub username and your_repository with the name of your repository.

Next, navigate to the project directory:

```bash
cd your_repository
```

Before running the application, you need to install the necessary dependencies.

```bash
pip install streamlit chromadb openai dotenv tiktoken
```

## Environment variables
You need to set the OPENAI_API_KEY environment variable for the OpenAI API. You can set it in a .env file in the project directory as follows:

```code
OPENAI_API_KEY=your_openai_api_key
```
Replace your_openai_api_key with your actual OpenAI API key.

## Local Project Directory

You need to specify your local directory for ChromaDB to persist data. Change the following line in the code:

```python
project_directory = r"Your local directory"  # Change this to your local directory
```

Replace Your local directory with your actual local directory.

## Running the Application

After setting up the environment variables and local directory, you can run the Streamlit app with the following command:

```bash
streamlit run your_python_script.py
```

Replace your_python_script.py with the name of your Python script.

## Using the AI Assistant

After running the Streamlit app, you will see a simple user interface with a text area and two buttons. You can type your message to the AI assistant in the text area, then click the "Send" button to get a response. The conversation history will be displayed below.

If you want to reset the conversation, click the "Reset" button. This will clear the conversation history in the current session.

The application also stores the conversation history in ChromaDB, with embeddings generated by the OpenAI API.

##  Contributing

If you would like to contribute to this project, please feel free to fork the repository, make changes, and create a pull request.
