# 📚 DocuChat AI

DocuChat AI is an intelligent multi-document assistant built with Streamlit. It allows users to upload PDF documents and ask questions about the content. The app uses a language model to provide accurate answers based on the provided documents.

## Features

- Upload multiple PDF documents
- Ask questions about the uploaded documents
- Receive accurate answers with source document references
- View response time for each query
- Customizable app styling

## Demo

<video width="600" controls>
  <source src="demo/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/Pandurangmopgar/ChatWDoc.git
    cd ChatWDoc
    ```

2. Create a virtual environment and activate it:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Set up environment variables:
    - Create a `.env` file in the root directory.
    - Add your API keys to the `.env` file:
        ```
        GROQ_API_KEY=your_groq_api_key
        GOOGLE_API_KEY=your_google_api_key
        ```

## Usage

1. Run the Streamlit app:
    ```sh
    streamlit run app.py
    ```

2. Use the sidebar to upload your PDF documents.

3. Enter your question in the main panel and click "Ask" to get your answer.

## Project Structure

