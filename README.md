# 🔬 EmbedViz
EmbedViz is a Streamlit-based web application that allows users to visualize and compare text embeddings generated using Ollama models. It provides an interactive interface to input text, generate embeddings, and visualize them using various plots and similarity metrics.

<img width="1016" alt="image" src="https://github.com/user-attachments/assets/076a213c-2d4f-4e21-ab28-44415747326f">


## Features

- Generate embeddings for two text inputs using Ollama models
- Visualize embeddings using stem-like plots and dimension-wise comparison
- Calculate and display cosine similarity and Euclidean distance between embeddings
- Display raw embedding values

## Prerequisites

- Python 3.12+
- Ollama (for generating embeddings)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/NeoVand/EmbedViz.git
   cd embedding-visualizer
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Setting up Ollama

1. Install Ollama by following the instructions at [https://ollama.ai/](https://ollama.ai/)

2. Pull some embedding models:
   ```
   ollama pull nomic-embed-text
   ollama pull all-minilm
   ```

   You can pull additional models as needed.

## Running the Application

1. Ensure Ollama is running in the background.

2. Start the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and navigate to the URL displayed in the terminal (usually `http://localhost:8501`).

## Usage

1. Select an embedding model from the dropdown menu.
2. Enter two texts in the provided text areas.
3. Click "Generate Embeddings" to visualize and compare the embeddings.
4. Explore the visualizations and similarity metrics displayed on the page.

## Note

Make sure Ollama is running and accessible at `http://localhost:11434` (default URL). If you're running Ollama on a different URL, you can change it in the app settings.
