import streamlit as st
import numpy as np
import requests
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, euclidean

# Constants
DEFAULT_OLLAMA_URL = 'http://localhost:11434'
APP_TITLE = "🔬 EmbedViz"
APP_ICON = "🔬"

# Ollama API functions
def check_ollama_connection(ollama_url):
    try:
        requests.get(f'{ollama_url}/api/tags', timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False

def get_ollama_models(ollama_url):
    try:
        response = requests.get(f'{ollama_url}/api/tags', timeout=5)
        response.raise_for_status()
        return [model['name'] for model in response.json()['models']]
    except requests.RequestException as e:
        st.sidebar.error(f"Error connecting to Ollama: {str(e)}")
        return []

def get_embedding(ollama_url, model, text):
    try:
        response = requests.post(
            f'{ollama_url}/api/embeddings',
            json={'model': model, 'prompt': text},
            timeout=30
        )
        response.raise_for_status()
        return response.json()['embedding']
    except requests.RequestException as e:
        st.error(f"Error getting embedding: {str(e)}")
        return None
    except KeyError as e:
        st.error(f"Unexpected response format: {str(e)}")
        st.error(f"Response content: {response.text}")
        return None

def get_model_card(ollama_url, model):
    try:
        response = requests.post(
            f'{ollama_url}/api/show',
            json={'name': model},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.sidebar.error(f"Error fetching model card: {str(e)}")
        return None

# Visualization functions
def plot_embedding(ax, embedding, color='blue', label='Embedding'):
    x = range(len(embedding))
    ax.plot(x, embedding, color=color, label=label, alpha=0.5)
    ax.scatter(x, embedding, color=color)
    for i, j in zip(x, embedding):
        ax.vlines(i, 0, j, color=color, alpha=0.2)

def plot_embeddings(embedding1, embedding2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=300)
    
    # First subplot: stem-like plot
    plot_embedding(ax1, embedding1, 'blue', 'Embedding 1')
    plot_embedding(ax1, embedding2, 'red', 'Embedding 2')
    ax1.set_xlabel('Embedding Dimension')
    ax1.set_ylabel('Value')
    ax1.set_title('Embedding Vectors Visualization')
    ax1.legend()

    # Second subplot: dimension-wise comparison plot
    num_dims = len(embedding1)
    ax2.scatter(embedding1, embedding2, color='green', alpha=0.5)
    
    ax2.set_xlabel('Embedding 1')
    ax2.set_ylabel('Embedding 2')
    ax2.set_title('Dimension-wise Comparison: Embedding 1 vs Embedding 2')
    ax2.set_aspect('equal')  # Set aspect ratio to 1
    
    # Set axes limits
    max_val = max(max(embedding1), max(embedding2))
    min_val = min(min(embedding1), min(embedding2))
    ax2.set_xlim(min_val, max_val)
    ax2.set_ylim(min_val, max_val)
    
    # Add diagonal line
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    plt.tight_layout()
    return fig

# Similarity functions
def cosine_similarity(embedding1, embedding2):
    return 1 - cosine(embedding1, embedding2)

def euclidean_distance(embedding1, embedding2):
    return euclidean(embedding1, embedding2)

# Main application
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")
    st.title(APP_TITLE)

    if 'ollama_url' not in st.session_state:
        st.session_state.ollama_url = DEFAULT_OLLAMA_URL

    with st.sidebar:
        st.header("Settings")
        
        with st.expander("🛠️ Ollama Settings", expanded=False):
            ollama_url = st.text_input("Ollama Server URL:", value=st.session_state.ollama_url)
            if ollama_url != st.session_state.ollama_url:
                st.session_state.ollama_url = ollama_url

            if not check_ollama_connection(st.session_state.ollama_url):
                st.error(f"Cannot connect to Ollama server at {st.session_state.ollama_url}. Please make sure it's running.")
            else:
                models = get_ollama_models(st.session_state.ollama_url)
                if not models:
                    st.error("No Ollama models available. Please check your Ollama installation.")

        embedding_models = get_ollama_models(st.session_state.ollama_url)
        selected_embedding_model = st.selectbox("Select the embedding model", embedding_models, key="model_select")

        if selected_embedding_model:
            model_card = get_model_card(st.session_state.ollama_url, selected_embedding_model)
            if model_card:
                model_card.pop("license",None)
                st.subheader("Model Card")
                st.json(model_card)
                    

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Text 1")
        text_input1 = st.text_area("Enter first text to embed:", "Hello, world!", key="text1", height=50)

    with col2:
        st.markdown("### Text 2")
        text_input2 = st.text_area("Enter second text to embed:", "Embedding visualization", key="text2", height=50)
    
    if st.button("Generate Embeddings", type="primary"):
        with st.spinner("Generating embeddings..."):
            embedding1 = get_embedding(st.session_state.ollama_url, selected_embedding_model, text_input1)
            embedding2 = get_embedding(st.session_state.ollama_url, selected_embedding_model, text_input2)
        
        if embedding1 and embedding2:
            
            st.subheader("📊 Embeddings Visualization")
            fig = plot_embeddings(embedding1, embedding2)
            st.pyplot(fig)
            
            st.subheader("📏 Similarity Metrics")
            cos_sim = cosine_similarity(embedding1, embedding2)
            euc_dist = euclidean_distance(embedding1, embedding2)
            
            metric_col1, metric_col2 = st.columns(2)
            with metric_col1:
                st.metric("Cosine Similarity", f"{cos_sim:.4f}")
            with metric_col2:
                st.metric("Euclidean Distance", f"{euc_dist:.4f}")
            
            st.markdown("---")
            st.subheader("🔢 Embedding Values")
            embed_col1, embed_col2 = st.columns(2)
            with embed_col1:
                st.markdown("**Embedding 1:**")
                st.json(embedding1)
            with embed_col2:
                st.markdown("**Embedding 2:**")
                st.json(embedding2)

if __name__ == "__main__":
    main()