# 🌐 English-to-Spanish Translation using Transformer (PyTorch NLP)

This repository presents a research-grade implementation of a Transformer-based neural machine translation (NMT) system from English to Spanish. Built entirely from scratch using PyTorch, it demonstrates deep understanding of sequence modeling, attention mechanisms, and modern NLP engineering practices.

## 📌 Project Overview

The goal is to translate English sentences into Spanish using a custom Transformer architecture. Unlike plug-and-play pre-trained models, this project emphasizes architectural transparency, reproducibility, and modularity—ideal for learning, experimentation, and extension.

## 🧠 Key Concepts

- **Neural Machine Translation (NMT)**
- **Transformer Architecture** (Vaswani et al., 2017)
- **Multi-Head Self-Attention**
- **Positional Encoding**
- **Custom Tokenization and Vocabulary Building**
- **Teacher Forcing and Masking in Training**
- **BLEU Score Evaluation**

## 🚀 Technologies Used

- **PyTorch**: Core deep learning framework
- **Hugging Face Tokenizers**: Fast, language-aware tokenization using `ByteLevelBPETokenizer`
- **Torchtext / Custom Dataset Loader**: Preprocessing and batching
- **NumPy & Matplotlib**: Data manipulation and visualization
- **Jupyter Notebook**: Exploratory data analysis and debugging
- **Python Scripts**: Modular training and model definition

## 🧠 Model Architecture

### 🔹 Encoder
- Embedding + Positional Encoding
- Multi-head Self-Attention
- Feedforward Network
- Layer Normalization + Residual Connections

### 🔹 Decoder
- Masked Multi-head Self-Attention
- Encoder-Decoder Attention
- Feedforward Network
- Final Linear Layer + Softmax

### 🔹 Training Strategy
- CrossEntropyLoss with padding mask
- Adam optimizer with learning rate scheduling
- Teacher forcing for faster convergence
- BLEU score for evaluation

## 🧩 Pipeline Highlights

- ✅ **Tokenizer Integration**:
  - Hugging Face `ByteLevelBPETokenizer` for English and Spanish
  - Saved vocabularies (`tokenizer_en.json`, `tokenizer_es.json`) for reproducibility
  - Efficient padding, masking, and batching

- 🔄 **Modular Design**:
  - Clear separation of data loading, model definition, training, and visualization
  - Configurable hyperparameters via `config.py`

- 📊 **Visualization & Analysis**:
  - `Dataset_Visualization.ipynb` for token distribution and sentence length analysis
  - `img.ipynb` for optional attention map visualization

## 📁 Repository Structure

```text
├── Dataset.py                      # Custom dataset loader and preprocessing  
├── Dataset_Visualization.ipynb    # EDA and token distribution plots  
├── Model.py                        # Transformer model implementation  
├── config.py                       # Hyperparameters and constants  
├── train.py                        # Training loop and evaluation logic  
├── img.ipynb                       # Optional image-based visualization  
├── tokenizer_en.json               # English tokenizer vocabulary  
├── tokenizer_es.json               # Spanish tokenizer vocabulary  
├── requirement.txt                 # Python dependencies  
├── .gitignore                      # Git ignore rules  
├── runs/transformer_model/        # Saved model checkpoints  
```
## 🛠️ How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/arvind207kumar/Pytorch-nlp-Eng2Es-using-Transformer.git
   cd Pytorch-nlp-Eng2Es-using-Transformer
2. **Install dependencies**:
   ```bash
   pip install -r requirement.txt
3. **Train the model**:
   ```bash
   python train.py
   ```
## 📊 Explore the Dataset

- Open `Dataset_Visualization.ipynb` to inspect:
  - Token distributions  
  - Sentence length statistics  

## 🧪 Visualize or Test Translations

- Use `img.ipynb` for:
  - Translation inference  
  - Attention map visualization  

- Alternatively, extend `train.py` to support:
  - Custom inference workflows  
  - Attention-based interpretability  

---

## 📈 Results

- ✅ **BLEU Score**: Competitive translation quality on test samples  
- 🔍 **Qualitative Evaluation**: Translations show semantic alignment and grammatical correctness  
- 📉 **Loss Curve**: Stable convergence across epochs  
- 🧪 **Inference**: Supports sentence-level translation with optional attention visualization  

---

## 🔮 Future Work

- 📦 **Model Export**: Convert to TorchScript or ONNX for deployment  
- 🧠 **Pretrained Embeddings**: Integrate GloVe or FastText for richer semantic understanding  
- 🧪 **Hyperparameter Tuning**: Use Optuna for automated search  
- 🌍 **Multilingual Extension**: Extend to other language pairs using shared vocabularies  
- 📱 **Web Interface**: Build a Streamlit or Gradio app for interactive translation  
- 🧩 **Attention Visualization**: Add heatmaps to interpret model focus during translation  
   
   
