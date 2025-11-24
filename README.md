# ASR-Bias-Computational-Analysis
How do automated transcription biases influence downstream computational social science tasks such as sentiment analysis, misinformation detection, and stance classification on social media?

Link:https://drive.google.com/drive/folders/1gE8Sihnfq9fvp26-9-CxMwwDJitkOmue?usp=sharing

## PROJECT STRUCTURE OVERVIEW
This Google Drive folder contains all code, datasets, audio files, processed outputs, and documentation for our project.  
The complete project report is available on Overleaf:

📌 **Overleaf Report:** https://www.overleaf.com/read/kgrbhhvnkmcx#2676e0

This project is organized into **three major analytical components**, each implemented as a separate module:

1. **Stance Classification**  
2. **Sentiment Analysis**  
3. **Misinformation Alignment**

Each component has its own folder containing:
* Jupyter notebooks (Colab)
* Processed CSV files

--------------------------------------------------------------------------------------------------------------------------------
# FOLDER STRUCTURE
--------------------------------------------------------------------------------------------------------------------------------

### **1. Stance Classification/**
- Contains all notebooks, intermediate CSV files, and processed outputs related to stance labeling (Phase 1 and Phase 2).
- Uses the normalized dataset: **`final_normalized_dataset.csv`**.

---

### **2. Sentiment_Analysis/**
- Contains the full sentiment-analysis workflow:
  - DistilBERT scoring  
  - WER correlation  
  - Demographic bias  
  - Audio feature extraction  
  - Multimodal ablation  
- Uses: **`combined_audio_mapping.csv`**  
- Stores:
  - `labels_sentiment_with_wps.csv`  
  - `labels_sentiment_with_shifts.csv`  
  - `audio_features_df.csv`  
  - `platform_embeddings_with_reference.npz`

---

### **3. Misinformation Alignment/**
- Contains lexicon construction, keyword matching, semantic similarity, MAS scoring, and ML classifiers.
- Uses the text-only dataset: **`final_normalized_dataset.csv`**

---

### **4. MonoTED_Audio/**
- Raw `.sph` audio files from the MonoTED dataset.
- Referenced via paths stored in `combined_audio_mapping.csv`.
- Required for:
  - Librosa acoustic feature extraction  
  - Multimodal learning  
  - WPS computation  

---

### **5. readme.txt**
The master documentation containing pipeline descriptions and output file summaries.

---

### **6. AI ML Project PPT.pptx**
Final presentation slides.

--------------------------------------------------------------------------------------------------------------------------------
# DATASETS USED IN THE PROJECT
--------------------------------------------------------------------------------------------------------------------------------

Two closely related CSV files are used across the three analytical components.

---

## **1. `combined_audio_mapping.csv`**  
**Used for:** Sentiment Analysis  

This file is identical to `final_normalized_dataset.csv` but contains **one extra column**:

#### **Additional Column**
- `audio_filepath` → Absolute path to the `.sph` file inside `MonoTED_Audio/`

#### **Purpose**
Required for audio-based computations:
- Librosa features  
- MFCCs, energy, pauses  
- Speech-rate (WPS)  
- Multimodal classification (Audio + Text)  
- Ablation experiments  

---

## **2. `final_normalized_dataset.csv`**  
**Used for:** Stance Classification & Misinformation Alignment

This dataset contains all text-based metadata **without** the audio path column.

#### **Purpose**
Used for:
- Zero-shot stance classification  
- ML-based stance prediction  
- Misinformation lexicon + MAS computation  
- ML-based misinformation detection  

---

### **Summary:**
- Use **`combined_audio_mapping.csv`** for **audio-dependent tasks**.
- Use **`final_normalized_dataset.csv`** for **text-only tasks**.

--------------------------------------------------------------------------------------------------------------------------------
# ABOUT THE AUDIO FILES (MonoTED_Audio/)
--------------------------------------------------------------------------------------------------------------------------------

- Contains all `.sph` audio files from MonoTED.
- These files are *not embedded* inside the CSV.
- Only their absolute paths are referenced in `combined_audio_mapping.csv`.

Used exclusively in Sentiment Analysis for:
- Librosa feature extraction  
- MFCCs, energy, pauses  
- Speech-duration and WPS  
- Multimodal ablation  

--------------------------------------------------------------------------------------------------------------------------------
# SENTIMENT ANALYSIS

This section documents the full sentiment-analysis pipeline implemented in:

### **`SentimentAnalysis.ipynb`**

The workflow includes:
- BERT-based sentiment scoring  
- WPS computation  
- WER–sentiment correlations  
- Demographic bias analysis  
- Audio–text multimodal classification  
- Ablation experiments  

Below is the exact chronological methodology.

--------------------------------------------------------------------------------------------------------------------------------
## **1. Notebook: SentimentAnalysis.ipynb**
--------------------------------------------------------------------------------------------------------------------------------

### **Purpose**
Implements the complete end-to-end pipeline:

- Load transcripts & audio paths  
- DistilBERT sentiment scoring  
- Speaking-rate (WPS) calculation  
- Sentiment-shift detection  
- Error-type analysis  
- Demographic bias measurement  
- Librosa feature extraction  
- DistilBERT text embeddings  
- Multimodal modelling  
- ASR degradation evaluation  

# STEP-BY-STEP WORKFLOW
--------------------------------------------------------------------------------------------------------------------------------
## **2. Data Loading and Setup**
--------------------------------------------------------------------------------------------------------------------------------

### **Input File**
- `combined_audio_mapping.csv` → Includes:
  - Reference + ASR transcripts  
  - Audio paths  
  - Demographic metadata  
  - WER metrics  
  - Readability + WPS metadata  

### **Output**
- DataFrame `df`

--------------------------------------------------------------------------------------------------------------------------------
## **3. Sentiment Scoring using DistilBERT**
--------------------------------------------------------------------------------------------------------------------------------

### **Model**
`distilbert-base-uncased-finetuned-sst-2-english`

### **Process**
- Chunk long transcripts  
- Convert logits → softmax → continuous polarity  
- Final label = Positive / Negative / Neutral  

### **Outputs**
- `ref_sentiment` (–1 to +1)
- `label` (categorical)
- Sentiment for all 7 ASRs  
- **Saved File:** `labels_sentiment.csv`

--------------------------------------------------------------------------------------------------------------------------------
## **4. WPS (Speaking Rate) Categorization**
--------------------------------------------------------------------------------------------------------------------------------

- WPS = word_count / duration  
- Quantile-based categories:
  - Slow  
  - Moderate  
  - Fast  
  - Very Fast  

### **Saved File:**  
`labels_sentiment_with_wps.csv`

--------------------------------------------------------------------------------------------------------------------------------
## **5. Sentiment Shift Detection**
--------------------------------------------------------------------------------------------------------------------------------

### **Definition**
`shift = 1` if ASR sentiment ≠ reference label  

Outputs:
- Shift columns for all platforms  
- Logistic regression: `P(Shift | WER)`  
- Sentiment intensity × WER correlation  

### **Saved File:**  
`labels_sentiment_with_shifts.csv`

--------------------------------------------------------------------------------------------------------------------------------
## **6. Fine-Grained Sentiment Change Types**
--------------------------------------------------------------------------------------------------------------------------------

Directionality:
- Positive → Negative  
- Negative → Positive  
- Neutral → Positive  
- Neutral → Negative  
- Polarity weakening/strengthening  

Outputs:
- Change-type annotations  
- Pivot tables + plots  

--------------------------------------------------------------------------------------------------------------------------------
## **7. Sentiment Disagreement Metrics**
--------------------------------------------------------------------------------------------------------------------------------

Outputs:
- Inter-ASR correlation heatmap  
- WER distribution for agreement/disagreement  
- Global shift-rate table  

--------------------------------------------------------------------------------------------------------------------------------
## **8. Demographic Bias Evaluation**
--------------------------------------------------------------------------------------------------------------------------------

Clusters evaluated:
- Gender  
- Race  
- Language group  
- WPS category  

Metrics:
- Shift rate  
- Drift distributions  
- Vulnerability Index  

--------------------------------------------------------------------------------------------------------------------------------
## **9. Text-Only ML Benchmark (Reference Only)**
--------------------------------------------------------------------------------------------------------------------------------

Models:
- Logistic Regression  
- Naive Bayes  
- Random Forest  

Inputs:
- TF-IDF of reference transcripts  
- Reference sentiment labels  

Outputs:
- Confusion matrices  
- Accuracy & Macro-F1 baseline  

--------------------------------------------------------------------------------------------------------------------------------
## **10. Feature Extraction for Multimodal Modelling**
--------------------------------------------------------------------------------------------------------------------------------

### **A. Audio Features via Librosa**
- MFCC statistics  
- Pause ratio  
- Energy + spectral features  

**Saved File:**  
`audio_features_df.csv`

### **B. Text Embeddings via DistilBERT**
- Final hidden states  
- Mean pooled vectors  

**Saved File:**  
`platform_embeddings_with_reference.npz`

--------------------------------------------------------------------------------------------------------------------------------
## **11. Multimodal Ablation Experiment**
--------------------------------------------------------------------------------------------------------------------------------

Classifier: Random Forest  

Modes:
1. Text-Only  
2. Audio-Only  
3. Combined (Audio + Text)  

Outputs:
- Accuracy & Macro-F1 comparison  
- Ablation bar charts  

--------------------------------------------------------------------------------------------------------------------------------
## **12. ASR Degradation Analysis**
--------------------------------------------------------------------------------------------------------------------------------

Evaluates sentiment prediction drop when replacing:
**Reference text → ASR text**

Outputs:
- Accuracy drop (%) per platform  
- Degradation bar plot  

# FINAL OUTPUT FILES

All generated files stored in the project folder:

1. **labels_sentiment.csv**  
2. **labels_sentiment_with_wps.csv**  
3. **labels_sentiment_with_shifts.csv**  
4. **audio_features_df.csv**  
5. **platform_embeddings_with_reference.npz**
