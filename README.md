# 🧠 SNARE Protein Predictor

This project is a deep learning-based web application for predicting whether a given protein sequence is a SNARE or non-SNARE protein. It combines a PSI-BLAST-powered PSSM feature generator with a DeepScan CNN model trained on protein sequences.

## 🚀 Features

- 🧬 Accepts input in FASTA format via form or file upload
- 🧠 Predicts SNARE protein likelihood using a DeepScan CNN model
- 📊 Logs metrics: Sensitivity, Specificity, Accuracy, MCC
- 🌐 Flask frontend with modern UI (demo mode supported)
- ⚙️ Modular Python architecture for flexibility and deployment

---

## 📂 Project Structure

```
snare-predictor/
├── app.py                   # Flask entrypoint
├── snare/                   # Core Python logic
│   ├── __init__.py
│   ├── model.py             # DeepScan CNN definition
│   ├── predict.py           # Inference logic
│   ├── utils.py             # PSI-BLAST + PSSM feature encoder
│   
├── static/                  # CSS and image assets
│   └── css/
│       └── styles.css
│   └── img/
│       └── hero.jpg
├── templates/
│   └── index.html           # Main HTML interface
├── requirements.txt         # Dependencies
├── .gitignore               # Clean-up rules
├── test.py                  # Unit test for local testing
├── training_scripts/        # Model training workflow
│   └── snare_deepscan_training.py
├── README.md                # You’re here
└── dataset/, pssm/, results/
```

---

## 💡 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare BLAST Tools
Make sure:
- `psiblast` is installed
- You have a valid BLAST database (e.g., SwissProt)
- Update your path in `snare/utils.py`:

```python
generate_pssm_from_fasta(seq, blast_db_path="/Users/yourname/blastdb/swissprot")
```

---

## 🧪 Try it Locally

### Web Interface:
```bash
python app.py
```
Visit: [http://localhost:5001](http://localhost:5001)

Paste a protein sequence:
```fasta
>seq1
MENSDSSNNGG...
```
Or click **Try a Demo Sequence** in the navbar.

---

## 🧠 Model Summary

- **Architecture**: DeepScan CNN with multi-scale convolutions
- **Input**: 20-dim PSSM × 4980 residues (padded/truncated)
- **Output**: Binary classification (SNARE / Non-SNARE)
- **Accuracy**:
  - Training: **91.9%**
  - Validation: **90.4%**
- **Weights Used**: `model_pipeline/weights.09.weights.h5`

---

## 🔗 Credits
Built by **Arri Hantz Max Nurbolot**, with support from ChatGPT ✨
**Inspired by:**
“Identifying SNARE Proteins Using an Alignment-Free Method Based on Multiscan Convolutional Neural Network and PSSM Profiles”
by **Quang-Hien Kha, Quang-Thai Ho, and Nguyen Quoc Khanh Lee**
