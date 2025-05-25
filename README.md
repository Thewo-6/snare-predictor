SNARE Predictor

A web-based tool for predicting SNARE‑motif proteins from amino acid FASTA sequences, built with Flask and scikit‑learn.

Features

Amino Acid Composition (AAC) and Dipeptide Composition (DPC) feature extraction

Support Vector Machine (SVM) classification

Simple Flask web interface for sequence input and prediction

Repository Structure

snare-web/
├── app.py                  # Flask application
├── features.py             # AACTransformer & DPCTransformer classes
├── train_and_pickle.py     # Train & pickle the feature pipeline + model
├── model_pipeline/         # Saved pipeline and model
│   └── feature_pipeline.pkl
├── data/
│   ├── train_snare.fasta
│   ├── train_non_snare.fasta
│   ├── test_snare.fasta
│   └── test_non_snare.fasta
├── static/                 # CSS, JS, and background image assets
├── templates/
│   └── index.html          # Main entrypoint template
├── requirements.txt        # Python dependencies
└── README.md               # This file

Installation

Clone the repository

git clone https://github.com/ArriSnnow/snare-predictor.git
cd snare-predictor/snare-web

Create a virtual environment and activate it

python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate     # Windows

Install dependencies

pip install -r requirements.txt

Model Training

Note: A pre-trained pipeline is provided in model_pipeline/feature_pipeline.pkl. To retrain from scratch:

python train_and_pickle.py

This will:

Read data/train_snare.fasta and data/train_non_snare.fasta

Extract AAC & DPC features

Scale features and train an SVM

Save the pipeline to model_pipeline/feature_pipeline.pkl

Running the Web App

export FLASK_APP=app.py
flask run --port 5001   # or python app.py

Open your browser at http://127.0.0.1:5001 (or the port you specified) to access the predictor.

Usage

Paste a FASTA sequence (with header line starting >) into the input box.

Click Predict.

View the result: SNARE or Non‑SNARE.
