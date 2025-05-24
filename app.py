from flask import Flask, request, render_template
import pickle
from Bio import SeqIO
import io

app = Flask(__name__)

# Load feature pipeline & model (place your .pkl files into model_pipeline/)
with open('model_pipeline/feature_pipeline.pkl', 'rb') as f:
    feature_pipe = pickle.load(f)
with open('model_pipeline/snare_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        fasta = request.form['sequence']
        record = next(SeqIO.parse(io.StringIO(fasta), 'fasta'))
        X = feature_pipe.transform([str(record.seq)])
        pred = model.predict(X)[0]
        prediction = 'SNARE' if pred == 1 else 'Non-SNARE'
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
