# snare-web/app.py

from flask import Flask, render_template, request
import pickle
from Bio import SeqIO
import io

app = Flask(__name__)

# ─── Load your trained pipeline ────────────────────────────────────────────────
with open("model_pipeline/feature_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_fasta = ""
    if request.method == "POST":
        input_fasta = request.form.get("sequence", "").strip()
        try:
            # Parse the textarea FASTA into a single sequence string
            handle = io.StringIO(input_fasta)
            records = list(SeqIO.parse(handle, "fasta"))
            if not records:
                raise ValueError("No valid FASTA record found.")
            # (if multiple records, we just take the first)
            seq_str = str(records[0].seq).upper().strip()

            # Predict with your pipeline
            pred = pipeline.predict([seq_str])[0]
            result = "SNARE" if pred == 1 else "Non-SNARE"

        except Exception as e:
            result = f"Error: Prediction error: {e}"

    return render_template(
        "index.html",
        result=result,
        sequence=input_fasta
    )


if __name__ == "__main__":
    # debug=True will auto-reload on code changes
    app.run(host="0.0.0.0", port=5001, debug=True)

