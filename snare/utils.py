import os
import subprocess
import uuid
import numpy as np

MAXSEQ = 4980
NUM_FEATURE = 20

# Path to current file (e.g., snare/utils.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root: one level up from snare/
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

# Paths relative to project structure
DEFAULT_BLAST_DB_PATH = os.path.join(PROJECT_ROOT, "blastdb", "swissprot")
DEFAULT_PSIBLAST_PATH = os.path.join(PROJECT_ROOT, "bin", "psiblast")

def generate_pssm_from_fasta(
    fasta_seq,
    blast_db_path=DEFAULT_BLAST_DB_PATH,
    psiblast_path=DEFAULT_PSIBLAST_PATH
):
    """
    Generate a PSSM feature vector from a FASTA string using PSI-BLAST.
    
    Args:
        fasta_seq (str): A protein sequence string.
        blast_db_path (str): Path to your BLAST database.
        psiblast_path (str): Full path to psiblast binary.
    
    Returns:
        np.ndarray: A flattened PSSM feature array of shape (MAXSEQ * NUM_FEATURE,)
    """
    temp_dir = "/tmp"
    tmp_id = str(uuid.uuid4())
    fasta_path = os.path.join(temp_dir, f"temp_{tmp_id}.fasta")
    pssm_path = os.path.join(temp_dir, f"temp_{tmp_id}.pssm")
    blast_output = os.path.join(temp_dir, f"temp_{tmp_id}.blastout")

    try:
        # Clean and validate input
        clean_seq = fasta_seq.strip().replace(" ", "").replace("\n", "")
        if not clean_seq or any(c.isdigit() for c in clean_seq) or ">" in clean_seq:
            raise ValueError("Invalid protein sequence format.")

        # Write to temporary FASTA file
        with open(fasta_path, "w") as f:
            f.write(">query\n")
            f.write(clean_seq + "\n")

        # Run PSI-BLAST
        cmd = [
            psiblast_path,
            "-query", fasta_path,
            "-db", blast_db_path,
            "-num_threads", "1",
            "-num_iterations", "3",
            "-out_ascii_pssm", pssm_path,
            "-out", blast_output
        ]
        subprocess.run(cmd, check=True)

        # Confirm output file exists
        if not os.path.exists(pssm_path):
            raise RuntimeError("PSI-BLAST failed: PSSM file was not generated.")

        # Parse the PSSM output
        with open(pssm_path, "r") as f:
            lines = f.readlines()[3:-6]

        features = []
        for line in lines:
            parts = line.split()
            if len(parts) < 22:
                continue
            features.append([float(x) for x in parts[2:22]])

        # Pad or truncate to MAXSEQ
        while len(features) < MAXSEQ:
            features.append([0.0] * NUM_FEATURE)
        features = features[:MAXSEQ]

        return np.array(features).flatten()

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"PSI-BLAST failed with return code {e.returncode}. "
            f"Command: {' '.join(cmd)}. Check output: {blast_output}"
        )
    except Exception as e:
        raise RuntimeError(f"Error during PSSM parsing: {e}")
    finally:
        for path in [fasta_path, pssm_path, blast_output]:
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception:
                pass