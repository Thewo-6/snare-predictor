from flask import Flask, render_template, request
from Bio import SeqIO
import numpy as np
import tensorflow as tf
import io
import os

from snare.model import DeepScan
from snare.utils import generate_pssm_from_fasta

# Constants
MAXSEQ = 4980
NUM_FEATURE = 20
CLASS_NAMES = ['Non-SNARE', 'SNARE']

# Initialize Flask app
app = Flask(__name__)

# Load model
model = DeepScan(num_filters=256, num_hidden=128)
_ = model(tf.random.normal([1, 1, MAXSEQ, NUM_FEATURE]))  # Build model
model.load_weights("snare/logs/weights.09.weights.h5")  # Update path if needed

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    probability = None
    sequence_input = ""

    if request.args.get("demo") == "true":
        sequence_input = """>snare_006
        MAAPESGPALSPGTAEGEEETILYDLLVNTEWPPETEVQPRGNQKHGASFIITKAIRDRL
        LFLRQYIWYSPAPFLLPDGLVRLVNKQINWHLVLASNGKLLAAVQDQCVEIRSAKDDFTS
        IIGKCQVPKDPKPQWRRVAWSYDCTLLAYAESTGTVRVFDLMGSELFVISPASSFIGDLS
        YAIAGLIFLEYKASAQWSAELLVINYRGELRSYLVSVGTNQSYQESHCFSFSSHYPHGIN
        TAIYHPGHRLLLVGGCETAEVGMSKASSCGLSAWRVLSGSPYYKQVTNGGDGVTAVPKTL
        GLLRMLSVKFYSRQGQEQDGIFKMSLSPDGMLLAAIHFSGKLSIWAIPSLKQQGEWGQNE
        QPGYDDLNPDWRLSTEKRKKIKDKESFYPLIDVNWWADSAVTLARCSGALTVSSVKTLKN
        LLGKSCEWFEPSPQVTATHDGGFLSLECEIKLAPKRSRLETRAGEEDEGEEDSDSDYEIS
        AKARYFGYIKQGLYLVTEMERFAPPRKRPRTITKNYRLVSLRSTTPEELYQRKIESEEYE
        EALSLAHTYGLDTDLVYQRQWRKSAVNVASIQNYLSKIKKRSWVLHECLERVPENVDAAK
        ELLQYGLKGTDLEALLAIGKGADDGRFTLPGEIDIDSISYEELSPPDEEPAKNKKEKELK
        KRQELLKLVNFSKLTLEQKELCRCRRKLLTYLDRLATYEEILGVPHASEQRYDAEFFKKF
        RNQNIVLSARTYAQESNVQALEILFTYHGSDLLPHRLAILSNFPETTSPHEYSVLLPEAC
        FNGDSLMIIPWHEHKHRAKDWCEELACRMVVEPNLQDESEFLYAAQPELLRFRMTQLTVE
        KVMDWYQTRAEEIEHYARQVDCALSLIRLGMERNIPGLLVLCDNLVTLETLVYEARCDVT
        LTLKELQQMKDIEKLRLLMNSCSEDKYVTSAYQWMVPFLHRCEKQSPGVANELLKEYLVT
        LAKGDLKFPLKIFQHSKPDLQQKIIPDQDQLMAIALECIYTCERNDQLCLCYDLLECLPE
        RGYGDKTEATTKLHDMVDQLEQILSVSELLEKHGLEKPISFVKNTQSSSEEARKLMVRLT
        RHTGRKQPPVSESHWRTLLQDMLTMQQNVYTCLDSDACYEIFTESLLCSSRLENIHLAGQ
        MMHCSACSENPPAGIAHKGKPHYRVSYEKSIDLVLAASREYFNSSTNLTDSCMDLARCCL
        QLITDRPPAIQEELDLIQAVGCLEEFGVKILPLQVRLCPDRISLIKECISQSPTCYKQST
        KLLGLAELLRVAGENPEERRGQVLILLVEQALRFHDYKAASMHCQELMATGYPKSWDVCS
        QLGQSEGYQDLATRQELMAFALTHCPPSSIELLLAASSSLQTEILYQRVNFQIHHEGGEN
        ISASPLTSKAVQEDEVGVPGSNSADLLRWTTATTMKVLSNTTTTTKAVLQAVSDGQWWKK
        SLTYLRPLQGQKCGGAYQIGTTANEDLEKQGCHPFYESVISNPFVAESEGTYDTYQHVPV
        ESFAEVLLRTGKLAEAKNKGEVFPTTEVLLQLASEALPNDMTLALAYLLALPQVLDANRC
        FEKQSPSALSLQLAAYYYSLQIYARLAPCFRDKCHPLYRADPKELIKMVTRHVTRHEHEA
        WPEDLISLTKQLHCYNERLLDFTQAQILQGLRKGVDVQRFTADDQYKRETILGLAETLEE
        SVYSIAISLAQRYSVSRWEVFMTHLEFLFTDSGLSTLEIENRAQDLHLFETLKTDPEAFH
        QHMVKYIYPTIGGFDHERLQYYFTLLENCGCADLGNCAIKPETHIRLLKKFKVVASGLNY
        KKLTDENMSPLEALEPVLSSQNILSISKLVPKIPEKDGQMLSPSSLYTIWLQKLFWTGDP
        HLIKQVPGSSPEWLHAYDVCMKYFDRLHPGDLITVVDAVTFSPKAVTKLSVEARKEMTRK
        AIKTVKHFIEKPRKRNSEDEAQEAKDSKVTYADTLNHLEKSLAHLETLSHSFILSLKNSE
        QETLQKYSHLYDLSRSEKEKLHDEAVAICLDGQPLAMIQQLLEVAVGPLDISPKDIVQSA
        IMKIISALSGGSADLGGPRDPLKVLEGVVAAVHASVDKGEELVSPEDLLEWLRPFCADDA
        WPVRPRIHVLQILGQSFHLTEEDSKLLVFFRTEAILKASWPQRQVDIADIENEENRYCLF
        MELLESSHHEAEFQHLVLLLQAWPPMKSEYVITNNPWVRLATVMLTRCTMENKEGLGNEV
        LKMCRSLYNTKQMLPAEGVKELCLLLLNQSLLLPSLKLLLESRDEHLHEMALEQITAVTT
        VNDSNCDQELLSLLLDAKLLVKCVSTPFYPRIVDHLLASLQQGRWDAEELGRHLREAGHE
        AEAGSLLLAVRGTHQAFRTFSTALRAAQHWV"""

    if request.method == "POST":
        sequence_input = request.form.get("sequence", "").strip()
        uploaded_file = request.files.get("fasta_file")

        try:
            seq = ""

            # If file is uploaded, extract sequence from file
            if uploaded_file and uploaded_file.filename:
                contents = uploaded_file.read().decode("utf-8")
                handle = io.StringIO(contents)
                records = list(SeqIO.parse(handle, "fasta"))
                if not records:
                    raise ValueError("No valid FASTA record found in uploaded file.")
                seq = str(records[0].seq).strip()

            # Else use text area input
            elif sequence_input:
                handle = io.StringIO(sequence_input)
                records = list(SeqIO.parse(handle, "fasta"))
                if not records:
                    raise ValueError("No valid FASTA sequence found in text input.")
                seq = str(records[0].seq).strip()

            else:
                raise ValueError("Please provide a sequence via text or file upload.")

            # Generate PSSM feature vector
            features = generate_pssm_from_fasta(seq)
            if features.shape != (MAXSEQ * NUM_FEATURE,):
                raise ValueError("Feature vector shape mismatch.")

            features = np.reshape(features, [1, 1, MAXSEQ, NUM_FEATURE])

            # Predict
            probs = model.predict(features, verbose=0)[0]
            label = np.argmax(probs)
            result = CLASS_NAMES[label]
            probability = f"Confidence: {probs[label]:.2%}"

        except Exception as e:
            result = f"Error: {e}"

    return render_template(
        "index.html",
        result=result,
        probability=probability,
        sequence=sequence_input
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)