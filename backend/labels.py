"""
Plain Python constants shared between model.py (training/architecture,
needs TensorFlow) and app.py (deployed inference, deliberately does NOT
import TensorFlow - see app.py's docstring for why). Keeping these here
means app.py never has to import model.py just to get class names,
which would otherwise drag in TensorFlow's full import overhead and
defeat the point of using the lightweight ai-edge-litert interpreter.
"""

IMG_SIZE = 224
NUM_CLASSES = 7
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

CLASS_INFO = {
    "akiec": {"label": "Actinic Keratoses / Intraepithelial Carcinoma", "risk": "high"},
    "bcc": {"label": "Basal Cell Carcinoma", "risk": "high"},
    "bkl": {"label": "Benign Keratosis-like Lesion", "risk": "low"},
    "df": {"label": "Dermatofibroma", "risk": "low"},
    "mel": {"label": "Melanoma", "risk": "high"},
    "nv": {"label": "Melanocytic Nevus (mole)", "risk": "low"},
    "vasc": {"label": "Vascular Lesion", "risk": "low"},
}
