from sklearn.ensemble import RandomForestClassifier as Classifier
import joblib
from PIL import Image as image
from PIL.Image import Image

pedrita = joblib.load('Pedrita.joblib')

def load_image(fname: str) -> np.ndarray:
    foto = image.open(fname)
    foto = foto.convert('L').resize((64,64))
    return np.array(foto).flatten()

def predict_image(model: Classifier, fname: str):
	pixels = load_image(fname)
	return model.predict([pixels])[0]

