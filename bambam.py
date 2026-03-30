from __future__ import annotations
import torch, timm, numpy as np, joblib
from glob import glob
from PIL import Image as image
from PIL.Image import Image

from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.calibration import CalibratedClassifierCV as Calibrator

from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

try: import torch_directml # type: ignore
except ImportError: 
	torch_directml = None

DEFAULT_MODEL = 'efficientnet_b0'
TEMP_MODEL = 'bambam_temp.pkl'

class TSKLearn(ClassifierMixin, BaseEstimator):
	loaded: bool = False

	def __init__(self, model: str=TEMP_MODEL):
		try: 
			vars(self).update(vars(self.load(model)))
			self.loaded = True
			return
		except Exception as e:
			model = DEFAULT_MODEL  # fallback
			print(f"Não foi possível carregar o modelo pré-treinado: {e}")
		self.set_device()
		self.model = timm.create_model(
			model, 
			pretrained=True, 
			num_classes=2,
		).to(self.device)
		self.classes_ = [0, 1]

		# Transformação interna (exige que X seja uma lista de imagens ou caminhos)
		self.transform = transforms.Compose([
			transforms.Lambda(self._to_img),
			transforms.Resize((224, 224)),
			transforms.ToTensor(),
			# transforms.Normalize(
			#     mean=[0.485, 0.456, 0.406], 
			#     std=[0.229, 0.224, 0.225]
			# ),
		])

	def fit(self, X, y, epochs=2, batch_size=16, *, force=False):
		"""
		X: Lista de imagens PIL ou Tensores [N, 3, 224, 224]
		y: Array/Lista de labels [0, 1, 0...]
		"""
		if not force and self.loaded:
			return self
		self.model.train() # Ativa o modo de treino (ativa Dropout, etc)
		
		# 1. Prepara os dados (Transforma X e y em tensores do PyTorch)
		if len(X) == 0: return self  # Evita erros se X estiver vazio
		X_tensors = torch.stack(tuple(
			self.transform(img) for img in X
		)).to(self.device) # type: ignore
		y_tensors = torch.tensor(y).long().to(self.device)
		# Ensure sklearn sees the fitted classes_ attribute (non-empty)
		try:
			self.classes_ = np.unique(np.array(y))
		except Exception:
			self.classes_ = np.array([0, 1])
		
		dataset = TensorDataset(X_tensors, y_tensors)
		loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

		# 2. Define como o modelo aprende
		criterion = nn.CrossEntropyLoss() # Função de erro
		optimizer = optim.Adam(self.model.parameters()) # O "ajustador" de pesos

		# 3. O Loop de Treino
		for epoch in range(epochs):
			total_loss = 0
			for batch_X, batch_y in loader:
				optimizer.zero_grad()           # Limpa o gradiente anterior
				outputs = self.model(batch_X)   # Passa a imagem pela rede
				loss = criterion(outputs, batch_y) # Calcula o erro
				loss.backward()                 # Calcula como ajustar os pesos
				optimizer.step()                # Faz o ajuste real
				total_loss += loss.item()
			
			print(f"Epoch {epoch+1} | Loss: {total_loss/len(loader):.4f}")
		return self
	
	def predict_proba(self, X):
		"""
		X: Lista de imagens PIL ou um Tensor [N, 3, 224, 224]
		"""
		self.model.eval()
		
		# 1. Transforma a lista de imagens em um único tensor gigante [Batch, 3, 224, 224]
		if len(X) == 0:
			return np.array([])  # Retorna um array vazio se não houver dados para prever
		# Se X já for um tensor, apenas mande para o device
		X_tensor = torch.stack(tuple(
			self.transform(img) for img in X
		)).to(self.device)

		# 2. Roda o modelo uma única vez para o bloco inteiro (Muito mais rápido!)
		with torch.no_grad():
			output = self.model(X_tensor)
			probabilities = torch.nn.functional.softmax(output, dim=1)
			probabilities = torch.flip(probabilities, dims=[1]) 
		return probabilities.cpu().numpy()
	
	def predict(self, X):
		return np.argmax(self.predict_proba(X), axis=1)
	
	def set_device(self) -> None:
		# 1. NVIDIA (Padrão ouro para Deep Learning)
		if torch.cuda.is_available():
			device_name = torch.cuda.get_device_name(0)
			print(f"GPU NVIDIA: {device_name}")
			self.device = torch.device("cuda")
		# 2. Apple Silicon (M1, M2, M3 - Seu cenário atual)
		elif torch.backends.mps.is_available():
			print("GPU Apple Silicon: MPS")
			self.device = torch.device("mps")
		# 3. AMD / Intel via DirectML (Comum em Windows/Laptops sem NVIDIA)
		# Requer: pip install torch-directml
		elif torch_directml and torch_directml.is_available():
			print("GPU AMD/Intel: DirectML")
			self.device = torch_directml.device()
		# 4. Intel XPU (Específico para placas Intel Arc / Data Centers)
		elif hasattr(torch, 'xpu') and torch.xpu.is_available():
			print("GPU Intel XPU")
			self.device = torch.device("xpu")
		# 5. CPU (O "Porto Seguro" - Funciona em qualquer lugar)
		else:
			print(f"Nenhuma GPU compatível")
			self.device = torch.device("cpu")
	
	
	@staticmethod
	def load(path) -> TSKLearn:
		model = joblib.load(path)
		try: model.set_device() # Garantia
		except Exception: pass
		return model

	def save(self, path) -> TSKLearn:
		device = self.device
		self.device = None
		joblib.dump(self, path)
		self.device = device
		return self

	@staticmethod
	def _to_img(img):
		if isinstance(img, str):
			img = image.open(img)
		if getattr(img, 'mode', None) != 'RGB':
			img = img.convert('RGB')
		return img
	
class Bambam:
	'''Wrapper de conveniência para usar o TSKLearn com calibração e
	avaliação simples.'''
	def __init__(self, model=TEMP_MODEL, /, **kwgs):
		self.classi = TSKLearn(model)
		params = dict(method='sigmoid', ensemble=False)
		params.update(kwgs)
		self.master = Calibrator(self.classi, **params) # type: ignore
	def fit(self, X, y, epochs=2, batch_size=16, *, force=False):
		self.classi.fit(X, y, epochs=epochs, batch_size=batch_size, force=force)
		self.master.fit(X, y)
		return self.classi.save(TEMP_MODEL)
	def predict_proba(self, X):
		return self.master.predict_proba(X)[:, 1] # Probabilidade da classe "real"
	def predict(self, X):
		return self.master.predict(X)

if __name__ == "__main__":
	# Exemplo de uso rápido (ajuste o caminho do modelo e dos dados conforme necessário)
	pmodel = joblib.load('Bambam_v1.joblib')
	test_fake = glob('dataset/test/fake/*')[:50]
	test_real = glob('dataset/test/real/*')[:50]
	test_all = test_fake + test_real
	test_y = [0] * len(test_fake) + [1] * len(test_real)
	p_final = pmodel.predict_proba(test_all)
