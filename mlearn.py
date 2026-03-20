import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import kagglehub as kag

# Configuration
IMAGE_SIZE = (64, 64)  # Resize images to this size
RANDOM_STATE = 42
TRAIN_DIR = './dataset/train'
TEST_DIR = './dataset/test'
DATASET = 'tristanzhang32/ai-generated-images-vs-real-images'

# kaggle_download('train/fake', 401, 500)
# ou 'train/real', 'test/fake', 'test/real'
def kaggle_download(folder:str, first:int, last:int):
	for i in range(first, last+1):
		fname = f'{folder}/{i:04d}.jpg'
		fpath = kag.dataset_download(DATASET, fname)
		os.rename(fpath, f'./dataset/{fname}')
	
def load_images_from_directory(dir_path, label, target_size):
	# Load all images from a directory and return flattened feature vectors.
	images = []
	labels = []
	
	for img_file in os.listdir(dir_path):
		if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
			try:
				img_path = os.path.join(dir_path, img_file)
				# Open and resize image
				img = Image.open(img_path).convert('L')
				img = img.resize(target_size)
				
				# Convert to numpy array and flatten
				img_array = np.array(img).flatten()
				
				images.append(img_array)
				labels.append(label)
			except Exception as e:
				print(f'Error loading {img_file}: {e}')
	
	return images, labels

def load_dataset(data_dir, target_size):
	# Load fake and real images from the directory structure.
	fake_dir = os.path.join(data_dir, 'fake')
	real_dir = os.path.join(data_dir, 'real')
	
	# Load images: 0=fake, 1=real
	fake_images, fake_labels = load_images_from_directory(fake_dir, label=0, target_size=target_size)
	real_images, real_labels = load_images_from_directory(real_dir, label=1, target_size=target_size)
	
	# Combine
	images = np.append(fake_images, real_images, axis=0)
	labels =  np.append(fake_labels, real_labels, axis=0)
	
	print(f'Loaded {len(images)} images')
	print(f'Feature vector size: {images.shape[1]} (from {target_size[0]}x{target_size[1]} images with 3 channels)')
	print(f'Class distribution - Fake: {sum(labels==0)}, Real: {sum(labels==1)}')
	
	return images, labels


def train_random_forest(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None):
	# Train RandomForest classifier on train set and evaluate on test set.
	# Create and train model
	print(f'\nTraining RandomForest with {n_estimators} trees...')
	model = RandomForestClassifier(
		n_estimators=n_estimators,
		max_depth=max_depth,
		random_state=RANDOM_STATE,
		n_jobs=-1,  # Use all CPU cores
		verbose=1
	)
	model.fit(X_train, y_train)
	
	# Evaluate on test set
	print(f'\nEvaluating on test set...')
	y_pred = model.predict(X_test)
	
	accuracy = accuracy_score(y_test, y_pred)
	precision = precision_score(y_test, y_pred)
	recall = recall_score(y_test, y_pred)
	f1 = f1_score(y_test, y_pred)
	
	print(f'\nModel Performance:')
	print(f'Accuracy:  {accuracy:.4f}')
	print(f'Precision: {precision:.4f}')
	print(f'Recall:    {recall:.4f}')
	print(f'F1-Score:  {f1:.4f}')
	print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
	
	return model, y_pred

def feature_importance(model, top_n=10):
	# Show top important features.
	importances = model.feature_importances_
	indices = np.argsort(importances)[::-1][:top_n]
	
	print(f'\nTop {top_n} Most Important Features:')
	for i, idx in enumerate(indices):
		print(f'{i+1}. Feature {idx}: {importances[idx]:.4f}')

def main():
	# Load training dataset
	print('Loading training data...')
	X_train, y_train = load_dataset(TRAIN_DIR, target_size=IMAGE_SIZE)
	
	# Load test dataset
	print('\nLoading test data...')
	X_test, y_test = load_dataset(TEST_DIR, target_size=IMAGE_SIZE)
	
	if len(X_train) == 0 or len(X_test) == 0:
		print('Error: No images found. Check your directory structure.')
		return
	
	# Train model
	model, y_pred = train_random_forest(
		X_train, y_train, 
		X_test, y_test, 
		n_estimators=100, max_depth=None
	)
	
	# Show feature importance
	feature_importance(model, top_n=10)
	
	return model, X_test, y_test, y_pred

if __name__ == '__main__':
	model, X_test, y_test, y_pred = main()






