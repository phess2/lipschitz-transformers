import numpy as np
import os
import pickle
import urllib.request
import tarfile

def load_cifar10(normalize=True):
   """
   Downloads (if needed) and loads the CIFAR-10 dataset.
   Returns: (train_images, train_labels, test_images, test_labels)
   """
   # Create data directory
   data_dir = os.path.join(os.path.dirname(__file__), "cifar10_files")
   if not os.path.exists(data_dir):
       os.makedirs(data_dir)

   # Download files if needed
   url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
   filepath = os.path.join(data_dir, "cifar-10-python.tar.gz")
   extracted_dir = os.path.join(data_dir, "cifar-10-batches-py")
   
   if not os.path.exists(extracted_dir):
       if not os.path.isfile(filepath):
           print(f"Downloading {url}")
           urllib.request.urlretrieve(url, filepath)
       with tarfile.open(filepath, 'r:gz') as tar:
           tar.extractall(data_dir)

   def load_batch(filename):
       with open(os.path.join(extracted_dir, filename), 'rb') as f:
           batch = pickle.load(f, encoding='bytes')
       return batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(batch[b'labels'])

   # Load training data
   train_images, train_labels = [], []
   for i in range(1, 6):
       images, labels = load_batch(f'data_batch_{i}')
       train_images.append(images)
       train_labels.append(labels)
   
   train_images = np.concatenate(train_images)
   train_labels = np.concatenate(train_labels)
   
   # Load test data
   test_images, test_labels = load_batch('test_batch')

   if normalize:
       train_images = train_images.astype(np.float32) / 255.0
       test_images = test_images.astype(np.float32) / 255.0

   return train_images, train_labels, test_images, test_labels