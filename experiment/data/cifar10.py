import numpy as np
import os
import pickle
import urllib.request
import tarfile
import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, Iterator
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class ImageDataset:
    """JAX dataset for image data."""

    def __init__(self, images, labels):
        # Convert everything to JAX arrays from the beginning
        self.images = jnp.array(images)
        self.labels = jnp.array(labels)
        self._length = len(images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

    def __len__(self):
        return self._length

class DataLoader:
    """JAX dataloader for image data with parallel processing and prefetching."""
    
    def __init__(self, dataset: ImageDataset, batch_size: int, shuffle: bool = False, 
                 drop_last: bool = True, seed: int = 0, repeat: bool = True,
                 num_workers: int = 4, prefetch_size: int = 2):
        """Initialize the dataloader.
        
        Args:
            dataset: Dataset to load from
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the dataset
            drop_last: Whether to drop the last incomplete batch
            seed: Random seed for shuffling
            repeat: Whether to restart iteration after reaching the end
            num_workers: Number of parallel workers for data loading
            prefetch_size: Number of batches to prefetch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.key = jax.random.PRNGKey(seed)
        self.repeat = repeat
        self.num_workers = num_workers
        self.prefetch_size = prefetch_size
        
        # Pre-compile the batch loading function
        self._images = dataset.images
        self._labels = dataset.labels
        
        # Define a pure function that can be JIT-compiled
        self._get_batch = jax.jit(self._create_get_batch_fn())
        
    def _create_get_batch_fn(self):
        """Create a pure function for getting a batch that can be JIT-compiled."""
        def get_batch(images, labels, indices):
            # Use advanced indexing which works with JIT
            return images[indices], labels[indices]
        return get_batch
        
    def __iter__(self) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray]]:
        """Create an iterator over the dataset with parallel processing."""
        while True:  # This allows for infinite iteration if repeat=True
            indices = jnp.arange(len(self.dataset))
            
            if self.shuffle:
                self.key, subkey = jax.random.split(self.key)
                indices = jax.random.permutation(subkey, indices)
            
            # Calculate number of batches
            if self.drop_last:
                num_batches = len(self.dataset) // self.batch_size
            else:
                num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size
            
            # Create batches of indices
            batch_indices = [
                indices[i * self.batch_size:min((i + 1) * self.batch_size, len(self.dataset))]
                for i in range(num_batches)
            ]
            
            # Helper function to load next batch that uses the pre-compiled function
            def load_next_batch(batch_idx):
                if batch_idx >= len(batch_indices):
                    return None
                # Use the pre-compiled function with concrete indices
                return self._get_batch(self._images, self._labels, batch_indices[batch_idx])
            
            # Create prefetch queue
            prefetch_queue = []
            
            # Initialize prefetch queue
            for i in range(min(self.prefetch_size, num_batches)):
                batch = load_next_batch(i)
                if batch is not None:
                    prefetch_queue.append(batch)
            
            # Main iteration loop
            current_batch = 0
            while current_batch < num_batches:
                # Yield current batch
                if prefetch_queue:
                    yield prefetch_queue.pop(0)
                
                # Load next batch into prefetch queue
                next_batch_idx = current_batch + self.prefetch_size
                batch = load_next_batch(next_batch_idx)
                if batch is not None:
                    prefetch_queue.append(batch)
                
                current_batch += 1
            
            # If not repeating, break after one full iteration
            if not self.repeat:
                break

def _download_and_extract_cifar10():
    """
    Downloads (if needed) and extracts the CIFAR-10 dataset.
    Returns the path to the extracted directory.
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
    
    return extracted_dir

def _load_cifar10_data(normalize=True, randomize_labels=False):
    """
    Loads the CIFAR-10 dataset with parallel processing.
    Returns: (train_images, train_labels, test_images, test_labels)
    """
    extracted_dir = _download_and_extract_cifar10()

    def load_batch(filename):
        with open(os.path.join(extracted_dir, filename), 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        return batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(batch[b'labels'])

    # Load training data in parallel using multiprocessing
    train_filenames = [f'data_batch_{i}' for i in range(1, 6)]
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(load_batch, train_filenames))
    
    train_images = np.concatenate([r[0] for r in results])
    train_labels = np.concatenate([r[1] for r in results])
    
    # Load test data
    test_images, test_labels = load_batch('test_batch')

    if normalize:
        # Use JAX's parallel processing for normalization
        train_images = train_images.astype(np.float32) / 255.0
        test_images = test_images.astype(np.float32) / 255.0
    
    if randomize_labels:
        rng = np.random.default_rng()
        train_labels = rng.permutation(train_labels)

    return train_images, train_labels, test_images, test_labels

def classification_loss(model, w, inputs, targets):
    logits = model(inputs, w)  # shape is [batch, num_classes]
    num_classes = logits.shape[-1]
    one_hot_targets = jax.nn.one_hot(targets, num_classes)
    return -jnp.sum(one_hot_targets * jax.nn.log_softmax(logits)) / inputs.shape[0]

def load_cifar10(batch_size: int = 128, shuffle: bool = True, normalize: bool = True, 
                 repeat: bool = True, randomize_labels: bool = False) -> Dict[str, Any]:
    """
    Load the CIFAR-10 dataset and create dataloaders.
    
    Args:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the training data
        normalize: Whether to normalize the images to [0, 1]
        repeat: Whether to restart iteration after reaching the end
        randomize_labels: Whether to randomize the labels

    Returns:
        Dictionary containing train_loader, test_loader, and metadata
    """
    # Load the raw data
    train_images, train_labels, test_images, test_labels = _load_cifar10_data(
        normalize=normalize,
        randomize_labels=randomize_labels
    )
    
    # Create datasets
    train_dataset = ImageDataset(train_images, train_labels)
    test_dataset = ImageDataset(test_images, test_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, repeat=repeat)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, repeat=repeat)
    
    # Define class names
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    return {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'num_classes': 10,
        'class_names': class_names,
        'input_shape': (32, 32, 3),
        'loss': classification_loss
    }

# Example usage
if __name__ == "__main__":
    # Load the data with batch size of 64
    data = load_cifar10(batch_size=64)
    train_size = len(data['train_loader'].dataset)
    test_size = len(data['test_loader'].dataset)
    print(f"Number of training samples: {train_size} ({train_size // 64} full batches)")
    print(f"Number of test samples: {test_size} ({test_size // 64} full batches)")
    
    # Get the first batch from the training loader
    for x_batch, y_batch in data['train_loader']:
        print("Input shape:", x_batch.shape)
        print("Target shape:", y_batch.shape)
        
        # Print the first few labels
        print("First few labels:", y_batch[:5])
        print("Corresponding class names:", [data['class_names'][int(label)] for label in y_batch[:5]])
        break