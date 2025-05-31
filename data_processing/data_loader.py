import os
import numpy as np
import cv2
from pathlib import Path
import random

class DataLoader:
    """Class for loading and preparing facial recognition dataset."""
    
    def __init__(self, data_dir, img_size=(224, 224), train_ratio=0.8):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the processed face images
            img_size: Size of images to be used for training
            train_ratio: Ratio of images to use for training vs testing
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.class_names = []
        
        # Scan the directory to get class names (person names)
        self._scan_classes()
    
    def _scan_classes(self):
        """Scan the data directory to find all person classes."""
        self.class_names = [d.name for d in self.data_dir.iterdir() if d.is_dir()]
        self.class_names.sort()  # Sort for consistent ordering
        
        print(f"Found {len(self.class_names)} classes (persons)")
    
    def load_data(self):
        """Load the dataset for training.
        
        Returns:
            (X_train, y_train, X_test, y_test) tuple of numpy arrays
            Note: X_test and y_test will be empty arrays as all data is used for training
        """
        X_train, y_train = [], []
        X_test, y_test = [], []  # Empty arrays for compatibility
        
        # Process each person's directory
        for class_idx, person_name in enumerate(self.class_names):
            person_dir = self.data_dir / person_name
            
            # Get all image files
            image_files = sorted([f for f in person_dir.glob('*.*') 
                                if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']])
            
            # Load ALL images for training
            for img_path in image_files:
                try:
                    # Read image with OpenCV (handles Chinese characters)
                    img = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        print(f"Failed to load {img_path}")
                        continue
                        
                    # Convert to RGB (from BGR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Resize if needed
                    if img.shape[:2] != self.img_size:
                        img = cv2.resize(img, self.img_size)
                    
                    # Normalize pixel values to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    X_train.append(img)
                    y_train.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)  # Empty array
        y_test = np.array(y_test)  # Empty array
        
        print(f"Loaded {len(X_train)} training images and {len(X_test)} testing images")
        
        return X_train, y_train, X_test, y_test
    
    def get_class_names(self):
        """Get the list of class names (person names)."""
        return self.class_names
    
    def get_num_classes(self):
        """Get the number of classes (persons)."""
        return len(self.class_names)
    
    def get_batch_generator(self, X, y, batch_size=32, augment=False):
        """Create a batch generator for training.
        
        Args:
            X: Image data
            y: Labels
            batch_size: Size of batches to generate
            augment: Whether to apply data augmentation
            
        Returns:
            Generator yielding (batch_x, batch_y) tuples
        """
        num_samples = len(X)
        indices = np.arange(num_samples)
        
        while True:
            # Shuffle at the beginning of each epoch
            np.random.shuffle(indices)
            
            for start_idx in range(0, num_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                batch_x = X[batch_indices]
                batch_y = y[batch_indices]
                
                if augment:
                    # Apply data augmentation
                    batch_x = self._augment_batch(batch_x)
                
                yield batch_x, batch_y
    
    def _augment_batch(self, batch):
        """Apply data augmentation to a batch of images.
        
        Args:
            batch: Batch of images to augment
            
        Returns:
            Augmented batch
        """
        augmented_batch = np.copy(batch)
        
        for i in range(len(batch)):
            img = batch[i].copy()
            
            # Random horizontal flip
            if random.random() > 0.5:
                img = np.fliplr(img)
            
            # Random brightness/contrast adjustment
            if random.random() > 0.5:
                alpha = 0.8 + random.random() * 0.4  # 0.8-1.2
                beta = -10 + random.random() * 20  # -10 to 10
                img = np.clip(alpha * img + beta/255.0, 0, 1)
            
            # Random rotation (slight)
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                h, w = img.shape[:2]
                center = (w/2, h/2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
            
            augmented_batch[i] = img
        
        return augmented_batch