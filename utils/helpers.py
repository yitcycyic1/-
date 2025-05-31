import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def ensure_directory(directory):
    """Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Directory path to create
    """
    Path(directory).mkdir(parents=True, exist_ok=True)

def load_image_with_chinese_path(image_path):
    """Load an image with a path that may contain Chinese characters.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Loaded image in BGR format, or None if loading failed
    """
    try:
        # Read image with OpenCV (handles Chinese characters)
        img = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def save_image_with_chinese_path(image, output_path):
    """Save an image to a path that may contain Chinese characters.
    
    Args:
        image: Image to save
        output_path: Path where to save the image
    """
    try:
        # Ensure directory exists
        ensure_directory(os.path.dirname(output_path))
        # Save with encoding to handle Chinese characters
        cv2.imencode('.jpg', image)[1].tofile(output_path)
        return True
    except Exception as e:
        print(f"Error saving image to {output_path}: {e}")
        return False

def preprocess_image_for_model(image, target_size=(224, 224)):
    """Preprocess an image for the model.
    
    Args:
        image: Input image (BGR format)
        target_size: Target size for the image
        
    Returns:
        Preprocessed image ready for model input
    """
    # Resize if needed
    if image.shape[:2] != target_size:
        image = cv2.resize(image, target_size)
    
    # Convert to RGB (from BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    return image

def plot_confusion_matrix(cm, class_names):
    """Plot a confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')