import cv2
import os
import numpy as np
from pathlib import Path

class FaceDetector:
    """Class for detecting and cropping faces from images."""
    
    def __init__(self):
        # Load pre-trained face detection model from OpenCV
        # Using Haar Cascade for simplicity, but you can replace with more advanced models
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Alternative: Use a more advanced face detector
        # Uncomment to use DNN-based detector (more accurate but slower)
        self.face_net = cv2.dnn.readNetFromCaffe(
            'models/deploy.prototxt',
            'models/res10_300x300_ssd_iter_140000.caffemodel'
        )
    
    def detect_faces(self, image):
        """Detect faces in an image and return their bounding boxes.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def detect_faces_dnn(self, image):
        """Alternative face detection using DNN (more accurate).
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            List of (x, y, w, h) tuples for detected faces
        """
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                    (300, 300), (104.0, 177.0, 123.0))
        
        self.face_net.setInput(blob)
        detections = self.face_net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:  # Filter weak detections
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2-x1, y2-y1))
                
        return faces
    
    def crop_face(self, image, face_box, padding=0.2):
        """Crop a detected face from the image with padding.
        
        Args:
            image: Input image
            face_box: (x, y, w, h) tuple of face bounding box
            padding: Percentage of padding to add around the face
            
        Returns:
            Cropped face image
        """
        x, y, w, h = face_box
        
        # Add padding
        padding_x = int(w * padding)
        padding_y = int(h * padding)
        
        # Calculate coordinates with padding
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(image.shape[1], x + w + padding_x)
        y2 = min(image.shape[0], y + h + padding_y)
        
        # Crop the face
        face_image = image[y1:y2, x1:x2]
        
        return face_image
    
    def process_image(self, image_path, output_path=None, size=(224, 224)):
        """Process an image to detect and crop the largest face.
        
        Args:
            image_path: Path to the input image
            output_path: Path to save the cropped face (optional)
            size: Target size for the cropped face
            
        Returns:
            Cropped and resized face image, or None if no face detected
        """
        # Read image with OpenCV (handles Chinese characters in path)
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        # Detect faces using DNN instead of Haar Cascade
        faces = self.detect_faces_dnn(image)
        
        if len(faces) == 0:
            print(f"No face detected in {image_path}")
            return None
        
        # Get the largest face (assuming it's the main subject)
        largest_face = max(faces, key=lambda box: box[2] * box[3])
        
        # Crop the face
        face_image = self.crop_face(image, largest_face)
        
        # Resize to target size
        face_image = cv2.resize(face_image, size)
        
        # Save if output path is provided
        if output_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            # Save with encoding to handle Chinese characters
            cv2.imencode('.jpg', face_image)[1].tofile(output_path)
        
        return face_image
    
    def process_dataset(self, input_dir, output_dir, size=(224, 224)):
        """Process all images in a dataset directory structure.
        
        Args:
            input_dir: Root directory of the dataset
            output_dir: Output directory for processed faces
            size: Target size for the cropped faces
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Process each person's directory
        for person_dir in input_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            print(f"Processing images for {person_name}")
            
            # Create output directory for this person
            person_output_dir = output_path / person_name
            person_output_dir.mkdir(exist_ok=True)
            
            # Process each image
            for i, img_path in enumerate(sorted(person_dir.glob('*.*'))):
                if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                    continue
                    
                output_file = person_output_dir / f"{i+1:03d}.jpg"
                
                try:
                    self.process_image(str(img_path), str(output_file), size)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")