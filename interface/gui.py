import os
import sys
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from pathlib import Path

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_processing.face_detection import FaceDetector

class FaceRecognitionApp:
    """Interactive GUI application for face recognition."""
    
    def __init__(self, root, model_path, class_names):
        """Initialize the application.
        
        Args:
            root: Tkinter root window
            model_path: Path to the trained model
            class_names: List of class names (person names)
        """
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x700")
        
        # Load the model
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = class_names
        
        # Initialize face detector
        self.face_detector = FaceDetector()
        
        # Current image and face data
        self.current_image = None
        self.current_image_cv = None
        self.detected_faces = []
        self.selected_face_idx = -1
        
        # Create UI elements
        self._create_ui()
    
    def _create_ui(self):
        """Create the user interface elements."""
        # Main frame
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_panel = tk.Frame(main_frame)
        control_panel.pack(fill=tk.X, pady=10)
        
        # Upload button
        upload_btn = tk.Button(control_panel, text="Upload Image", command=self._upload_image, width=15, height=2)
        upload_btn.pack(side=tk.LEFT, padx=10)
        
        # Status label
        self.status_label = tk.Label(control_panel, text="Ready", anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Image display area (side by side)
        display_frame = tk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left panel - Original image
        left_panel = tk.LabelFrame(display_frame, text="Original Image")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.image_canvas = tk.Canvas(left_panel, bg="#f0f0f0")
        self.image_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.image_canvas.bind("<Button-1>", self._canvas_click)
        
        # Right panel - Recognition results
        right_panel = tk.LabelFrame(display_frame, text="Recognition Results")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        
        self.result_frame = tk.Frame(right_panel)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Results display
        self.result_canvas = tk.Canvas(self.result_frame, bg="#f0f0f0")
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bottom panel for recognition controls
        bottom_panel = tk.Frame(main_frame)
        bottom_panel.pack(fill=tk.X, pady=10)
        
        # Label for selected face
        tk.Label(bottom_panel, text="Selected Face:").pack(side=tk.LEFT, padx=5)
        self.selected_face_label = tk.Label(bottom_panel, text="None")
        self.selected_face_label.pack(side=tk.LEFT, padx=5)
        
        # Recognize button
        self.recognize_btn = tk.Button(bottom_panel, text="Recognize Face", command=self._recognize_face, state=tk.DISABLED)
        self.recognize_btn.pack(side=tk.LEFT, padx=10)
        
        # Mark as stranger button
        self.stranger_btn = tk.Button(bottom_panel, text="Mark as Stranger", command=self._mark_as_stranger, state=tk.DISABLED)
        self.stranger_btn.pack(side=tk.LEFT, padx=10)
    
    def _upload_image(self):
        """Handle image upload button click."""
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if not file_path:
            return
        
        try:
            # Read image with OpenCV (handles Chinese characters)
            self.current_image_cv = cv2.imdecode(
                np.fromfile(file_path, dtype=np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            if self.current_image_cv is None:
                messagebox.showerror("Error", "Failed to load the image")
                return
            
            # Convert to RGB for display
            rgb_image = cv2.cvtColor(self.current_image_cv, cv2.COLOR_BGR2RGB)
            
            # Resize if too large while maintaining aspect ratio
            h, w = rgb_image.shape[:2]
            max_size = 600
            if h > max_size or w > max_size:
                if h > w:
                    new_h, new_w = max_size, int(w * max_size / h)
                else:
                    new_h, new_w = int(h * max_size / w), max_size
                rgb_image = cv2.resize(rgb_image, (new_w, new_h))
                self.current_image_cv = cv2.resize(self.current_image_cv, (new_w, new_h))
            
            # Convert to PIL Image for Tkinter
            self.current_image = Image.fromarray(rgb_image)
            
            # Display the image
            self._display_image()
            
            # Detect faces
            self._detect_faces()
            
            self.status_label.config(text=f"Loaded image: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def _display_image(self):
        """Display the current image on the canvas."""
        if self.current_image is None:
            return
        
        # Clear canvas
        self.image_canvas.delete("all")
        
        # Create PhotoImage
        self.photo = ImageTk.PhotoImage(self.current_image)
        
        # Display on canvas
        self.image_canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.image_canvas.config(scrollregion=self.image_canvas.bbox(tk.ALL),
                               width=self.photo.width(),
                               height=self.photo.height())
    
    def _detect_faces(self):
        """Detect faces in the current image."""
        if self.current_image_cv is None:
            return
        
        # Reset face data
        self.detected_faces = []
        self.selected_face_idx = -1
        self.selected_face_label.config(text="None")
        self.recognize_btn.config(state=tk.DISABLED)
        self.stranger_btn.config(state=tk.DISABLED)
        
        # Detect faces
        faces = self.face_detector.detect_faces(self.current_image_cv)
        
        # Store detected faces
        self.detected_faces = faces
        
        # Draw rectangles around faces
        self._draw_face_rectangles()
        
        # Update status
        self.status_label.config(text=f"Detected {len(faces)} faces")
    
    def _draw_face_rectangles(self):
        """Draw rectangles around detected faces."""
        if self.current_image is None:
            return
        
        # Redisplay the image
        self._display_image()
        
        # Draw rectangles for each face
        for i, (x, y, w, h) in enumerate(self.detected_faces):
            color = "#00ff00" if i == self.selected_face_idx else "#ff0000"
            self.image_canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=2)
            self.image_canvas.create_text(x, y-10, text=f"Face {i+1}", anchor=tk.SW, fill=color)
    
    def _canvas_click(self, event):
        """Handle click on the image canvas."""
        if len(self.detected_faces) == 0:
            return
        
        # Check if click is inside any face rectangle
        for i, (x, y, w, h) in enumerate(self.detected_faces):
            if x <= event.x <= x+w and y <= event.y <= y+h:
                self.selected_face_idx = i
                self.selected_face_label.config(text=f"Face {i+1}")
                self.recognize_btn.config(state=tk.NORMAL)
                self.stranger_btn.config(state=tk.NORMAL)
                self._draw_face_rectangles()  # Redraw with selected face highlighted
                return
        
        # If click is not on any face
        self.selected_face_idx = -1
        self.selected_face_label.config(text="None")
        self.recognize_btn.config(state=tk.DISABLED)
        self.stranger_btn.config(state=tk.DISABLED)
        self._draw_face_rectangles()
    
    def _recognize_face(self):
        """Recognize the selected face."""
        if self.selected_face_idx < 0 or self.current_image_cv is None:
            return
        
        try:
            # Get the selected face
            x, y, w, h = self.detected_faces[self.selected_face_idx]
            
            # Crop the face with padding
            face_img = self.face_detector.crop_face(self.current_image_cv, (x, y, w, h))
            
            # Preprocess for the model
            face_img = cv2.resize(face_img, (224, 224))  # Resize to model input size
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face_img = face_img.astype(np.float32) / 255.0  # Normalize
            
            # Make prediction
            face_img = np.expand_dims(face_img, axis=0)  # Add batch dimension
            predictions = self.model.predict(face_img)
            
            # Get the predicted class and confidence
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class]
            
            # Display result
            if confidence > 0.7:  # Confidence threshold
                person_name = self.class_names[predicted_class]
                self._display_result(face_img[0], f"{person_name} ({confidence:.2f})")
                self.status_label.config(text=f"Recognized as {person_name} with {confidence:.2f} confidence")
            else:
                self._display_result(face_img[0], f"Unknown ({confidence:.2f})")
                self.status_label.config(text=f"Low confidence: {confidence:.2f}, might be a stranger")
                
        except Exception as e:
            messagebox.showerror("Error", f"Recognition failed: {str(e)}")
    
    def _mark_as_stranger(self):
        """Mark the selected face as a stranger."""
        if self.selected_face_idx < 0 or self.current_image_cv is None:
            return
        
        try:
            # Get the selected face
            x, y, w, h = self.detected_faces[self.selected_face_idx]
            
            # Crop the face with padding
            face_img = self.face_detector.crop_face(self.current_image_cv, (x, y, w, h))
            
            # Preprocess for display
            face_img = cv2.resize(face_img, (224, 224))  # Resize to standard size
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB
            face_img = face_img.astype(np.float32) / 255.0  # Normalize
            
            # Display result
            self._display_result(face_img, "Stranger")
            self.status_label.config(text="Marked as stranger")
                
        except Exception as e:
            messagebox.showerror("Error", f"Operation failed: {str(e)}")
    
    def _display_result(self, face_img, label_text):
        """Display the recognition result."""
        # Clear result canvas
        self.result_canvas.delete("all")
        
        # Convert the face image to PIL format
        face_pil = Image.fromarray((face_img * 255).astype(np.uint8))
        
        # Create PhotoImage
        self.result_photo = ImageTk.PhotoImage(face_pil)
        
        # Calculate center position
        canvas_width = self.result_canvas.winfo_width()
        canvas_height = self.result_canvas.winfo_height()
        x = (canvas_width - self.result_photo.width()) // 2
        y = (canvas_height - self.result_photo.height()) // 2 - 20
        
        # Display face image
        self.result_canvas.create_image(x, y, anchor=tk.NW, image=self.result_photo)
        
        # Display label
        self.result_canvas.create_text(
            canvas_width // 2, 
            y + self.result_photo.height() + 20, 
            text=label_text,
            font=("Arial", 16, "bold"),
            fill="#0000ff"
        )