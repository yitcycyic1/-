import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, applications

class FaceRecognitionModel:
    """Deep learning model for face recognition."""
    
    def __init__(self, num_classes, input_shape=(224, 224, 3), use_gpu=False):
        """Initialize the face recognition model.
        
        Args:
            num_classes: Number of people to recognize
            input_shape: Input image shape (height, width, channels)
            use_gpu: Whether to use GPU for training
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # Configure GPU usage
        if use_gpu:
            # Use GPU if available
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
                print("Using GPU for training")
            else:
                print("No GPU found, falling back to CPU")
        else:
            # Force CPU usage
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            print("Using CPU for training (GPU disabled)")
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build and compile the face recognition model.
        
        Returns:
            Compiled Keras model
        """
        # Use a pre-trained model as the base
        # MobileNetV2 is a good balance between accuracy and speed
        base_model = applications.MobileNetV2(
            input_shape=self.input_shape,
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Build the model on top of the base model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def fine_tune(self, unfreeze_layers=50):
        """Fine-tune the model by unfreezing some layers of the base model.
        
        Args:
            unfreeze_layers: Number of layers to unfreeze from the end
        """
        # Get the base model (first layer in our sequential model)
        base_model = self.model.layers[0]
        
        # Unfreeze the last n layers
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),  # Lower learning rate
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=32, epochs=10, callbacks=None):
        """Train the model.
        
        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            batch_size: Batch size for training
            epochs: Number of epochs to train
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks
        )
        
        return history
    
    def train_with_generator(self, train_gen, steps_per_epoch, val_gen, validation_steps, epochs=10, callbacks=None):
        """Train the model using data generators.
        
        Args:
            train_gen: Training data generator
            steps_per_epoch: Number of batches per epoch
            val_gen: Validation data generator
            validation_steps: Number of validation batches
            epochs: Number of epochs to train
            callbacks: List of Keras callbacks
            
        Returns:
            Training history
        """
        history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks
        )
        
        return history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            (loss, accuracy) tuple
        """
        return self.model.evaluate(X_test, y_test)
    
    def predict(self, image):
        """Predict the class of a single image.
        
        Args:
            image: Input image (should be preprocessed)
            
        Returns:
            (predicted_class_index, confidence) tuple
        """
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Get predictions
        predictions = self.model.predict(image)
        
        # Get the predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        return predicted_class, confidence
    
    def save_model(self, filepath):
        """Save the model to disk.
        
        Args:
            filepath: Path to save the model
        """
        self.model.save(filepath)
    
    def load_model(self, filepath):
        """Load a model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)