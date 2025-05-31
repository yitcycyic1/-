import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

class ModelTrainer:
    """Class for training and evaluating the face recognition model."""
    
    def __init__(self, model, data_loader, output_dir='output'):
        """Initialize the trainer.
        
        Args:
            model: FaceRecognitionModel instance
            data_loader: DataLoader instance
            output_dir: Directory to save outputs (models, plots)
        """
        self.model = model
        self.data_loader = data_loader
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def train(self, batch_size=32, epochs=50, use_augmentation=True, validation_split=0.1):
        """Train the model.
        
        Args:
            batch_size: Batch size for training
            epochs: Maximum number of epochs to train
            use_augmentation: Whether to use data augmentation
            validation_split: Fraction of training data to use for validation
            
        Returns:
            Training history
        """
        # Load the data
        X_train, y_train, _, _ = self.data_loader.load_data()  # Ignore empty test set
        
        # Split training data for validation
        val_size = int(len(X_train) * validation_split)
        indices = np.random.permutation(len(X_train))
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]
        
        X_val = X_train[val_indices]
        y_val = y_train[val_indices]
        X_train = X_train[train_indices]
        y_train = y_train[train_indices]
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train the model
        print("Training the base model...")
        if use_augmentation:
            # Use data generators with augmentation
            train_gen = self.data_loader.get_batch_generator(
                X_train, y_train, batch_size=batch_size, augment=True)
            val_gen = self.data_loader.get_batch_generator(
                X_val, y_val, batch_size=batch_size, augment=False)
            
            steps_per_epoch = len(X_train) // batch_size
            validation_steps = len(X_val) // batch_size
            
            history = self.model.train_with_generator(
                train_gen, steps_per_epoch, val_gen, validation_steps,
                epochs=epochs, callbacks=callbacks
            )
        else:
            # Train without generators
            history = self.model.train(
                X_train, y_train, X_val, y_val,
                batch_size=batch_size, epochs=epochs, callbacks=callbacks
            )
        
        # Fine-tuning
        print("\nFine-tuning the model...")
        self.model.fine_tune()
        
        if use_augmentation:
            history_ft = self.model.train_with_generator(
                train_gen, steps_per_epoch, val_gen, validation_steps,
                epochs=epochs//2, callbacks=callbacks
            )
        else:
            history_ft = self.model.train(
                X_train, y_train, X_val, y_val,
                batch_size=batch_size, epochs=epochs//2, callbacks=callbacks
            )
        
        # Combine histories
        combined_history = {}
        for k in history.history.keys():
            combined_history[k] = history.history[k] + history_ft.history[k]
        
        # Save the final model
        self.model.save_model(os.path.join(self.output_dir, 'final_model.h5'))
        
        # Plot training history
        self._plot_training_history(combined_history)
        
        # Skip evaluation on test set since we don't have one
        # Instead, report validation metrics
        print("\nFinal validation metrics:")
        val_loss = combined_history['val_loss'][-1]
        val_accuracy = combined_history['val_accuracy'][-1]
        print(f"Validation loss: {val_loss:.4f}")
        print(f"Validation accuracy: {val_accuracy:.4f}")
        
        # Save validation results
        with open(os.path.join(self.output_dir, 'validation_results.txt'), 'w') as f:
            f.write(f"Validation loss: {val_loss:.4f}\n")
            f.write(f"Validation accuracy: {val_accuracy:.4f}\n")
        
        return combined_history
    
    def evaluate(self, X_test, y_test):
        """Evaluate the model on test data.
        
        Args:
            X_test: Test images
            y_test: Test labels
            
        Returns:
            (loss, accuracy) tuple
        """
        print("\nEvaluating on test set...")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"Test loss: {loss:.4f}")
        print(f"Test accuracy: {accuracy:.4f}")
        
        # Save test results
        with open(os.path.join(self.output_dir, 'test_results.txt'), 'w') as f:
            f.write(f"Test loss: {loss:.4f}\n")
            f.write(f"Test accuracy: {accuracy:.4f}\n")
        
        return loss, accuracy
    
    def _plot_training_history(self, history):
        """Plot and save training history.
        
        Args:
            history: Training history dictionary
        """
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['accuracy'], label='Training')
        plt.plot(history['val_accuracy'], label='Validation')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['loss'], label='Training')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))