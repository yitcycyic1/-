import os
import sys
import argparse
import tkinter as tk
from pathlib import Path

# Import our modules
from data_processing.data_loader import DataLoader
from data_processing.face_detection import FaceDetector
from model.face_model import FaceRecognitionModel
from model.trainer import ModelTrainer
from interface.gui import FaceRecognitionApp
from utils.helpers import ensure_directory

def process_dataset(args):
    """Process the raw dataset to extract faces."""
    print("\n===== Processing Dataset =====")
    face_detector = FaceDetector()
    
    # Process the dataset
    face_detector.process_dataset(
        input_dir=args.data_dir,
        output_dir=args.processed_dir,
        size=(224, 224)
    )
    
    print(f"Processed faces saved to {args.processed_dir}")

def train_model(args):
    """Train the face recognition model."""
    print("\n===== Training Model =====")
    
    # Load the processed data
    data_loader = DataLoader(
        data_dir=args.processed_dir,
        img_size=(224, 224)
    )
    
    # Create the model
    model = FaceRecognitionModel(
        num_classes=data_loader.get_num_classes(),
        input_shape=(224, 224, 3),
        use_gpu=args.use_gpu
    )
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        data_loader=data_loader,
        output_dir=args.output_dir
    )
    
    # Train the model
    trainer.train(
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_augmentation=True
    )
    
    print(f"Model training completed. Model saved to {args.output_dir}")

def run_gui(args):
    """Run the GUI application."""
    print("\n===== Starting GUI Application =====")
    
    # Load class names
    data_loader = DataLoader(data_dir=args.processed_dir)
    class_names = data_loader.get_class_names()
    
    # Create Tkinter root
    root = tk.Tk()
    
    # Create app
    app = FaceRecognitionApp(
        root=root,
        model_path=os.path.join(args.output_dir, 'best_model.h5'),
        class_names=class_names
    )
    
    # Run the app
    root.mainloop()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Face Recognition System")
    
    # Add arguments
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing the raw dataset')
    parser.add_argument('--processed_dir', type=str, default='processed_faces',
                        help='Directory to save processed faces')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save outputs (models, plots)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs for training')
    parser.add_argument('--use_gpu', action='store_true',
                        help='Use GPU for training if available')
    parser.add_argument('--mode', type=str, choices=['process', 'train', 'gui', 'all'],
                        default='gui', help='Mode of operation')
    
    args = parser.parse_args()
    
    # Create output directories
    ensure_directory(args.processed_dir)
    ensure_directory(args.output_dir)
    
    # Check if data directory is provided for processing or training
    if args.mode in ['process', 'train', 'all'] and args.data_dir is None:
        print("Error: --data_dir must be provided for 'process', 'train', or 'all' modes")
        parser.print_help()
        return
    
    # Run the selected mode
    if args.mode == 'process' or args.mode == 'all':
        process_dataset(args)
        
    if args.mode == 'train' or args.mode == 'all':
        train_model(args)
        
    if args.mode == 'gui' or args.mode == 'all':
        run_gui(args)

if __name__ == "__main__":
    main()