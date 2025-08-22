#!/usr/bin/env python3
"""
DART Model Evaluation Script
Evaluates the deepfake detection model's accuracy and performance metrics.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import cv2
import json
from pathlib import Path

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import from our app
from app import load_model_with_compatibility, build_feature_extractor, load_video, prepare_single_video

class ModelEvaluator:
    def __init__(self):
        """Initialize the model evaluator."""
        print("🔍 Initializing DART Model Evaluator...")
        
        # Load the model
        try:
            self.model = load_model_with_compatibility()
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            self.model = None
            return
            
        # Load feature extractor
        try:
            self.feature_extractor = build_feature_extractor()
            print("✅ Feature extractor loaded successfully")
        except Exception as e:
            print(f"❌ Error loading feature extractor: {e}")
            self.feature_extractor = None
            return
            
        # Model configuration
        self.IMG_SIZE = 224
        self.MAX_SEQ_LENGTH = 20
        self.NUM_FEATURES = 2048
        
    def load_test_data(self, data_path="test_data"):
        """Load test data with labels."""
        print(f"📂 Loading test data from {data_path}...")
        
        # Check if we have a metadata file
        metadata_path = os.path.join(data_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            videos = []
            labels = []
            
            for filename, info in metadata.items():
                video_path = os.path.join(data_path, filename)
                if os.path.exists(video_path):
                    videos.append(video_path)
                    labels.append(1 if info['label'] == 'FAKE' else 0)  # 1 for FAKE, 0 for REAL
                    
            return videos, labels
        else:
            print("⚠️ No metadata.json found. Creating synthetic test data...")
            return self.create_synthetic_test_data()
    
    def create_synthetic_test_data(self):
        """Create synthetic test data for demonstration."""
        print("🎭 Creating synthetic test data...")
        
        # Generate random video-like data
        num_samples = 20
        videos = []
        labels = []
        
        for i in range(num_samples):
            # Create random frames (simulating video data)
            frames = np.random.randint(0, 255, (10, self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
            videos.append(frames)
            labels.append(np.random.randint(0, 2))  # Random REAL/FAKE labels
            
        return videos, labels
    
    def predict_video(self, video_data):
        """Predict if a video is fake or real."""
        try:
            if isinstance(video_data, str):  # If it's a file path
                frames = load_video(video_data, max_frames=self.MAX_SEQ_LENGTH)
            else:  # If it's already frame data
                frames = video_data
                
            if len(frames) == 0:
                return 0.5  # Neutral prediction if no frames
                
            # Prepare video for prediction
            frame_features, frame_mask = prepare_single_video(frames)
            
            # Make prediction
            prediction = self.model.predict([frame_features, frame_mask])[0]
            return float(prediction)
            
        except Exception as e:
            print(f"⚠️ Error predicting video: {e}")
            return 0.5  # Return neutral prediction on error
    
    def evaluate_model(self, test_videos, test_labels):
        """Evaluate the model on test data."""
        print("🧪 Evaluating model performance...")
        
        predictions = []
        true_labels = test_labels
        
        for i, video in enumerate(test_videos):
            print(f"Processing video {i+1}/{len(test_videos)}...", end='\r')
            
            # Get prediction
            pred_prob = self.predict_video(video)
            pred_label = 1 if pred_prob >= 0.5 else 0
            predictions.append(pred_label)
        
        print("\n")
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': predictions,
            'true_labels': true_labels
        }
    
    def print_evaluation_results(self, results):
        """Print detailed evaluation results."""
        print("\n" + "="*60)
        print("🎯 DART MODEL EVALUATION RESULTS")
        print("="*60)
        
        print(f"📊 Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
        print(f"🎯 Precision: {results['precision']:.4f}")
        print(f"🔍 Recall: {results['recall']:.4f}")
        print(f"⚖️ F1-Score: {results['f1_score']:.4f}")
        
        print("\n📈 Confusion Matrix:")
        print("     Predicted")
        print("       0    1")
        print(f"True 0 [{results['confusion_matrix'][0,0]:3d}  {results['confusion_matrix'][0,1]:3d}]")
        print(f"     1 [{results['confusion_matrix'][1,0]:3d}  {results['confusion_matrix'][1,1]:3d}]")
        print("(0=REAL, 1=FAKE)")
        
        # Calculate additional metrics
        tn, fp, fn, tp = results['confusion_matrix'].ravel() if results['confusion_matrix'].size == 4 else (0, 0, 0, 0)
        
        if tp + fn > 0:
            sensitivity = tp / (tp + fn)  # True Positive Rate
            print(f"🔬 Sensitivity (TPR): {sensitivity:.4f}")
        
        if tn + fp > 0:
            specificity = tn / (tn + fp)  # True Negative Rate
            print(f"🛡️ Specificity (TNR): {specificity:.4f}")
        
        print("\n" + "="*60)
    
    def benchmark_model(self):
        """Run a comprehensive benchmark of the model."""
        print("🚀 Starting DART Model Benchmark...")
        
        if self.model is None:
            print("❌ Cannot benchmark: Model not loaded")
            return None
        
        # Load or create test data
        try:
            test_videos, test_labels = self.load_test_data()
            print(f"📊 Loaded {len(test_videos)} test samples")
        except Exception as e:
            print(f"⚠️ Error loading test data: {e}")
            test_videos, test_labels = self.create_synthetic_test_data()
            print(f"🎭 Created {len(test_videos)} synthetic test samples")
        
        # Evaluate model
        results = self.evaluate_model(test_videos, test_labels)
        
        # Print results
        self.print_evaluation_results(results)
        
        # Additional model information
        print("\n🔧 Model Information:")
        print(f"   • Architecture: InceptionV3 + Fallback Model")
        print(f"   • Input Size: {self.IMG_SIZE}x{self.IMG_SIZE}")
        print(f"   • Max Sequence Length: {self.MAX_SEQ_LENGTH} frames")
        print(f"   • Feature Dimensions: {self.NUM_FEATURES}")
        print(f"   • Model Type: {'Original' if hasattr(self.model, 'layers') else 'Fallback Demonstration'}")
        
        return results

def main():
    """Main function to run model evaluation."""
    print("🎭 DART - Deepfake Analysis and Realism Test")
    print("Model Accuracy Evaluation Tool")
    print("-" * 50)
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Run benchmark
    results = evaluator.benchmark_model()
    
    if results:
        print("\n✅ Evaluation completed successfully!")
        
        # Save results to file
        results_file = "evaluation_results.json"
        try:
            # Convert numpy arrays to lists for JSON serialization
            json_results = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'model_type': 'fallback_demonstration'
            }
            
            with open(results_file, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            print(f"💾 Results saved to {results_file}")
            
        except Exception as e:
            print(f"⚠️ Could not save results: {e}")
    else:
        print("❌ Evaluation failed!")

if __name__ == "__main__":
    main()