import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from tqdm import tqdm
from transformers import CLIPTokenizer

# Import our custom models
from models import (
    TextileDataset, TextileSwinCLIPClassifier, ContrastiveLoss, 
    create_data_collate_fn
)


class TextileTrainer:

    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.tokenizer = None
        
        # Training state
        self.best_accuracy = 0.0
        self.train_history = {'train_loss': [], 'val_loss': [], 'val_accuracy': []}
        
        print(f"Trainer initialized - Using device: {self.device}")
    
    def setup_data_loaders(self):
        """Setup training, validation, and test data loaders"""
        print("Setting up data loaders...")
        
        # Define paths
        data_dir = self.config['data_directory'] 
        train_csv = os.path.join(data_dir, "data", "train.csv")
        val_csv = os.path.join(data_dir, "data", "val.csv")
        test_csv = os.path.join(data_dir, "data", "test.csv")
        image_dir = os.path.join(data_dir, "new_augmented")
        
        # Check if files exist
        for path in [train_csv, val_csv, test_csv, image_dir]:
            if not os.path.exists(path):
                print(f"Warning: {path} does not exist")
        
        # Load tokenizer
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config['clip_model'])
        
        # Create datasets
        train_dataset = TextileDataset(
            csv_file=train_csv,
            image_dir=image_dir,
            tokenizer=self.tokenizer,
            training_mode=True,
            use_texture_features=self.config['use_texture']
        )
        
        val_dataset = TextileDataset(
            csv_file=val_csv,
            image_dir=image_dir,
            tokenizer=self.tokenizer,
            training_mode=False,
            use_texture_features=self.config['use_texture']
        )
        
        test_dataset = TextileDataset(
            csv_file=test_csv,
            image_dir=image_dir,
            tokenizer=self.tokenizer,
            training_mode=False,
            use_texture_features=self.config['use_texture']
        )
        
        # Ensure consistent class mappings across all datasets
        all_classes = self._merge_class_labels([train_dataset, val_dataset, test_dataset])
        
        # Create data loaders
        collate_fn = create_data_collate_fn()
        
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=True, 
            num_workers=0, 
            collate_fn=collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=False, 
            num_workers=0, 
            collate_fn=collate_fn
        )
        
        self.test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'],
            shuffle=False, 
            num_workers=0, 
            collate_fn=collate_fn
        )
        
        # Store class information
        self.num_classes = len(all_classes)
        self.class_names = all_classes
        self.texture_channels = train_dataset.texture_channels
        
        print(f"Data loaders created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples") 
        print(f"  Test: {len(test_dataset)} samples")
        print(f"  Classes: {self.num_classes}")
    
    def _merge_class_labels(self, datasets):
        """Ensure all datasets have the same class label mappings"""
        all_classes = set()
        for dataset in datasets:
            all_classes.update(dataset.process_classes)
        
        # Sort for consistent ordering
        all_classes = sorted(list(all_classes))
        
        # Update all datasets with the merged class list
        for dataset in datasets:
            dataset.process_classes = all_classes
            dataset.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
            dataset.idx_to_class = {idx: cls for cls, idx in dataset.class_to_idx.items()}
            
            # Recompute class indices
            dataset.data['class_idx'] = dataset.data['GT'].apply(
                lambda x: dataset.class_to_idx.get(str(x).strip(), -1) if pd.notna(x) else -1
            )
            dataset.data = dataset.data[dataset.data['class_idx'] >= 0].reset_index(drop=True)
        
        print(f"Merged class labels - Found {len(all_classes)} unique classes")
        return all_classes
    
    def setup_model(self):
        """Initialize model, optimizer, scheduler, and loss function"""
        print("Setting up model...")
        
        # Initialize model
        self.model = TextileSwinCLIPClassifier(
            num_classes=self.num_classes,
            swin_model_name=self.config['swin_model'],
            clip_model_name=self.config['clip_model'],
            hidden_dim=self.config['hidden_dim'],
            dropout_rate=self.config['dropout'],
            texture_channels=self.texture_channels
        )
        
        self.model.to(self.device)
        
        # Setup optimizer (only for trainable parameters)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config['num_epochs']
        )
        
        # Setup loss function
        self.criterion = ContrastiveLoss(temperature=self.config['temperature'])
        
        print("Model setup complete!")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Setup mixed precision if available
        use_amp = self.config.get('use_mixed_precision', False) and self.device.type == 'cuda'
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            try:
                # Move data to device
                images = batch['image'].to(self.device)
                class_indices = batch['class_idx'].to(self.device)
                ing_ids = batch['ingredients_input_ids'].to(self.device)
                ing_mask = batch['ingredients_attention_mask'].to(self.device)
                proc_ids = batch['process_input_ids'].to(self.device)
                proc_mask = batch['process_attention_mask'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass with mixed precision if enabled
                if use_amp:
                    with torch.cuda.amp.autocast():
                        fused_features, process_features = self.model(
                            images, ing_ids, ing_mask, proc_ids, proc_mask
                        )
                        loss = self.criterion(fused_features, process_features, class_indices)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    fused_features, process_features = self.model(
                        images, ing_ids, ing_mask, proc_ids, proc_mask
                    )
                    loss = self.criterion(fused_features, process_features, class_indices)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        num_batches = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        with torch.no_grad():
            for batch in progress_bar:
                try:
                    # Move data to device
                    images = batch['image'].to(self.device)
                    class_indices = batch['class_idx'].to(self.device)
                    ing_ids = batch['ingredients_input_ids'].to(self.device)
                    ing_mask = batch['ingredients_attention_mask'].to(self.device)
                    proc_ids = batch['process_input_ids'].to(self.device)
                    proc_mask = batch['process_attention_mask'].to(self.device)
                    
                    # Forward pass
                    fused_features, process_features = self.model(
                        images, ing_ids, ing_mask, proc_ids, proc_mask
                    )
                    
                    loss = self.criterion(fused_features, process_features, class_indices)
                    total_loss += loss.item()
                    
                    # Compute predictions using similarity matching
                    fused_norm = F.normalize(fused_features, p=2, dim=1)
                    process_norm = F.normalize(process_features, p=2, dim=1)
                    
                    similarity = torch.matmul(fused_norm, process_norm.T)
                    pred_indices = similarity.argmax(dim=1)
                    predictions = class_indices[pred_indices]
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(class_indices.cpu().numpy())
                    num_batches += 1
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = accuracy_score(all_labels, all_predictions) if all_labels else 0.0
        
        return avg_loss, accuracy
    
    def train_model(self):
        """Main training loop"""
        print(f"Starting training for {self.config['num_epochs']} epochs...")
        
        patience = self.config.get('early_stopping_patience', 10)
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # Training phase
            train_loss = self.train_epoch()
            
            # Validation phase
            val_loss, val_accuracy = self.validate_epoch()
            
            # Update learning rate
            if self.scheduler:
                self.scheduler.step()
            
            # Record metrics
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_accuracy'].append(val_accuracy)
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
            
            # Save best model
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self._save_checkpoint(epoch, is_best=True)
                patience_counter = 0
                print(f"New best model saved! Accuracy: {val_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        print(f"\nTraining completed! Best validation accuracy: {self.best_accuracy:.4f}")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_accuracy': self.best_accuracy,
            'class_names': self.class_names,
            'config': self.config,
            'train_history': self.train_history
        }
        
        if is_best:
            torch.save(checkpoint, 'best_textile_model.pt')
        
        # Also save final checkpoint
        torch.save(checkpoint, 'textile_model_final.pt')
    
    def generate_validation_report(self):
        """Generate comprehensive validation report with confusion matrix"""
        print("\nGenerating validation report...")
        
        try:
            # Load best model
            checkpoint = torch.load('best_textile_model.pt', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from epoch {checkpoint['epoch']}")
        except:
            print("Using current model for evaluation")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Generating predictions"):
                try:
                    # Move data to device
                    images = batch['image'].to(self.device)
                    class_indices = batch['class_idx'].to(self.device)
                    ing_ids = batch['ingredients_input_ids'].to(self.device)
                    ing_mask = batch['ingredients_attention_mask'].to(self.device)
                    proc_ids = batch['process_input_ids'].to(self.device)
                    proc_mask = batch['process_attention_mask'].to(self.device)
                    
                    # Forward pass
                    fused_features, process_features = self.model(
                        images, ing_ids, ing_mask, proc_ids, proc_mask
                    )
                    
                    # Compute predictions
                    fused_norm = F.normalize(fused_features, p=2, dim=1)
                    process_norm = F.normalize(process_features, p=2, dim=1)
                    
                    similarity = torch.matmul(fused_norm, process_norm.T)
                    pred_indices = similarity.argmax(dim=1)
                    predictions = class_indices[pred_indices]
                    
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(class_indices.cpu().numpy())
                    
                except Exception as e:
                    print(f"Error in evaluation: {e}")
                    continue
        
        if not all_predictions:
            print("No valid predictions generated")
            return
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        cm = confusion_matrix(all_labels, all_predictions)
        
        print(f"Final validation accuracy: {accuracy:.4f}")
        
        # Generate and save visualizations
        self._save_confusion_matrices(cm, accuracy)
        self._save_classification_report(all_labels, all_predictions)
        self._save_training_plots()
        
        print("Validation report generated successfully!")
    
    def _save_confusion_matrices(self, cm, accuracy):
        """Save confusion matrix visualizations"""
        # Normalized confusion matrix
        plt.figure(figsize=(12, 10))
        cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        sns.heatmap(cm_normalized, 
                   annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Normalized Count'})
        
        plt.title(f'Validation Confusion Matrix (Normalized)\\nAccuracy: {accuracy:.4f}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_normalized.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Raw confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.title(f'Validation Confusion Matrix (Raw Counts)\\nAccuracy: {accuracy:.4f}',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Labels', fontsize=12)
        plt.ylabel('True Labels', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix_raw.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_classification_report(self, labels, predictions):
        """Save detailed classification report"""
        report = classification_report(labels, predictions, target_names=self.class_names)
        
        with open('classification_report.txt', 'w', encoding='utf-8') as f:
            f.write("Textile Process Classification Report\\n")
            f.write("=" * 50 + "\\n\\n")
            f.write(report)
        
        # Print summary
        report_dict = classification_report(labels, predictions, target_names=self.class_names, output_dict=True)
        print("\\nClassification Summary:")
        print(f"Macro Avg Precision: {report_dict['macro avg']['precision']:.4f}")
        print(f"Macro Avg Recall: {report_dict['macro avg']['recall']:.4f}")
        print(f"Macro Avg F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    
    def _save_training_plots(self):
        """Save training history plots"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        axes[0].plot(self.train_history['train_loss'], label='Training Loss', color='blue')
        axes[0].plot(self.train_history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Training and Validation Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.train_history['val_accuracy'], label='Validation Accuracy', color='green')
        axes[1].set_title('Validation Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main training function"""
    # Configuration
    config = {
        # Data paths - Use relative path or environment variable
        'data_directory': os.environ.get('TEXTILE_DATA_DIR', './data'),
        
        # Model configuration
        'swin_model': "microsoft/swin-base-patch4-window7-224",
        'clip_model': "openai/clip-vit-base-patch32", 
        'hidden_dim': 768,
        'dropout': 0.3,
        'use_texture': True,
        
        # Training configuration
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'temperature': 0.03,
        'early_stopping_patience': 10,
        'use_mixed_precision': True,
    }
    
    print("Starting Textile Process Classification Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    try:
        # Initialize trainer
        trainer = TextileTrainer(config)
        
        # Setup data and model
        trainer.setup_data_loaders()
        trainer.setup_model()
        
        # Train the model
        trainer.train_model()
        
        # Generate evaluation report
        trainer.generate_validation_report()
        
        print("\\nTraining completed successfully!")
        print("Generated files:")
        print("  - best_textile_model.pt (best model)")
        print("  - textile_model_final.pt (final model)")
        print("  - confusion_matrix_normalized.png")
        print("  - confusion_matrix_raw.png")
        print("  - classification_report.txt")
        print("  - training_history.png")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
