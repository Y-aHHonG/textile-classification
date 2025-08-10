import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import CLIPTokenizer, CLIPTextModel, SwinModel, AutoFeatureExtractor
from torchvision import transforms
import cv2
from skimage import feature


class TextureAnalyzer:
    
    def __init__(self, radius=1, n_points=8):
        self.radius = radius
        self.n_points = n_points
    
    def extract_texture_features(self, image):
        """Extract normalized LBP texture features"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Extract LBP features using uniform pattern
        lbp = feature.local_binary_pattern(
            gray, self.n_points, self.radius, method='uniform'
        )
        
        # Normalize to [0, 1] range for stable training
        if lbp.max() > lbp.min():
            lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())
        else:
            lbp = np.zeros_like(lbp)
            
        return lbp.astype(np.float32)


class TextileDataset(Dataset):
    
    def __init__(self, csv_file, image_dir, tokenizer=None, max_text_length=77, 
                 training_mode=True, use_texture_features=True):
        
        # Load data and setup basic attributes
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.training_mode = training_mode
        self.use_texture = use_texture_features
        
        # Clean up the data
        self._prepare_data()
        
        # Setup texture analyzer if needed
        if self.use_texture:
            self.texture_analyzer = TextureAnalyzer()
            self.texture_channels = 1
        else:
            self.texture_channels = 0
        
        # Setup image transformations
        self._setup_image_transforms()
        
        print(f"Loaded {len(self.data)} samples from {csv_file}")
        print(f"Found {len(self.process_classes)} different textile processes")
        if self.training_mode:
            print("Using data augmentation for training")
    
    def _prepare_data(self):
        """Clean data and setup class labels"""
        # Clean column names
        self.data.columns = [col.strip() for col in self.data.columns]
        
        # Clean ingredients text
        if 'Ingredients' in self.data.columns:
            self.data['Ingredients'] = self.data['Ingredients'].fillna("")
            self.data['Ingredients'] = self.data['Ingredients'].astype(str).str.strip()
        
        # Process GT labels and create class mappings
        self._create_class_mappings()
    
    def _create_class_mappings(self):
        """Create mappings between process names and class indices"""
        if 'GT' not in self.data.columns:
            raise ValueError("CSV must contain a 'GT' column with process labels")
        
        # Extract all unique processes
        all_processes = []
        for gt_label in self.data['GT']:
            if pd.notna(gt_label):
                process = str(gt_label).strip()
                if process:
                    all_processes.append(process)
        
        # Create class mappings
        self.process_classes = sorted(list(set(all_processes)))
        self.class_to_idx = {process: idx for idx, process in enumerate(self.process_classes)}
        self.idx_to_class = {idx: process for process, idx in self.class_to_idx.items()}
        
        # Add class indices to dataframe
        self.data['class_idx'] = self.data['GT'].apply(
            lambda x: self.class_to_idx.get(str(x).strip(), -1) if pd.notna(x) else -1
        )
        
        # Filter out invalid labels
        valid_data = self.data[self.data['class_idx'] >= 0]
        if len(valid_data) < len(self.data):
            print(f"Removed {len(self.data) - len(valid_data)} samples with invalid labels")
            self.data = valid_data.reset_index(drop=True)
    
    def _setup_image_transforms(self):
        """Setup different transforms for training vs validation"""
        if self.training_mode:
            # More aggressive augmentation for training
            self.spatial_transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ])
            
            # Separate color augmentation to not interfere with LBP
            self.color_transform = transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
            )
        else:
            # Simple resize for validation/testing
            self.spatial_transform = transforms.Resize((224, 224))
            self.color_transform = None
        
        # Standard normalization for RGB channels
        self.normalize_rgb = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    
    def _process_image(self, image_path):
        """Load and process a single image"""
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print("Warning: Could not load image, using fallback")
            image = Image.new('RGB', (224, 224), color='black')
        
        # Apply spatial transforms
        image = self.spatial_transform(image)
        
        # Apply color transforms if in training mode
        if self.training_mode and self.color_transform:
            image = self.color_transform(image)
        
        # Convert to tensor and numpy for different processing paths
        image_tensor = transforms.ToTensor()(image)
        image_np = np.array(transforms.ToPILImage()(image_tensor))
        
        # Extract texture features if enabled
        if self.use_texture:
            texture_features = self.texture_analyzer.extract_texture_features(image_np)
            texture_tensor = torch.from_numpy(texture_features).unsqueeze(0)  # Add channel dim
            
            # Combine RGB and texture channels
            combined_tensor = torch.cat([image_tensor, texture_tensor], dim=0)  # (4, H, W)
            
            # Normalize only RGB channels
            combined_tensor[:3] = self.normalize_rgb(combined_tensor[:3])
        else:
            # Use only RGB channels
            combined_tensor = self.normalize_rgb(image_tensor)
        
        return combined_tensor
    
    def _tokenize_text(self, text):
        """Tokenize text input using the provided tokenizer"""
        if not self.tokenizer or not text:
            return {
                "input_ids": torch.zeros(self.max_text_length, dtype=torch.long),
                "attention_mask": torch.zeros(self.max_text_length, dtype=torch.long)
            }
        
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded.input_ids.squeeze(0),
            "attention_mask": encoded.attention_mask.squeeze(0)
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        sample = self.data.iloc[idx]
        
        # Load and process image
        image_path = os.path.join(self.image_dir, sample['image'])
        processed_image = self._process_image(image_path)
        
        # Get text data
        ingredients = sample.get('Ingredients', "")
        process_label = str(sample.get('GT', "")).strip()
        class_idx = sample['class_idx']
        
        # Tokenize text inputs
        ingredients_tokens = self._tokenize_text(ingredients)
        process_tokens = self._tokenize_text(process_label)
        
        return {
            'image': processed_image,
            'class_idx': torch.tensor(class_idx, dtype=torch.long),
            'ingredients_input_ids': ingredients_tokens['input_ids'],
            'ingredients_attention_mask': ingredients_tokens['attention_mask'],
            'process_input_ids': process_tokens['input_ids'],
            'process_attention_mask': process_tokens['attention_mask'],
            'image_path': image_path,
            'process_name': process_label
        }
    
    def get_class_names(self):
        """Return list of all process class names"""
        return self.process_classes.copy()
    
    def get_class_mapping(self):
        """Return class name to index mapping"""
        return self.class_to_idx.copy()


class ChannelAdapter(nn.Module):
    """Adapts multi-channel input to 3-channel RGB for pretrained models"""
    
    def __init__(self, input_channels, output_channels=3):
        super().__init__()
        self.adapter = nn.Conv2d(input_channels, output_channels, kernel_size=1, bias=False)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.adapter.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        return self.adapter(x)


class CrossModalAttention(nn.Module):
    
    def __init__(self, feature_dim, num_heads=8, dropout_rate=0.1):
        super().__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(feature_dim)
        self.norm2 = nn.LayerNorm(feature_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim * 4, feature_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, query_features, key_value_features):
        # Apply layer norm before attention
        query_norm = self.norm1(query_features)
        kv_norm = self.norm1(key_value_features)
        
        # Self-attention
        attn_output, _ = self.attention(query_norm, kv_norm, kv_norm)
        
        # Residual connection
        output = query_features + attn_output
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(self.norm2(output))
        output = output + ffn_output
        
        return output


class BidirectionalFusion(nn.Module):
    
    def __init__(self, feature_dim, num_heads=8, dropout_rate=0.1):
        super().__init__()
        
        self.visual_to_text = CrossModalAttention(feature_dim, num_heads, dropout_rate)
        self.text_to_visual = CrossModalAttention(feature_dim, num_heads, dropout_rate)
        
        # Final projection layer
        self.fusion_projection = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, visual_features, text_features):
        
        # Cross-modal attention in both directions
        v2t = self.visual_to_text(visual_features, text_features)
        t2v = self.text_to_visual(text_features, visual_features)
        
        # Concatenate and project
        fused = torch.cat([v2t, t2v], dim=-1)
        output = self.fusion_projection(fused)
        
        return output


class TextileSwinCLIPClassifier(nn.Module):
    
    def __init__(self, num_classes, 
                 swin_model_name="microsoft/swin-base-patch4-window7-224",
                 clip_model_name="openai/clip-vit-base-patch32",
                 hidden_dim=768, dropout_rate=0.3, texture_channels=1):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.texture_channels = texture_channels
        
        # Channel adapter for multi-channel input
        input_channels = 3 + texture_channels  # RGB + texture
        self.channel_adapter = ChannelAdapter(input_channels, 3)
        
        # Load pretrained models
        self.swin_model = SwinModel.from_pretrained(swin_model_name)
        self.text_model = CLIPTextModel.from_pretrained(clip_model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
        
        # Feature projection layers
        vision_dim = self.swin_model.config.hidden_size
        text_dim = self.text_model.config.hidden_size
        
        self.vision_projector = nn.Linear(vision_dim, hidden_dim)
        self.text_projector = nn.Linear(text_dim, hidden_dim)
        
        # Multimodal fusion
        self.fusion_module = BidirectionalFusion(hidden_dim, num_heads=8, dropout_rate=dropout_rate)
        
        # Learnable temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # Freeze pretrained encoders to prevent overfitting
        self._freeze_pretrained_models()
        
        # Initialize new layers
        self._initialize_new_layers()
        
        self._print_model_info()
    
    def _freeze_pretrained_models(self):
        """Freeze pretrained Swin and CLIP models"""
        for param in self.swin_model.parameters():
            param.requires_grad = False
            
        for param in self.text_model.parameters():
            param.requires_grad = False
    
    def _initialize_new_layers(self):
        """Initialize newly added layers with proper weights"""
        def init_weights(m):
            if isinstance(m, nn.Linear):
                if m.weight.requires_grad:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.weight.requires_grad:
                    nn.init.ones_(m.weight)
                if m.bias is not None and m.bias.requires_grad:
                    nn.init.zeros_(m.bias)
        
        # Apply to trainable modules
        self.channel_adapter.apply(init_weights)
        self.vision_projector.apply(init_weights)
        self.text_projector.apply(init_weights)
        self.fusion_module.apply(init_weights)
    
    def _print_model_info(self):
        """Print model parameter information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Model initialized with {total_params:,} total parameters")
        print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    def encode_image(self, images):
        """Encode images using Swin Transformer"""
        # Adapt multi-channel input to RGB
        rgb_images = self.channel_adapter(images)
        
        # Extract features using Swin
        outputs = self.swin_model(rgb_images)
        pooled_features = outputs.pooler_output
        
        # Project to common dimension
        return self.vision_projector(pooled_features)
    
    def encode_text(self, input_ids, attention_mask):
        """Encode text using CLIP text model"""
        outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_features = outputs.pooler_output
        
        # Project to common dimension
        return self.text_projector(pooled_features)
    
    def forward(self, images, ingredients_input_ids=None, ingredients_attention_mask=None,
                process_input_ids=None, process_attention_mask=None):
        """
        Forward pass for training or inference
        
        Returns fused features for ingredients, and process text features if provided
        """
        # Encode images
        visual_features = self.encode_image(images)
        
        # Encode ingredient descriptions
        ingredient_features = self.encode_text(ingredients_input_ids, ingredients_attention_mask)
        
        # Fuse visual and ingredient features
        visual_seq = visual_features.unsqueeze(1)  # (B, 1, D)
        ingredient_seq = ingredient_features.unsqueeze(1)  # (B, 1, D)
        
        fused_features = self.fusion_module(visual_seq, ingredient_seq)
        fused_features = fused_features.squeeze(1)  # (B, D)
        
        # Encode process labels if provided (for training)
        if process_input_ids is not None and process_attention_mask is not None:
            process_features = self.encode_text(process_input_ids, process_attention_mask)
            return fused_features, process_features
        
        return fused_features


class ContrastiveLoss(nn.Module):
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, image_features, text_features, labels=None):
        """
        Compute contrastive loss
        
        Args:
            image_features: Fused image+ingredient features (B, D)
            text_features: Process text features (B, D)
            labels: Class labels (B,)
        """
        batch_size = image_features.size(0)
        
        # Normalize features
        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_features, text_features.T) / self.temperature
        
        # Create labels for contrastive learning
        if labels is not None:
            # Use actual class labels to create positive pairs
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(similarity_matrix.device)
        else:
            # Assume diagonal elements are positive pairs
            mask = torch.eye(batch_size, device=similarity_matrix.device)
        
        # Compute loss
        exp_sim = torch.exp(similarity_matrix)
        log_prob = similarity_matrix - torch.log(exp_sim.sum(1, keepdim=True))
        
        # Mean of log-likelihood of positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = -mean_log_prob_pos.mean()
        
        return loss


def create_data_collate_fn():

    def collate_fn(batch):
        # Filter out non-tensor fields that can't be batched
        keys_to_exclude = ['image_path', 'process_name']
        
        filtered_batch = []
        for item in batch:
            filtered_item = {k: v for k, v in item.items() if k not in keys_to_exclude}
            filtered_batch.append(filtered_item)
        
        # Use default collate for the rest
        from torch.utils.data.dataloader import default_collate
        return default_collate(filtered_batch)
    
    return collate_fn
