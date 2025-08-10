import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import gradio as gr
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings("ignore")

# Import our custom models
from models import TextileSwinCLIPClassifier, TextureAnalyzer
from transformers import CLIPTokenizer
from torchvision import transforms


class TextilePredictor:
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model file
            device: Computing device ('cuda', 'cpu', or None for auto)
        """
        self.device = torch.device(device) if device else torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.model_path = model_path
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.class_names = []
        self.texture_analyzer = None
        
        # Load the model
        self._load_model()
        self._setup_image_processing()
        
        print(f"Textile predictor ready!")
        print(f"   Device: {self.device}")
        print(f"   Classes: {len(self.class_names)}")
        print(f"   Model: {model_path}")
    
    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model configuration
            config = checkpoint.get('config', {})
            self.class_names = checkpoint.get('class_names', [])
            
            if not self.class_names:
                raise ValueError("No class names found in model checkpoint")
            
            # Initialize tokenizer
            clip_model = config.get('clip_model', 'openai/clip-vit-base-patch32')
            self.tokenizer = CLIPTokenizer.from_pretrained(clip_model)
            
            # Initialize model architecture
            self.model = TextileSwinCLIPClassifier(
                num_classes=len(self.class_names),
                swin_model_name=config.get('swin_model', 'microsoft/swin-base-patch4-window7-224'),
                clip_model_name=clip_model,
                hidden_dim=config.get('hidden_dim', 768),
                dropout_rate=config.get('dropout', 0.3),
                texture_channels=config.get('n_tex_ch', 1)
            )
            
            # Load weights
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            # Setup texture analyzer if model uses texture features
            if config.get('use_texture', True):
                self.texture_analyzer = TextureAnalyzer()
            
            print(f"Model loaded successfully from epoch {checkpoint.get('epoch', 'unknown')}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _setup_image_processing(self):
        """Setup image preprocessing transforms"""
        # Standard preprocessing for inference
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        # RGB normalization (matches training)
        self.rgb_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Process a single image for inference
        
        Args:
            image: PIL Image object
            
        Returns:
            Processed image tensor ready for model input
        """
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.image_transform(image)
        
        # Extract texture features if available
        if self.texture_analyzer:
            # Convert tensor back to numpy for texture extraction
            image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # Get texture features
            texture_features = self.texture_analyzer.extract_texture_features(image_np)
            texture_tensor = torch.from_numpy(texture_features).unsqueeze(0)
            
            # Combine RGB and texture
            combined_tensor = torch.cat([image_tensor, texture_tensor], dim=0)
            
            # Normalize only RGB channels
            combined_tensor[:3] = self.rgb_normalize(combined_tensor[:3])
        else:
            # Use only RGB
            combined_tensor = self.rgb_normalize(image_tensor)
        
        return combined_tensor.unsqueeze(0)  # Add batch dimension
    
    def _process_text(self, text: str, max_length: int = 77) -> Dict[str, torch.Tensor]:

        if not text:
            # Return empty tokens if no text provided
            return {
                "input_ids": torch.zeros((1, max_length), dtype=torch.long),
                "attention_mask": torch.zeros((1, max_length), dtype=torch.long)
            }
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoded.input_ids,
            "attention_mask": encoded.attention_mask
        }
    
    def predict(self, image: Image.Image, ingredients: str = "", 
                top_k: int = 5) -> List[Tuple[str, float]]:
        try:
            with torch.no_grad():
                # Process inputs
                image_tensor = self._process_image(image).to(self.device)
                ingredient_tokens = self._process_text(ingredients)
                
                # Move text tokens to device
                ing_ids = ingredient_tokens["input_ids"].to(self.device)
                ing_mask = ingredient_tokens["attention_mask"].to(self.device)
                
                # Get image-ingredient fused features
                fused_features = self.model(image_tensor, ing_ids, ing_mask)
                
                # Compute similarities with all class text representations
                similarities = []
                for class_name in self.class_names:
                    # Get text representation for this class
                    class_tokens = self._process_text(class_name)
                    class_ids = class_tokens["input_ids"].to(self.device)
                    class_mask = class_tokens["attention_mask"].to(self.device)
                    
                    # Get fused features and class text features
                    _, class_text_features = self.model(
                        image_tensor, ing_ids, ing_mask, class_ids, class_mask
                    )
                    
                    # Compute cosine similarity
                    fused_norm = F.normalize(fused_features, p=2, dim=1)
                    class_norm = F.normalize(class_text_features, p=2, dim=1)
                    
                    similarity = torch.matmul(fused_norm, class_norm.T).item()
                    similarities.append(similarity)
                
                # Convert to probabilities using softmax
                similarities_tensor = torch.tensor(similarities)
                probabilities = F.softmax(similarities_tensor, dim=0)
                
                # Get top-k predictions
                top_k_probs, top_k_indices = torch.topk(probabilities, min(top_k, len(self.class_names)))
                
                results = []
                for i in range(len(top_k_indices)):
                    class_idx = top_k_indices[i].item()
                    prob = top_k_probs[i].item()
                    class_name = self.class_names[class_idx]
                    results.append((class_name, prob))
                
                return results
                
        except Exception as e:
            print(f"Prediction error: {e}")
            return [("Error occurred", 0.0)]
    
    def batch_predict(self, images: List[Image.Image], 
                     ingredients_list: List[str]) -> List[List[Tuple[str, float]]]:

        if len(images) != len(ingredients_list):
            raise ValueError("Number of images and ingredients must match")
        
        results = []
        for image, ingredients in zip(images, ingredients_list):
            predictions = self.predict(image, ingredients)
            results.append(predictions)
        
        return results
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported textile process classes"""
        return self.class_names.copy()


def create_gradio_interface(model_path: str = "best_textile_model.pt"):
    
    # Initialize predictor
    try:
        predictor = TextilePredictor(model_path)
        print("Predictor initialized successfully")
    except Exception as e:
        print(f"Failed to initialize predictor: {e}")
        return None
    
    def predict_fabric(image, ingredients_text):
        """Gradio prediction function"""
        if image is None:
            return "Please upload a fabric image", "0%", []
        
        try:
            # Make prediction
            results = predictor.predict(image, ingredients_text, top_k=5)
            
            if not results:
                return "No predictions generated", "0%", []
            
            # Format results
            best_class, best_confidence = results[0]
            confidence_str = f"{best_confidence:.1%}"
            
            # Create detailed results table
            detailed_results = []
            for class_name, confidence in results:
                detailed_results.append([class_name, f"{confidence:.1%}"])
            
            return best_class, confidence_str, detailed_results
            
        except Exception as e:
            error_msg = f"Prediction failed: {str(e)}"
            return error_msg, "0%", []
    
    # Get class information for display
    supported_classes = predictor.get_supported_classes()
    class_info = f"Supporting {len(supported_classes)} textile processes: " + ", ".join(supported_classes)
    
    # Create the Gradio interface
    with gr.Blocks(
        title="Textile Process Classification",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .info-box {
            background-color: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
        }
        .result-box {
            background-color: #e8f5e8;
            border: 1px solid #28a745;
            border-radius: 8px;
            padding: 1rem;
        }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>Intelligent Textile Process Classification</h1>
            <p>Advanced AI system using Swin Transformer + CLIP + LBP texture analysis</p>
        </div>
        """)
        
        with gr.Row():
            # Input column
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                
                image_input = gr.Image(
                    type="pil",
                    label="Upload Fabric Image",
                    height=350
                )
                
                ingredients_input = gr.Textbox(
                    label="Fabric Ingredients (Optional)",
                    placeholder="e.g., Cotton, Polyester, Elastane",
                    lines=3,
                    info="Providing ingredient information can improve prediction accuracy"
                )
                
                analyze_btn = gr.Button(
                    "Analyze Fabric",
                    variant="primary",
                    size="lg"
                )
            
            # Output column
            with gr.Column(scale=1):
                gr.Markdown("### Analysis Results")
                
                prediction_output = gr.Textbox(
                    label="Predicted Process",
                    interactive=False,
                    container=True
                )
                
                confidence_output = gr.Textbox(
                    label="Confidence Score",
                    interactive=False
                )
                
                detailed_output = gr.Dataframe(
                    headers=["Process Type", "Confidence"],
                    label="Top 5 Predictions",
                    interactive=False,
                    wrap=True
                )
        
        # Information section
        gr.HTML(f"""
        <div class="info-box">
            <h4>System Information</h4>
            <p><strong>Architecture:</strong> Swin Transformer + CLIP + LBP Texture Analysis</p>
            <p><strong>Supported Classes:</strong> {len(supported_classes)} different textile processes</p>
            <details>
                <summary><strong>View All Supported Processes</strong></summary>
                <div style="margin-top: 10px; padding: 10px; background: white; border-radius: 5px;">
                    {', '.join(supported_classes)}
                </div>
            </details>
        </div>
        """)
        
        # Usage instructions
        gr.Markdown("""
        ### How to Use
        
        1. **Upload Image**: Select a clear, well-lit fabric image
        2. **Add Ingredients** (Optional): Describe the fabric materials for better accuracy
        3. **Analyze**: Click the analyze button to get AI predictions
        4. **Review Results**: Check the predicted process and confidence scores
        
        **Tips for Best Results:**
        - Use high-quality, well-focused images
        - Include ingredient information when available
        - Ensure good lighting and contrast
        - Avoid blurry or heavily distorted images
        """)
        
        # Set up the prediction event
        analyze_btn.click(
            fn=predict_fabric,
            inputs=[image_input, ingredients_input],
            outputs=[prediction_output, confidence_output, detailed_output]
        )
        
        # Example section
        gr.HTML("""
        <div class="info-box">
            <h4>Example Use Cases</h4>
            <ul>
                <li><strong>Quality Control:</strong> Automated fabric process verification</li>
                <li><strong>Supply Chain:</strong> Intelligent material categorization</li>
                <li><strong>Research:</strong> Textile process analysis and optimization</li>
                <li><strong>Education:</strong> Learning tool for textile manufacturing</li>
            </ul>
        </div>
        """)
    
    return demo


def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Textile Process Classification Inference")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="best_textile_model.pt",
        help="Path to trained model file"
    )
    parser.add_argument(
        "--interface", 
        choices=["web", "cli"], 
        default="web",
        help="Interface type: 'web' for Gradio UI, 'cli' for command line"
    )
    parser.add_argument(
        "--image", 
        type=str,
        help="Path to image file (for CLI mode)"
    )
    parser.add_argument(
        "--ingredients", 
        type=str, 
        default="",
        help="Fabric ingredients description (for CLI mode)"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="127.0.0.1",
        help="Host for web interface"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Port for web interface"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create public sharing link (WARNING: This exposes your model publicly)"
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Model file not found: {args.model_path}")
        return
    
    if args.interface == "web":
        # Launch web interface
        print("Launching Textile Classification Web Interface...")
        
        # Security warning for public sharing
        if args.share:
            print("WARNING: --share flag is enabled. This will create a public URL accessible from anywhere.")
            print("Your model and any uploaded images will be exposed to the internet.")
            response = input("Do you want to continue? (y/N): ")
            if response.lower() != 'y':
                print("Aborted. Run without --share for local access only.")
                return
        
        demo = create_gradio_interface(args.model_path)
        if demo:
            demo.launch(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                inbrowser=True
            )
        else:
            print("Failed to create web interface")
    
    elif args.interface == "cli":
        # Command line interface
        if not args.image:
            print("Please provide --image path for CLI mode")
            return
        
        if not os.path.exists(args.image):
            print(f"Image file not found: {args.image}")
            return
        
        print("Loading model and making prediction...")
        
        try:
            # Initialize predictor
            predictor = TextilePredictor(args.model_path)
            
            # Load and process image
            image = Image.open(args.image)
            results = predictor.predict(image, args.ingredients, top_k=3)
            
            # Display results
            print("\\nPrediction Results:")
            print("=" * 50)
            for i, (class_name, confidence) in enumerate(results, 1):
                print(f"{i}. {class_name:<20} | {confidence:.1%}")
            
        except Exception as e:
            print(f"CLI prediction failed: {e}")


if __name__ == "__main__":
    main()
