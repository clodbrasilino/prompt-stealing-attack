"""
Modifier Detector module for PromptStealer
Adapted from ML-Decoder (https://github.com/Alibaba-MIIL/ML_Decoder)
"""
import torch
import torchvision.transforms as transforms
from ml_decoder.models import create_model
import argparse


class ModifierDetector:
    """
    Modifier Detector using ML-Decoder for detecting style words and modifiers
    """
    
    def __init__(self, device="cuda", num_classes=7672, threshold=0.6):
        """
        Initialize Modifier Detector
        
        Args:
            device: Device to run the model ('cuda' or 'cpu')
            num_classes: Number of modifier classes
            threshold: Confidence threshold for predictions
        """
        self.device = device
        self.num_classes = num_classes
        self.threshold = threshold
        self.model = None
        self.transform = None
        self.category_map = None
        
    def load_ckpt(self, checkpoint_path):
        """
        Load checkpoint for the modifier detector
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
        """
        print(f"\nLoading modifier detector from {checkpoint_path}...")
        
        # Create ML-Decoder args
        args = self._create_ml_decoder_args()
        
        # Create model
        self.model = create_model(args).to(self.device)
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in ckpt:
            self.model.load_state_dict(ckpt['model'], strict=True)
        else:
            self.model.load_state_dict(ckpt, strict=True)
        print(f'✅ Resume from checkpoint: {checkpoint_path}')
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor(),
        ])
        
        self.model.eval()
        print("✅ Modifier detector loaded successfully")
        
    def _create_ml_decoder_args(self):
        """Create arguments for ML-Decoder model creation"""
        parser = argparse.ArgumentParser(description='ML Decoder Args')
        parser.add_argument('--model-name', default='tresnet_l')
        parser.add_argument('--num-classes', default=self.num_classes, type=int)
        parser.add_argument('--use-ml-decoder', default=1, type=int)
        parser.add_argument('--num-of-groups', default=-1, type=int)  # full-decoding
        parser.add_argument('--decoder-embedding', default=768, type=int)
        parser.add_argument('--zsl', default=0, type=int)
        parser.add_argument('--load-pretrain', default=0, type=int)
        args, _ = parser.parse_known_args()
        return args
        
    def detect(self, images, category_map=None):
        """
        Detect modifiers for input images
        
        Args:
            images: Tensor of images [B, C, H, W] or list of tensors/PIL images
            category_map: Dictionary mapping class indices to modifier names
            
        Returns:
            List of dictionaries with detected modifiers and their confidences
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_ckpt() first.")
            
        self.model.eval()
        self.category_map = category_map
        
        # Handle different input types
        if isinstance(images, list):
            # List of tensors or PIL images
            images_transformed = []
            for img in images:
                if hasattr(img, 'cpu'):  # Tensor
                    # Convert to PIL and back to tensor after transform
                    img_pil = transforms.ToPILImage()(img.cpu())
                else:  # PIL Image
                    img_pil = img
                images_transformed.append(self.transform(img_pil))
            images_tensor = torch.stack(images_transformed).to(self.device)
        else:
            # Batch tensor
            if images.dim() == 3:
                images = images.unsqueeze(0)
            images_tensor = images.to(self.device)
            
        # Detect modifiers
        sigmoid = torch.nn.Sigmoid()
        with torch.no_grad():
            outputs = sigmoid(self.model(images_tensor)).cpu()
            
        # Process predictions
        pred_batch = []
        for row_idx in range(len(outputs)):
            one_output = outputs[row_idx].numpy()
            
            if self.category_map is not None:
                # Return a list of modifier *names* (strings) above threshold.
                # This is what downstream code expects for ', '.join(modifiers).
                d = dict(zip(self.category_map.keys(), one_output))
                pred_keywords = [str(k) for k, v in d.items() if v > self.threshold]
            else:
                # Return empty list if no category map — avoids "numpy array in join" errors
                pred_keywords = []
                
            pred_batch.append(pred_keywords)

        # For the single-image case return the inner List[str] directly so that
        # callers can do `', '.join(modifiers)` without an extra [0] indexing.
        if len(pred_batch) == 1:
            return pred_batch[0]
        return pred_batch
    
    def set_threshold(self, threshold):
        """Update the confidence threshold"""
        self.threshold = threshold
        
    def set_category_map(self, category_map):
        """Set the category mapping"""
        self.category_map = category_map
        
    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
