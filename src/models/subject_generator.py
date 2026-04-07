"""
Subject Generator module for PromptStealer
Adapted from BLIP (https://github.com/salesforce/BLIP)
"""
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from ruamel.yaml import YAML
from src.BLIP_finetune.models.blip import blip_decoder


class SubjectGenerator:
    """
    Subject Generator using BLIP decoder for image captioning
    """
    
    def __init__(self, device="cuda", config_path='./src/BLIP_finetune/configs/lexica_subject.yaml'):
        """
        Initialize Subject Generator
        
        Args:
            device: Device to run the model ('cuda' or 'cpu')
            config_path: Path to BLIP configuration file
        """
        self.device = device
        self.blip_image_eval_size = 384
        self.config_path = config_path
        self.model = None
        self.transform = None
        
    def load_ckpt(self, checkpoint_path):
        """
        Load checkpoint for the subject generator
        
        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
        """
        print(f"Loading subject generator from {checkpoint_path}...")
        
        # Load configuration (using ruamel.yaml API for version >= 0.18)
        yaml_obj = YAML(typ='rt')
        config = yaml_obj.load(open(self.config_path, 'r'))
        
        # Create model
        self.model = blip_decoder(
            pretrained=config['pretrained'],
            image_size=config['image_size'],
            vit=config['vit'],
            vit_grad_ckpt=config['vit_grad_ckpt'],
            vit_ckpt_layer=config['vit_ckpt_layer'],
            prompt=config['prompt'],
            med_config=config['med_config']
        )
        
        # Load checkpoint
        print(f"Resume from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model = self.model.to(self.device)
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((self.blip_image_eval_size, self.blip_image_eval_size), 
                            interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                               (0.26862954, 0.26130258, 0.27577711))
        ])
        
        self.model.eval()
        print("✅ Subject generator loaded successfully")
        
    def generate(self, images):
        """
        Generate subjects for input images
        
        Args:
            images: Tensor of images [B, C, H, W] or list of PIL images
            
        Returns:
            List of generated subjects (strings)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_ckpt() first.")
        
        self.model.eval()
        
        # Handle different input types
        if isinstance(images, list):
            # List of PIL images or tensors
            images_transformed = []
            for img in images:
                if hasattr(img, 'cpu'):  # Tensor
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
            
        # Generate subjects
        with torch.no_grad():
            generated_subjects = self.model.generate(
                images_tensor,
                sample=False,
                num_beams=3,
                max_length=20,
                min_length=5
            )
            
        return generated_subjects
    
    def eval(self):
        """Set model to eval mode"""
        if self.model is not None:
            self.model.eval()
