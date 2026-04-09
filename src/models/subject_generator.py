"""
Subject Generator module for PromptStealer
Adapted from BLIP (https://github.com/salesforce/BLIP)

Can be imported in two ways:
  1. pip install git+https://github.com/clodbrasilino/prompt-stealing-attack.git
     → config is found automatically via importlib.resources
  2. sys.path.insert(0, 'path/to/repo/src')
     → pass config_path explicitly or rely on relative fallback
"""
import os
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import InterpolationMode
from ruamel.yaml import YAML
# Absolute import: with package_dir={"": "src"}, pip install puts
# src/BLIP_finetune/ → BLIP_finetune/ (top-level in site-packages).
# With sys.path.insert(0, 'src') the src/ dir IS on sys.path, so
# BLIP_finetune is also found as a top-level package → works in both modes.
from BLIP_finetune.models.blip import blip_decoder

# Try to find the default config via importlib.resources (pip-installed).
# Fall back to a relative path (cloned repo + sys.path.insert).
try:
    import importlib.resources as _resources

    _CONFIG_FILE = "lexica_subject.yaml"

    def _get_default_config_path() -> str:
        """Return path to lexica_subject.yaml inside the installed package."""
        ref = _resources.files("models.BLIP_finetune") / "configs" / _CONFIG_FILE
        if ref.is_file():
            return str(ref)
        # Try older API
        with _resources.as_file(ref) as p:
            return str(p)

    _DEFAULT_CONFIG = _get_default_config_path()

except Exception:
    # Older Python or importlib.resources without files()
    _DEFAULT_CONFIG = os.path.join(
        os.path.dirname(__file__),
        "..", "BLIP_finetune", "configs", "lexica_subject.yaml",
    )


class SubjectGenerator:
    """
    Subject Generator using BLIP decoder for image captioning.
    """

    def __init__(self, device="cuda", config_path=None):
        """
        Initialize Subject Generator.

        Args:
            device: Device to run the model ('cuda' or 'cpu')
            config_path: Path to BLIP configuration file.
                         Auto-detected when installed via pip; pass explicitly
                         when using sys.path.insert from a cloned repo.
        """
        self.device = device
        self.blip_image_eval_size = 384
        self.config_path = config_path or _DEFAULT_CONFIG
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
