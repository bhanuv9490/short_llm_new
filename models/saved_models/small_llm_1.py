from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
import time
from tqdm import tqdm
import warnings

# Suppress some unnecessary warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

class SmallLLM1:
    def __init__(self, model_name="microsoft/phi-2", device=None):
        print(f"Initializing {model_name}...")
        
        # Check CUDA availability and compatibility
        self.has_cuda = torch.cuda.is_available()
        self.device = device or ("cuda" if self.has_cuda else "cpu")
        
        # Fall back to CPU if CUDA is not properly configured
        if self.device.startswith('cuda') and not self.has_cuda:
            print("Warning: CUDA is not available. Falling back to CPU.")
            self.device = "cpu"
            
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_initialized = False
        
        # Use mixed precision for CUDA if available
        self.use_amp = self.device.startswith('cuda') and torch.cuda.is_available()
        if self.use_amp:
            try:
                # Use the new GradScaler API
                self.scaler = torch.amp.GradScaler(device_type='cuda')
                print("Initialized CUDA with mixed precision")
            except Exception as e:
                print(f"Warning: Could not initialize mixed precision: {e}")
                self.use_amp = False
                self.device = 'cpu'
        
    def _download_with_progress(self):
        """Download model with progress bar"""
        print(f"Downloading {self.model_name} (this may take a few minutes, ~2.7GB)...")
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Configure model with appropriate precision
            torch_dtype = torch.float16 if self.use_amp else torch.float32
            
            # Try loading with CUDA if available
            if self.use_amp:
                # Use device_map='auto' for better memory management with CUDA
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True,
                    device_map='auto' if self.has_cuda else None
                )
            else:
                # For CPU or when CUDA is not available
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                ).to(self.device)
            
            self.is_initialized = True
            print(f"\n{self.model_name} loaded successfully on {self.device.upper()}" +
                  (" with mixed precision" if self.use_amp else ""))
                  
        except RuntimeError as e:
            if 'CUDA' in str(e):
                print(f"CUDA error: {e}")
                print("Falling back to CPU...")
                self.device = "cpu"
                self.use_amp = False
                # Retry with CPU
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)
                self.is_initialized = True
                print(f"{self.model_name} loaded on CPU")
            else:
                raise
        
    def ensure_initialized(self):
        """Ensure model is downloaded and initialized"""
        if not self.is_initialized:
            self._download_with_progress()

    def predict(self, text, max_length=100, temperature=0.7):
        """
        Generate text completion based on the input text.
        
        Args:
            text (str): Input prompt
            max_length (int): Maximum length of the generated text
            temperature (float): Controls randomness in generation (lower = more deterministic)
            
        Returns:
            str: Generated text completion or error message
        """
        try:
            # Ensure model is loaded
            self.ensure_initialized()
            
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            
            print(f"Generating response (max_length={max_length}, temp={temperature})...")
            
            # Generate response with progress
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model.generate(
                            **inputs,
                            max_length=max_length,
                            temperature=temperature,
                            do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id,
                            num_return_sequences=1,
                            no_repeat_ngram_size=2
                        )
                else:
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        no_repeat_ngram_size=2
                    )
                
            # Decode and clean up the output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response if it's there
            if response.startswith(text):
                response = response[len(text):].strip()
                
            return response
            
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            print(error_msg)
            return error_msg