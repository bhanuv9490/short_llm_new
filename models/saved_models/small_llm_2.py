from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class SmallLLM2:
    def __init__(self, model_name="gpt2", device=None):
        print(f"Loading {model_name}...")
        
        # Check CUDA availability and compatibility
        self.has_cuda = torch.cuda.is_available()
        self.device = device or ("cuda" if self.has_cuda else "cpu")
        
        # Fall back to CPU if CUDA is not properly configured
        if self.device.startswith('cuda') and not self.has_cuda:
            print("Warning: CUDA is not available. Falling back to CPU.")
            self.device = "cpu"
            
        self.use_amp = self.device.startswith('cuda')
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Configure model with appropriate precision
        torch_dtype = torch.float16 if self.use_amp else torch.float32
        
        try:
            # Try loading with CUDA if available
            if self.use_amp and self.has_cuda:
                try:
                    # Initialize GradScaler for mixed precision first
                    self.scaler = torch.amp.GradScaler(device_type='cuda')
                    print("Initialized CUDA with mixed precision")
                    
                    # Use device_map='auto' for better memory management with CUDA
                    self.model = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        pad_token_id=self.tokenizer.eos_token_id,
                        device_map='auto'
                    )
                except Exception as e:
                    print(f"Warning: Mixed precision initialization failed: {e}")
                    print("Falling back to FP32 on CUDA...")
                    self.use_amp = False
                    torch_dtype = torch.float32
                    self.model = GPT2LMHeadModel.from_pretrained(
                        model_name,
                        torch_dtype=torch_dtype,
                        pad_token_id=self.tokenizer.eos_token_id,
                        device_map='auto' if self.has_cuda else None
                    )
            else:
                # For CPU or when CUDA is not available
                self.model = GPT2LMHeadModel.from_pretrained(
                    model_name,
                    torch_dtype=torch_dtype,
                    pad_token_id=self.tokenizer.eos_token_id
                ).to(self.device)
                
            print(f"{model_name} loaded successfully on {self.device.upper()}" + 
                  (" with mixed precision" if self.use_amp else ""))
                  
        except RuntimeError as e:
            if 'CUDA' in str(e):
                print(f"CUDA error: {e}")
                print("Falling back to CPU...")
                self.device = "cpu"
                self.use_amp = False
                # Retry with CPU
                self.model = GPT2LMHeadModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    pad_token_id=self.tokenizer.eos_token_id
                ).to(self.device)
                print(f"{model_name} loaded on CPU")
            else:
                raise

    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9):
        """
        Generate text based on the input prompt using GPT-2.
        
        Args:
            prompt (str): Input text prompt
            max_length (int): Maximum length of the generated text
            temperature (float): Controls randomness in generation (lower = more deterministic)
            top_p (float): Nucleus sampling parameter (0.0 to 1.0)
            
        Returns:
            str: Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        **inputs,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)