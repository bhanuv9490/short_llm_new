from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class SmallLLM2:
    def __init__(self, model_name="gpt2", device=None):
        print(f"Loading {model_name}...")
        
        # Force CPU for GPT-2 due to CUDA compatibility issues
        self.device = "cpu"
        self.use_amp = False
        print("Note: GPT-2 is configured to use CPU for better compatibility")
        print("This model will run slower but should be more stable")
        
        # Keep track of original device for reference
        self.original_device = device or "auto"
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Always use float32 for CPU
        torch_dtype = torch.float32
        
        try:
            # Simple CPU loading for maximum compatibility
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
        try:
            # Tokenize input and move to device
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate text with CPU
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    no_repeat_ngram_size=2  # Add some variety
                )
            
            # Decode and clean up the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input prompt from response if it's there
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            return generated_text
            
        except Exception as e:
            return f"Error generating text: {str(e)}"