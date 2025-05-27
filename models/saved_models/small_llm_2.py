from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

class SmallLLM2:
    def __init__(self, model_name="gpt2", device='cpu'):
        print(f"Loading {model_name}...")
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Always use float32 for CPU
            pad_token_id=self.tokenizer.eos_token_id
        ).to(self.device)
        print(f"{model_name} loaded successfully on {self.device.upper()}")

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
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)