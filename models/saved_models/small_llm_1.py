from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class SmallLLM1:
    def __init__(self, model_name="microsoft/phi-2"):
        print(f"Loading {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            trust_remote_code=True
        ).to(self.device)
        print(f"{model_name} loaded successfully on {self.device}")

    def predict(self, text, max_length=100, temperature=0.7):
        """
        Generate text completion based on the input text.
        
        Args:
            text (str): Input prompt
            max_length (int): Maximum length of the generated text
            temperature (float): Controls randomness in generation (lower = more deterministic)
            
        Returns:
            str: Generated text completion
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)