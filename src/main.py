from models.saved_models.small_llm_1 import SmallLLM1
from models.saved_models.small_llm_2 import SmallLLM2
from typing import Optional, Dict, Any

class LLMInterface:
    def __init__(self, device: str = None):
        """
        Initialize the LLM interface with both models.
        
        Args:
            device (str, optional): Device to run models on ('cuda' or 'cpu').
                                 If None, automatically uses CUDA if available.
        """
        print("Initializing LLM Interface...")
        self.llm1 = SmallLLM1()
        self.llm2 = SmallLLM2()
        print("LLM Interface initialized successfully!")

    def process_text_with_llm1(self, text: str, max_length: int = 100, temperature: float = 0.7) -> str:
        """
        Process text using Microsoft's Phi-2 model.
        
        Args:
            text (str): Input text to process
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness (0.1-1.0)
            
        Returns:
            str: Processed text completion
        """
        print(f"Processing text with Phi-2: {text[:50]}...")
        try:
            return self.llm1.predict(text, max_length=max_length, temperature=temperature)
        except Exception as e:
            return f"Error processing text with Phi-2: {str(e)}"

    def generate_text_with_llm2(self, prompt: str, max_length: int = 100, 
                              temperature: float = 0.7, top_p: float = 0.9) -> str:
        """
        Generate text using GPT-2 model.
        
        Args:
            prompt (str): Input prompt
            max_length (int): Maximum length of generated text
            temperature (float): Controls randomness (0.1-1.0)
            top_p (float): Nucleus sampling parameter (0.0-1.0)
            
        Returns:
            str: Generated text
        """
        print(f"Generating text with GPT-2 based on: {prompt[:50]}...")
        try:
            return self.llm2.generate(
                prompt, 
                max_length=max_length, 
                temperature=temperature, 
                top_p=top_p
            )
        except Exception as e:
            return f"Error generating text with GPT-2: {str(e)}"

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LLM Interface')
    parser.add_argument('--mode', type=str, choices=['phi2', 'gpt2', 'both'], default='both',
                       help='Which model to test')
    parser.add_argument('--prompt', type=str, 
                       default="Explain the concept of artificial intelligence in simple terms.",
                       help='Prompt to use for generation')
    args = parser.parse_args()
    
    print("Starting LLM Interface...")
    interface = LLMInterface()
    
    if args.mode in ['phi2', 'both']:
        print("\n--- Testing Phi-2 (SmallLLM1) ---")
        result = interface.process_text_with_llm1(args.prompt)
        print("\nPhi-2 Response:")
        print("-" * 50)
        print(result)
        print("-" * 50 + "\n")
    
    if args.mode in ['gpt2', 'both']:
        print("\n--- Testing GPT-2 (SmallLLM2) ---")
        result = interface.generate_text_with_llm2(args.prompt)
        print("\nGPT-2 Response:")
        print("-" * 50)
        print(result)
        print("-" * 50 + "\n")