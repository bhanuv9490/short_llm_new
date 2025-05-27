import torch
from models.saved_models.small_llm_1 import SmallLLM1
from models.saved_models.small_llm_2 import SmallLLM2
from typing import Optional, Dict, Any
import warnings

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)

class LLMInterface:
    def __init__(self, device: str = None):
        """
        Initialize the LLM interface with both models.
        
        Args:
            device (str, optional): Device to run models on ('cuda' or 'cpu').
                                 If None, automatically uses CUDA if available.
        """
        print("=" * 50)
        print("Initializing LLM Interface...")
        
        # Check CUDA availability and compatibility
        self.has_cuda = torch.cuda.is_available()
        
        # Print CUDA info if available
        if self.has_cuda:
            try:
                print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
                print(f"CUDA Version: {torch.version.cuda}")
                print(f"PyTorch CUDA Version: {torch.version.cuda}")
                print(f"GPU Memory: {torch.cuda.memory_allocated(0)/1e9:.1f}GB / {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
                # Check compute capability
                capability = torch.cuda.get_device_capability()
                print(f"CUDA Capability: {capability[0]}.{capability[1]}")
            except Exception as e:
                print(f"Warning: Could not get CUDA device info: {e}")
                self.has_cuda = False
        else:
            print("CUDA is not available. Using CPU.")
        
        # Auto-detect CUDA if device is not specified
        if device is None:
            self.device = "cuda" if self.has_cuda else "cpu"
        else:
            self.device = device
        
        # Fall back to CPU if CUDA is not properly configured
        if self.device.startswith('cuda') and not self.has_cuda:
            print("Warning: CUDA was requested but is not available. Falling back to CPU.")
            self.device = "cpu"
        
        print(f"\nUsing device: {self.device.upper()}")
        if self.device.startswith('cuda'):
            print("Note: Using CUDA with mixed precision if available")
        print("=" * 50)
        
        # Initialize models (will be loaded on first use)
        print("\nInitializing models...")
        try:
            self.llm1 = SmallLLM1(device=self.device)
            self.llm2 = SmallLLM2(device=self.device)
            print("\n" + "=" * 50)
            print("LLM Interface initialized successfully!")
            print(f"Models will be loaded on first use on {self.device.upper()}")
            print("=" * 50)
        except Exception as e:
            print(f"\nError initializing models: {e}")
            print("\n" + "!" * 50)
            print("ERROR: Could not initialize models with current configuration.")
            print("Attempting to initialize with CPU...")
            print("!" * 50 + "\n")
            
            self.device = "cpu"
            try:
                self.llm1 = SmallLLM1(device=self.device)
                self.llm2 = SmallLLM2(device=self.device)
                print("\n" + "=" * 50)
                print("LLM Interface initialized on CPU.")
                print("Note: Performance will be slower than with CUDA")
                print("=" * 50)
            except Exception as cpu_e:
                print("\n" + "!" * 50)
                print("FATAL: Could not initialize models on CPU either.")
                print(f"Error: {cpu_e}")
                print("Please check your installation and try again.")
                print("!" * 50)
                raise

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