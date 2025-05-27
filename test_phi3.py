import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_phi3_loading():
    print("Testing Phi-3 model loading...")
    model_name = "microsoft/phi-3-mini-4k-instruct"
    
    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left',
            truncation_side='left'
        )
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map='auto',
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        )
        
        print(f"Model loaded successfully on {model.device}")
        print("Model architecture:", model.config.architectures[0])
        
        return True
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

if __name__ == "__main__":
    test_phi3_loading()
