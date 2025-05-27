from transformers import AutoModelForCausalLM, AutoTokenizer, logging
import torch
import time
from tqdm import tqdm
import warnings

# Suppress some unnecessary warnings
logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning)

class SmallLLM1:
    def __init__(self, model_name="microsoft/phi-4", device=None):
        print(f"Initializing {model_name}...")
        
        # Check CUDA availability and compatibility
        self.has_cuda = torch.cuda.is_available()
        self.device = device or ("cuda" if self.has_cuda else "cpu")
        self.use_amp = self.device.startswith('cuda')  # Enable AMP for CUDA
        self.is_initialized = False
        
        print(f"Initializing {model_name} on {self.device.upper()}")
        print(f"CUDA available: {self.has_cuda}")
        if self.has_cuda:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left',
            truncation_side='left'
        )
        
        # Set data type based on device and AMP
        torch_dtype = torch.bfloat16 if (self.use_amp and self.has_cuda) else torch.float32
        
        try:
            if self.has_cuda:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Try different loading strategies
                load_kwargs = {
                    'torch_dtype': torch_dtype,
                    'device_map': 'auto',
                    'trust_remote_code': True,
                    'offload_folder': 'offload',
                    'offload_state_dict': True
                }
                
                # Try loading with different configurations
                for attempt in range(3):
                    try:
                        if attempt == 0 and self.has_cuda:
                            # First try with flash attention if available
                            load_kwargs['attn_implementation'] = 'flash_attention_2'
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name, **load_kwargs)
                            print("Loaded with flash attention 2")
                            break
                        elif attempt == 1:
                            # Then try with default settings
                            if 'attn_implementation' in load_kwargs:
                                del load_kwargs['attn_implementation']
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name, **load_kwargs)
                            print("Loaded with default attention")
                            break
                        else:
                            # Finally try with safe tensors
                            load_kwargs['use_safetensors'] = True
                            self.model = AutoModelForCausalLM.from_pretrained(
                                model_name, **load_kwargs)
                            print("Loaded with safe tensors")
                            break
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            raise RuntimeError(f"Failed to load model after multiple attempts: {str(e)}")
                        print(f"Attempt {attempt + 1} failed: {str(e)}")
                        continue
            else:
                print("CUDA not available. Falling back to CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to('cpu')
            
            print(f"Model loaded on {str(self.model.device).upper()}" + 
                  (" with mixed precision" if self.use_amp else ""))
            
            # Initialize GradScaler if using mixed precision
            self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
            self.is_initialized = True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to CPU...")
            self.device = 'cpu'
            self.use_amp = False
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                trust_remote_code=True
            ).to('cpu')
            self.is_initialized = True
            print(f"{model_name} loaded on CPU")
        
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
            
            # Move inputs to the same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate text with enhanced parameters for Phi-4
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda' if self.use_amp else 'cpu', 
                                     dtype=torch.bfloat16 if self.use_amp else torch.float32):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=min(max_length, 4096),  # Phi-4 supports longer context
                        temperature=max(0.1, min(1.0, temperature)),
                        top_p=max(0.1, min(1.0, top_p)),
                        top_k=max(1, top_k),
                        do_sample=True,
                        no_repeat_ngram_size=4,  # Slightly larger for better coherence
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        early_stopping=True,
                        max_new_tokens=max_length,
                        use_cache=True,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
            
            # Decode and clean the output
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Remove prompt from response if present
            if generated_text.startswith(text):
                generated_text = generated_text[len(text):].strip()
                
            # Clean up any partial sentences at the end
            last_punct = max(
                generated_text.rfind('.'),
                generated_text.rfind('!'),
                generated_text.rfind('?')
            )
            
            if last_punct > 0:
                generated_text = generated_text[:last_punct + 1].strip()
            
            # Add fallback response if empty
            if not generated_text.strip():
                return "I'm not sure how to respond to that. Could you please rephrase your question?"
                
            return generated_text
            
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            print(error_msg)
            return error_msg