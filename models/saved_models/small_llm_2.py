import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class SmallLLM2:
    def __init__(self, model_name="Qwen/Qwen3-235B-A22B", device=None):
        # Check CUDA availability and memory
        self.has_cuda = torch.cuda.is_available()
        if self.has_cuda:
            # Check available GPU memory
            total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # in GB
            if total_mem < 40:  # Need significant memory for this model
                print(f"Warning: Qwen3-235B-A22B requires substantial GPU memory. Found {total_mem:.1f}GB.")
                print("Consider using model parallelism or a smaller model if you encounter memory issues.")
        
        self.device = device or ('cuda' if self.has_cuda else 'cpu')
        self.use_amp = self.device.startswith('cuda')  # Enable AMP for CUDA
        self.use_4bit = True  # Enable 4-bit quantization by default
        
        print(f"Initializing {model_name} on {self.device.upper()}")
        print(f"CUDA available: {self.has_cuda}")
        if self.has_cuda:
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}GB")
        
        # Initialize tokenizer for Qwen
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left',
            truncation_side='left'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set chat template if available
        if hasattr(self.tokenizer, 'chat_template') and not self.tokenizer.chat_template:
            self.tokenizer.chat_template = "{{ bos_token }}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        
        # Configure model with appropriate precision and quantization
        torch_dtype = torch.bfloat16 if self.use_amp else torch.float32
        
        try:
            if self.has_cuda:
                # Clear CUDA cache
                torch.cuda.empty_cache()
                
                # Configure model loading with 4-bit quantization if enabled
                quantization_config = None
                if self.use_4bit:
                    from transformers import BitsAndBytesConfig
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                    )
                
                # Try different loading strategies
                load_kwargs = {
                    'torch_dtype': torch_dtype,
                    'device_map': 'auto',
                    'trust_remote_code': True,
                    'quantization_config': quantization_config,
                    'pad_token_id': self.tokenizer.eos_token_id,
                    'offload_folder': 'offload',
                    'offload_state_dict': True
                }
                
                # Try loading with different configurations
                for attempt in range(3):
                    try:
                        if attempt == 0:
                            # First try with flash attention if available
                            if torch.cuda.is_available():
                                load_kwargs['attn_implementation'] = 'flash_attention_2'
                                self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                                print("Loaded with flash attention 2")
                                break
                        elif attempt == 1:
                            # Then try with default settings
                            if 'attn_implementation' in load_kwargs:
                                del load_kwargs['attn_implementation']
                            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                            print("Loaded with default attention")
                            break
                        else:
                            # Finally try with safe tensors
                            load_kwargs['use_safetensors'] = True
                            self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                            print("Loaded with safe tensors")
                            break
                    except Exception as e:
                        if attempt == 2:  # Last attempt
                            raise RuntimeError(f"Failed to load model after multiple attempts: {str(e)}")
                        print(f"Attempt {attempt + 1} failed: {str(e)}")
                        continue
                
                # If model is not already on GPU, move it
                if str(self.model.device) != 'cuda:0':
                    self.model = self.model.to('cuda:0')
            else:
                print("CUDA not available. Qwen3-235B-A22B is not recommended for CPU usage.")
                print("Loading in 8-bit mode for CPU to reduce memory usage...")
                
                from transformers import BitsAndBytesConfig
                
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    bnb_8bit_compute_dtype=torch.float32
                )
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    device_map='auto',
                    trust_remote_code=True,
                    quantization_config=quantization_config,
                    pad_token_id=self.tokenizer.eos_token_id,
                    offload_folder='offload',
                    offload_state_dict=True
                )
                
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

    def generate(self, prompt, max_length=512, temperature=0.7, top_p=0.9, top_k=40, repetition_penalty=1.1, system_message=None):
        """
        Generate text based on the input prompt using Qwen3-235B-A22B
        
        Args:
            prompt (str): Input text prompt or conversation
            max_length (int): Maximum length of the generated text (up to 8192)
            temperature (float): Controls randomness (0.1-1.0, lower is more deterministic)
            top_p (float): Nucleus sampling parameter (0.0-1.0)
            top_k (int): Top-k sampling parameter (1-1000)
            repetition_penalty (float): Penalty for repeating tokens (1.0-2.0)
            system_message (str, optional): System message to set the assistant's behavior
            
        Returns:
            str: Generated text
        """
        
        # Prepare messages for chat format
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        try:
            # Clean and prepare the prompt
            prompt = prompt.strip()
            if not prompt.endswith(('.', '!', '?')):
                prompt += '.'
                
            # Tokenize inputs with chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                # Use chat template if available
                tokenized = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(self.device)
                inputs = {"input_ids": tokenized, "attention_mask": torch.ones_like(tokenized).to(self.device)}
            else:
                # Fallback to regular tokenization
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=min(8192, max_length),  # Qwen3 supports up to 32K tokens
                    return_attention_mask=True
                ).to(self.device)
            
            # Generate text with enhanced parameters for Qwen
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda' if self.use_amp else 'cpu', 
                                     dtype=torch.bfloat16 if self.use_amp else torch.float32):
                    outputs = self.model.generate(
                        **inputs,
                        max_length=min(max_length, 8192),  # Qwen3 supports up to 32K tokens
                        temperature=max(0.1, min(1.0, temperature)),
                        top_p=max(0.1, min(1.0, top_p)),
                        top_k=max(1, top_k),
                        do_sample=True,
                        no_repeat_ngram_size=4,  # Slightly larger for better coherence
                        repetition_penalty=repetition_penalty,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        num_return_sequences=1,
                        early_stopping=True,
                        max_new_tokens=max_length,
                        use_cache=True,
                        return_dict_in_generate=True
                    )
            
            # Decode and clean the output
            if hasattr(outputs, 'sequences'):
                # Handle newer Transformers output format
                output_ids = outputs.sequences[0]
            else:
                # Fallback to older format
                output_ids = outputs[0]
                
            generated_text = self.tokenizer.decode(
                output_ids, 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Remove the input prompt from the output if it's present
            if prompt and prompt in generated_text:
                generated_text = generated_text.split(prompt, 1)[-1].strip()
            
            # Remove prompt from response if present
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
                
            # Clean up any partial sentences at the end
            last_punct = max(
                generated_text.rfind('.'),
                generated_text.rfind('!'),
                generated_text.rfind('?')
            )
            
            if last_punct > 0:
                generated_text = generated_text[:last_punct + 1].strip()
                
            return generated_text or "I'm not sure how to respond to that."
            
        except Exception as e:
            return f"I encountered an error: {str(e)}"