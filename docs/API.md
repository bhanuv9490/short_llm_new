# API Reference

## LLMInterface

The main interface for interacting with different LLM models.

### `__init__(self)`

Initialize the LLM interface and load all available models.

**Example:**
```python
from src.main import LLMInterface

interface = LLMInterface()
```

### `process_text_with_llm1(self, text: str) -> str`

Process the input text using LLM1.

**Parameters:**
- `text` (str): The input text to process

**Returns:**
- str: Processed text result

**Example:**
```python
result = interface.process_text_with_llm1("Example text")
```

### `generate_text_with_llm2(self, prompt: str) -> str`

Generate text based on the given prompt using LLM2.

**Parameters:**
- `prompt` (str): The prompt for text generation

**Returns:**
- str: Generated text

**Example:**
```python
generated = interface.generate_text_with_llm2("Write a story about")
```

## Model Implementations

### SmallLLM1

A simple language model implementation.

#### `predict(self, text: str) -> str`

Process the input text.

### SmallLLM2

Another simple language model implementation.

#### `generate(self, prompt: str) -> str`

Generate text based on the prompt.
