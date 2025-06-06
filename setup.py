from setuptools import setup, find_packages

setup(
    name="short_llm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "sentencepiece>=0.1.99",
        "accelerate>=0.20.0",
        "numpy>=1.24.0",
        "fastapi>=0.95.0",
        "uvicorn>=0.21.0",
        "python-multipart>=0.0.6",
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
        "python-dotenv>=1.0.0",
        "jinja2>=3.1.2",
        "aiofiles>=23.1.0"
    ],
    python_requires=">=3.8",
)
