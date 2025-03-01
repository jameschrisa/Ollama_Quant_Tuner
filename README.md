# Ollama Quantized Performance Analyis Manager

A beautiful terminal application for managing Ollama models, parameters, and memory usage.

## Features

- 🎨 Beautiful command-line UI using Rich for Python
- 🔍 Search for Ollama models and get detailed information
- ⚙️ Easy configuration of model parameters:
  - Quantization levels (Q4_0, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
  - Flash Attention toggle
  - KV Cache type selection (auto, f16, f32)
  - Context window size adjustment
- 📊 Memory usage visualization with before/after comparison
- 💻 System analysis and parameter recommendations based on hardware
- 📝 Comprehensive error handling and logging
- 🚀 Streamlined workflow for optimizing model performance

## Requirements

- Python 3.7+
- Ollama installed and running on your system
- Access to Terminal/Command Prompt

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ollama-manager.git
   cd ollama-manager
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Make sure Ollama is installed on your system:
   - Follow the installation instructions at [Ollama's official website](https://ollama.com/download)

4. Start the Ollama service before running this tool.

## Usage

### Interactive Mode

```bash
python ollama_manager.py
```

This launches the interactive menu where you can:

1. **Check System Information**: View your hardware specs and get recommendations
2. **View Local Models**: See all Ollama models installed on your system
3. **Search Ollama Library**: Find models from Ollama's online repository
4. **Pull a Model**: Download new models to your system
5. **Configure and Run a Model**: Customize parameters and test with before/after memory analysis

### Example Workflow

1. Start by checking your system information to understand your hardware capabilities
2. Search for an appropriate model based on your system's constraints
3. Pull the model you want to optimize
4. Configure parameters based on the recommendations
5. Run a test prompt to see memory impact
6. Adjust parameters as needed for your specific use case

## Understanding Ollama Parameters

### Quantization Levels

Quantization reduces model size and memory requirements at the cost of some accuracy:

- **Q4_0**: 4-bit quantization with no K-means clustering. Smallest file size, fastest loading, lower quality.
- **Q4_K_M**: 4-bit quantization with K-means clustering and medium quality. Good balance for low-memory systems.
- **Q5_K_M**: 5-bit quantization with K-means clustering. Better quality than Q4 variants while still saving memory.
- **Q6_K**: 6-bit quantization with K-means clustering. Good quality with moderate memory savings.
- **Q8_0**: 8-bit quantization, highest quality among quantized models but largest size.

### Flash Attention

Flash Attention is an optimization that can dramatically speed up transformer models on compatible GPUs. Enable this if you have a GPU with sufficient memory.

### KV Cache Types

- **auto**: Let Ollama decide based on your system
- **f16**: Half precision (16-bit) cache - faster, uses less memory, slightly less accurate
- **f32**: Full precision (32-bit) cache - more accurate but uses more memory

### Context Size

The context window determines how much text the model can "remember" at once. Larger values allow processing longer documents but require more memory. Common values are:

- 2048: Minimal context for basic conversations
- 4096: Standard for most models
- 8192: Extended context for longer conversations
- 16384+: Large context for document analysis (requires significant memory)

## Tips for Optimizing Memory Usage

1. Start with higher quantization (Q4_K_M) if memory is limited
2. Reduce context size if you don't need long-context conversations
3. Use F16 KV cache when working with GPUs
4. Monitor memory usage differences to find the optimal balance for your system

## Troubleshooting

- **"Ollama service is not running"**: Make sure to start the Ollama service before using this tool
- **Out of memory errors**: Try a smaller model or higher quantization level
- **Slow performance**: Enable Flash Attention if you have a compatible GPU
- **Model not found**: Check your spelling or search for available models

## License

MIT

## Acknowledgments

- [Ollama](https://ollama.com/) for making local LLM deployment accessible
- [Rich](https://github.com/Textualize/rich) for the beautiful terminal interface
- All the open-source quantized LLM models
