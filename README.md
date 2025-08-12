# Diffusion Model Examples

This repository contains several examples that demonstrate the core concepts of diffusion models in a simple, visual manner.

## 🎯 Overview

This repository contains a minimal implementation of a diffusion model that:
- Trains on several examples including 2D circle
- Learns to denoise data step by step
- Visualizes the entire forward and reverse diffusion process
- Provides debugging tools to understand model behavior

## 📊 What You'll See

### Forward Process (Adding Noise)
```
Clean Circle → Slightly Noisy → More Noise → ... → Pure Gaussian Noise
```

### Reverse Process (Denoising)
```
Pure Noise → Structured Noise → Emerging Pattern → ... → Clean Circle
```

## 🛠️ Requirements

```bash
pip install torch matplotlib numpy
```

## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/jkw0701/diffusion-examples.git
cd diffusion-examples
```

2. **Run the example**
```bash
python example-circle.py
```

3. **Optional: Check model accuracy**
When prompted, type `y` to see detailed model prediction analysis.

## 📚 Further Reading

- [Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)
- [Understanding Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)

## 🤝 Contributing

Contributions are welcome!

## 📄 License

This project is licensed under the MIT License

## 📧 Contact

If you have questions or suggestions, please open an issue or reach out!
