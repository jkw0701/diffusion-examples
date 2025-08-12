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

## 📈 Results 
### Forward Process 
<img width="900" height="450" alt="Image" src="https://github.com/user-attachments/assets/970b13a9-4844-40b2-b924-c366b1cdb37f" />

### Reverse Process (Denoising process)
<img width="900" height="450" alt="Image" src="https://github.com/user-attachments/assets/93285a37-5c6d-42fd-87eb-cee1f730a370" />

### Comparison between original image and generated image
<img width="900" height="300" alt="Image" src="https://github.com/user-attachments/assets/d2168026-c130-4785-8584-a3b1ce0bdeb7" />   

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
