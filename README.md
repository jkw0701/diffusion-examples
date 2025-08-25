# Diffusion Model Examples Collection

This repository contains a comprehensive collection of step-by-step examples to learn diffusion models, from basic 2D pattern generation to real robot navigation applications.

## 🎯 Overview

We provide 6 progressive examples covering various applications of diffusion models, from fundamental 2D pattern generation to practical robot navigation systems.

## 📋 Example List

### 🔵 example-circle.py - Basic Diffusion Model

- Purpose: 2D circle pattern generation
- Key Concepts: DDPM fundamentals, forward/reverse process
- Features:
1) Simple U-Net architecture
2) Noise scheduling
3) Visual denoising process visualization


#### Forward Process (Adding Noise)
```
Clean Circle → Slightly Noisy → More Noise → ... → Pure Gaussian Noise
```
<img width="900" height="450" alt="Image" src="https://github.com/user-attachments/assets/970b13a9-4844-40b2-b924-c366b1cdb37f" />


#### Reverse Process (Denoising)
```
Pure Noise → Structured Noise → Emerging Pattern → ... → Clean Circle
```
<img width="900" height="450" alt="Image" src="https://github.com/user-attachments/assets/93285a37-5c6d-42fd-87eb-cee1f730a370" />

#### Comparison between original image and generated image
<img width="900" height="300" alt="Image" src="https://github.com/user-attachments/assets/d2168026-c130-4785-8584-a3b1ce0bdeb7" />   


### 🎮 example-vector.py - 2D Direction Vector Prediction

- Purpose: Learning direction vectors between current position and goal
- Key Concepts: Vector regression, HuggingFace Diffusers
- Features:
1) Direction = Goal - Current position
2) Normalized direction vectors
3) Interactive testing system

### 🗺️ example-vector2.py - Obstacle Avoidance Navigation

- Purpose: Smart direction prediction considering obstacles
- Key Concepts: A* path planning, multi-modal fusion
- Features:
1) A* algorithm-based optimal paths
2) CNN + MLP hybrid model
3) Collision avoidance loss function

### 🏞️ example-vector3.py - Real-world Environment Navigation

- Purpose: RGB image-based real environment navigation
- Key Concepts: Semantic segmentation, ResNet backbone
- Features:
1) Indoor/outdoor environment simulation
2) Joint semantic segmentation and navigation learning
3) Confidence prediction

### 🦾 example-robot-arm.py - Robot Arm Trajectory Generation

- Purpose: Joint space trajectory generation for robot arms
- Key Concepts: Inverse kinematics, interactive demo
- Features:
1) Forward/inverse kinematics computation
2) Real-time interactive simulation
3) DDPM vs DDIM comparison

#### Result
<img width="900" height="450" alt="Image" src="https://github.com/user-attachments/assets/9266d20b-3f39-4055-9e22-7dd9d980ba95" />


## 🛠️ Requirements
### Basic Requirements
```bash
pip install torch torchvision matplotlib numpy pillow
```
### HuggingFace Diffusers (for some examples)
```bash
pip install diffusers transformers accelerate
```
### Optional Requirements (for GUI examples)
```bash
pip install opencv-python PyQt5
```


## 🚀 Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/jkw0701/diffusion-examples.git
cd diffusion-examples
```

2. **Run the example**
```bash
# Simple circle generation
python example-circle.py

# Trajectory pattern generation
python example-traj.py

# Direction vector learning
python example-vector.py
```

```bash
# Obstacle avoidance navigation
python example-vector2.py

# Real environment simulation
python example-vector3.py

# Robot arm simulation
python example-robot-arm.py --mode train  # Training
python example-robot-arm.py --mode demo   # Demo
```


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
