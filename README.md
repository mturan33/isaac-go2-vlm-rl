# 🤖 Go2 VLM-RL Navigation

Language-conditioned quadruped navigation using Vision-Language Models (VLM) and Reinforcement Learning.

## 📋 Overview

This project implements a hierarchical VLM-RL system for Unitree Go2 robot navigation:

```
"Mavi sandalyeye git" (Go to the blue chair)
         │
    ┌────▼────┐
    │   VLM   │ ← RGB Image (640×480)
    │ Phi-3-V │ → Target: {x, y, confidence}
    └────┬────┘
         │ Every ~1 second
    ┌────▼────────┐
    │ Nav Control │ → cmd_vel: [vx, vy, vyaw]
    └────┬────────┘
         │ Every 20ms
    ┌────▼────────┐
    │ Student RL  │ ← Depth (64×64) + Proprio (48)
    │   Policy    │ → Joint Actions (12 DoF)
    └────┬────────┘
         │
      [GO2 🐕]
```

## 🛠️ Requirements

- Python 3.10+
- PyTorch 2.0+
- Transformers 4.40+
- NVIDIA GPU with 12GB+ VRAM
- Isaac Lab (for simulation)

## 📦 Installation

```powershell
# Install dependencies
pip install transformers accelerate pillow

# Clone/copy files to Isaac Lab
# C:\IsaacLab\source\isaaclab_tasks\isaaclab_tasks\direct\go2_vlm_rl\
```

## 🚀 Quick Start

### 1. Test Scene Generator
```powershell
python scene_generator.py
```

### 2. Test VLM (requires ~8GB VRAM)
```powershell
python vlm_wrapper.py test_scene.png "mavi sandalyeye git"
```

### 3. Full Demo
```powershell
python run_vlm_nav.py --test-vlm --image test_scene.png
python run_vlm_nav.py --test-scene
```

## 📁 File Structure

```
go2_vlm_rl/
├── vlm_wrapper.py          # Phi-3-Vision wrapper (Windows compatible)
├── scene_generator.py      # Random colored objects spawner
├── run_vlm_nav.py         # Full pipeline demo
├── test_scene.png         # Test image
└── README.md              # This file
```

## 🎨 Supported Objects & Colors

**Objects:** chair, table, cabinet, sofa, box, ball, cylinder, cone

**Colors:** blue, red, green, yellow, orange, purple, white, pink, brown, cyan

**Languages:** Turkish and English commands supported!

| Turkish | English |
|---------|---------|
| mavi sandalyeye git | go to the blue chair |
| kırmızı masayı bul | find the red table |
| yeşil kutuya git | navigate to green box |

## ⚙️ Configuration

### VLM Settings
- Model: `microsoft/Phi-3-vision-128k-instruct`
- VRAM: ~8-10GB
- Inference: ~100-200ms per call
- Attention: SDPA (Windows compatible, no Flash Attention needed)

### Navigation Controller
- Max linear velocity: 1.0 m/s
- Max angular velocity: 1.0 rad/s
- Goal threshold: 0.15m

## 🔧 Troubleshooting

### Flash Attention Error
The code uses `_attn_implementation="eager"` which works on Windows without Flash Attention.

### VRAM Issues
If running out of memory:
1. Images are automatically resized to 768px max
2. Use `low_memory=True` in VLMWrapper
3. Close other GPU applications

### Model Download
First run downloads ~8GB model. This is cached in `~/.cache/huggingface/`.

## 📊 Architecture

### High-Level (VLM)
- **Input:** RGB image + text command
- **Output:** Target coordinates (x, y) + confidence
- **Frequency:** Every 20-30 simulation steps (~1 second)

### Low-Level (RL Policy)
- **Input:** Depth image (64×64) + proprioception (48) + cmd_vel (3)
- **Output:** Joint positions (12 DoF)
- **Frequency:** Every simulation step (50Hz)

## 🔮 Roadmap

- [x] VLM wrapper (Phi-3-Vision)
- [x] Scene generator with colored objects
- [x] Navigation controller
- [ ] Isaac Lab environment integration
- [ ] Full pipeline testing
- [ ] Real robot deployment

## 📚 References

- [Berkeley Quadruped LLM](https://arxiv.org/abs/2404.05291) - ICRA 2024
- [Phi-3-Vision](https://huggingface.co/microsoft/Phi-3-vision-128k-instruct)
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/)

## 📝 License

MIT License

## 👤 Author

Mehmet Turan Yardımcı
- GitHub: [@mturan33](https://github.com/mturan33)
- LinkedIn: [/in/mehmetturanyardimci](https://linkedin.com/in/mehmetturanyardimci)

# Kendi projenin LICENSE veya README'sine ekle:

## Acknowledgements & Third-Party Code
This project includes code from:
- unitree_sim_isaaclab (https://github.com/unitreerobotics/unitree_sim_isaaclab)
  Copyright 2025 HangZhou YuShu TECHNOLOGY CO.,LTD. ("Unitree Robotics")
  Licensed under Apache License 2.0


This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:

1. https://github.com/isaac-sim/IsaacLab
2. https://github.com/isaac-sim/IsaacSim
3. https://github.com/zeromq/pyzmq
4. https://github.com/unitreerobotics/unitree_sdk2_python
5. https://github.com/unitreerobotics/unitree_sim_isaaclab

------------------
