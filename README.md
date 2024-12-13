<h1 align="center">Exposure Correction</h1>
<div align="center">
  <h4>Restore over-exposure images based on Zero-DCE</h4>
</div>
<p align="center">
  <a href="#related-works">Related Works</a>&nbsp;&nbsp;•&nbsp;
  <a href="#contributors">Contributors</a>&nbsp;&nbsp;•&nbsp;
  <a href="#pipeline">Pipeline</a>&nbsp;&nbsp;•&nbsp;
  <a href="#environment">Environment</a>&nbsp;&nbsp;•&nbsp;
  <a href="#installation">Installation</a>&nbsp;&nbsp;•&nbsp;
</p>

> [!NOTE]
> 
> Exposure Correction  is used to adjust the brightness of photos or images to achieve an ideal exposure level, correcting detail loss caused by overexposure (too bright) or underexposure (too dark), thereby enhancing image quality and visual effects.  
> Our work combines the methods from three papers, modifying the network framework to achieve effective exposure correction.

## Related Works
- [Zero DCE, CVPR 2020](https://github.com/Li-Chongyi/Zero-DCE)
- [Learning Multi-Scale Photo Exposure Correction, CVPR 2021](https://github.com/mahmoudnafifi/Exposure_Correction)
- [Reversed and Fused Zero-DCE](https://ieeexplore.ieee.org/document/10604009)

## Contributors
 <a href="https://github.com/ChuEating1005/Exposure-Correction/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ChuEating1005/Exposure-Correction" />
</a>

## Pipeline
<div align="center">
  <img src="./docs/pipeline.png">
</div> 

Our pipeline consists of three main components:
1. **Multi-scale Processing**: Input image is decomposed into n-level Laplacian pyramid
2. **Progressive Enhancement**:
   - Each level is processed by a dedicated DCE-Net
   - Forward enhancement: Apply curve parameters A
   - Backward reasoning: Apply inverse parameters -A
   - Add Laplacian details to enhanced result
3. **Cascading Structure**: Output from each level serves as input to the next level

## Environment
> [!CAUTION]
> **We are using the following environment to develop this project.**
> - NVIDIA GeForce RTX 4090
> - Ubuntu 24.04 LTS
> - python = 3.7
> - pytorch = 1.13.1
> - torchvision = 0.14.1
> - torchaudio = 0.13.1
> - Cuda = 11.7
> - OpenCV

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ChuEating1005/Exposure-Correction
   cd Exposure-Correction
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
Zero-DCE/
├── data/               # Dataset directory
├── models/            # Model architectures
├── train/             # Training scripts
├── test/              # Testing scripts
├── utils/             # Utility functions
└── snapshots/         # Model weights
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
