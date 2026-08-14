<h1 align="center">[TMLR 26] EdgeCrafter: Compact ViTs for Edge Dense Prediction via
Task-Specialized Distillation</h1>

<h3 align="center">
  <b>English</b> | <a href="README_zh.md">简体中文</a>
</h3>

<p align="center">
  <a href="https://intellindust-ai-lab.github.io/projects/EdgeCrafter/"><img src="https://img.shields.io/badge/Webpage-EdgeCrafter-blue.svg" alt="Webpage"></a>
  <a href="https://arxiv.org/abs/2603.18739"><img src="https://img.shields.io/badge/arXiv-EdgeCrafter-orange.svg" alt="arXiv"></a>
  <a href="#"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>

</p>

<p align="center">
        <a href="https://capsule2077.github.io/">Longfei Liu <sup>*</sup><sup>‡</sup></a>&nbsp;
        <a >Yongjie Hou <sup>*</sup></a>&nbsp;
        <a href='https://liyangggggg.github.io/LIYangggggg/'>Yang Li <sup>*</sup></a>&nbsp;
        <a href='https://qiruiwang0728.github.io/homepage/'>Qirui Wang <sup>*</sup></a>&nbsp;
        <a >Youyang Sha</a>&nbsp; </br>
        <a >Yongjun Yu</a>&nbsp;
        <a >Yinzhi Wang</a>&nbsp;
        <a >Peizhe Ru</a>&nbsp;
        <a href="https://xuanlong-yu.github.io/">Xuanlong Yu<sup>†</sup></a>&nbsp
        <a href="https://xishen0220.github.io/">Xi Shen <sup>†</sup></a> <br><br>
      <a> * Equal Contribution &nbsp;&nbsp; ‡ Project Lead &nbsp;&nbsp; † Corresponding Author</a> <br>
</p>

<p align="center">
    <sup></sup> <a href="https://intellindust-ai-lab.github.io">Intellindust AI Lab</a> <br> 
</p>

<p align="center" style="margin:0; padding:0;">
  <img src=".github/teaser.png">
</p>


---

## 🚀 Updates
- **[2026-08-14]** We released the Objects365 pretrained checkpoints. Enjoy it!
- **[2026-08-13]** EdgeCrafter has been integrated into [Intel® Geti™](https://github.com/open-edge-platform/geti): fine-tune ECDet-S/M/L/X on your own data with no code, train on Intel dGPU / iGPU / CPU, export to OpenVINO™ IR and optimized for INT8 for edge deployment.
- **[2026-08-11]** EdgeCrafter has been accepted for publication in **TMLR 2026** 🎉 
- **[2026-07-23]** EdgeCrafter has been integrated into [LightlyTrain](https://docs.lightly.ai/train/stable/pretrain_distill/models/edgecrafter.html)
- **[2026-07-22]** EdgeCrafter has been integrated into [LibreYOLO](https://www.libreyolo.com/zh).
- **[2026-04-20]** Our previous version [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) was used by two winning teams at the [CVPR 2026 Maritime Computer Vision Workshop](https://arxiv.org/abs/2604.13244), taking home 2nd place in Thermal Object Detection Challenge and 3rd place in Vision-to-Chart Data Association Challenge.
- **[2026-03-21]** <a href="https://huggingface.co/Intellindust">We have uploaded our models on 🤗 Hugging Face</a>.
- **[2026-03-19]** Initial release of EdgeCrafter.

---

## 🤗 Hugging Face

<a href="https://huggingface.co/Intellindust">We have uploaded our models on 🤗 Hugging Face</a>! You can also access these models via [hf_models.ipynb](./hf_models.ipynb). Have a try!

---

## 📍 Reproducing the Results

- **Detection & Instance Segmentation:** [Instructions](./ecdetseg)
- **Pose Estimation:** [Instructions](./ecpose)

---

## 🏆 Model Zoo

### COCO2017 Validation Results

> **Note**: Latency is measured on an NVIDIA T4 GPU with batch size 1 under FP16 precision using TensorRT (v10.6).

> `--` denotes COCO-only training, and `O365` denotes additional Objects365 pretraining. For ECSeg and ECPose, the `O365` results are obtained by directly transferring the detection model pretrained on Objects365.

### Object Detection

| Model | Extra Sup. | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----------:|:----:|:------------------:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECDet-S** | -- | 640 | 51.7 | 10 | 26 | 5.41 | [config](ecdetseg/configs/ecdet/ecdet_s.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_s.pth) |
| **ECDet-S** | O365 | 640 | 53.6 | 10 | 26 | 5.41 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_s_o3652coco.pth) |
| **ECDet-M** | -- | 640 | 54.3 | 19 | 53 | 7.98 | [config](ecdetseg/configs/ecdet/ecdet_m.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_m.pth) |
| **ECDet-M** | O365 | 640 | 56.7 | 19 | 53 | 7.98 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_m_o3652coco.pth) |
| **ECDet-L** | -- | 640 | 57.0 | 33 | 101 | 10.49 | [config](ecdetseg/configs/ecdet/ecdet_l.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth) |
| **ECDet-L** | O365 | 640 | 59.0 | 33 | 101 | 10.49 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_l_o3652coco.pth) |
| **ECDet-X** | -- | 640 | 57.9 | 49 | 151 | 12.70 | [config](ecdetseg/configs/ecdet/ecdet_x.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecdet_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_x.pth) |
| **ECDet-X** | O365 | 640 | 59.9 | 49 | 151 | 12.70 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_x_o3652coco.pth) |

### Objects365 Pretrained Checkpoints

| Model | Checkpoint |
|:-----:|:----------:|
| **ECDet-S** | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_s_o365.pth) |
| **ECDet-M** | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_m_o365.pth) |
| **ECDet-L** | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_l_o365.pth) |
| **ECDet-X** | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecdet_x_o365.pth) |

### Instance Segmentation

| Model | Extra Sup. | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----------:|:----:|:------------------:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECSeg-S** | -- | 640 | 43.0 | 10 | 33 | 6.96 | [config](ecdetseg/configs/ecseg/ecseg_s.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_s.pth) |
| **ECSeg-S** | O365 | 640 | 43.9 | 10 | 33 | 6.96 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecseg_s_o3652coco.pth) |
| **ECSeg-M** | -- | 640 | 45.2 | 20 | 64 | 9.85 | [config](ecdetseg/configs/ecseg/ecseg_m.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_m.pth) |
| **ECSeg-M** | O365 | 640 | 46.9 | 20 | 64 | 9.85 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecseg_m_o3652coco.pth) |
| **ECSeg-L** | -- | 640 | 47.1 | 34 | 111 | 12.56 | [config](ecdetseg/configs/ecseg/ecseg_l.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_l.pth) |
| **ECSeg-L** | O365 | 640 | 48.8 | 34 | 111 | 12.56 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecseg_l_o3652coco.pth) |
| **ECSeg-X** | -- | 640 | 48.4 | 50 | 168 | 14.96 | [config](ecdetseg/configs/ecseg/ecseg_x.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecseg_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecseg_x.pth) |
| **ECSeg-X** | O365 | 640 | 49.8 | 50 | 168 | 14.96 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecseg_x_o3652coco.pth) |

### Pose Estimation

| Model | Extra Sup. | Size | AP<sub>50:95</sub> | #Params | GFLOPs | Latency (ms) | Config | Log | Checkpoint |
|:-----:|:----------:|:----:|:------------------:|:-------:|:------:|:------------:|:------:|:---:|:----------:|
| **ECPose-S** | -- | 640 | 68.9 | 10 | 30 | 5.54 | [config](ecpose/configs/ecpose/ecpose_s_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_s.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_s.pth) |
| **ECPose-S** | O365 | 640 | 69.7 | 10 | 30 | 5.54 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecpose_s_o3652coco.pth) |
| **ECPose-M** | -- | 640 | 72.4 | 20 | 63 | 9.25 | [config](ecpose/configs/ecpose/ecpose_m_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_m.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_m.pth) |
| **ECPose-M** | O365 | 640 | 73.1 | 20 | 63 | 9.25 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecpose_m_o3652coco.pth) |
| **ECPose-L** | -- | 640 | 73.5 | 34 | 112 | 11.83 | [config](ecpose/configs/ecpose/ecpose_l_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_l.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_l.pth) |
| **ECPose-L** | O365 | 640 | 74.5 | 34 | 112 | 11.83 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecpose_l_o3652coco.pth) |
| **ECPose-X** | -- | 640 | 74.8 | 51 | 172 | 14.31 | [config](ecpose/configs/ecpose/ecpose_x_coco.yml) | [log](https://github.com/capsule2077/edgecrafter/raw/refs/heads/main/logs/ecpose_x.log) | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecpose_x.pth) |
| **ECPose-X** | O365 | 640 | 75.9 | 51 | 172 | 14.31 | - | - | [model](https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1_o365/ecpose_x_o3652coco.pth) |
---

## 📦 Installation

```bash
# Create conda environment
conda create -n ec python=3.11 -y
conda activate ec

# Install dependencies
pip install -r requirements.txt
```

### ⚡ Quick Start (Inference)
The easiest way to test EdgeCrafter is to run inference on a sample image using a pre-trained model.
```bash
# 1. Download a pre-trained model (e.g., ECDet-L)
cd ecdetseg
wget https://github.com/capsule2077/edgecrafter/releases/download/edgecrafterv1/ecdet_l.pth
# 2. Run PyTorch inference
# Make sure to replace `path/to/your/image.jpg` with an actual image path
python tools/inference/torch_inf.py -c configs/ecdet/ecdet_l.yml -r ecdet_l.pth -i path/to/your/image.jpg
```


## 📄 License

This project is released under the [Apache 2.0 License](./LICENSE).

---

## 🙏 Acknowledgements

We thank the authors of the following open-source projects that made this work possible: [RT-DETR](https://github.com/lyuwenyu/RT-DETR), [D-FINE](https://github.com/Peterande/D-FINE), [DEIM](https://github.com/Intellindust-AI-Lab/DEIM), [lightly-train](https://github.com/lightly-ai/lightly-train), [DETRPose](https://github.com/SebastianJanampa/DETRPose), [RF-DETR](https://github.com/roboflow/rf-detr), [DINOv3](https://github.com/facebookresearch/dinov3)

--- 

## 📚 Citation

If you find this project useful in your research, please consider citing:

```bibtex
@article{liu2026edgecrafter,
  title={EdgeCrafter: Compact ViTs for Edge Dense Prediction via Task-Specialized Distillation},
  author={Liu, Longfei and Hou, Yongjie and Li, Yang and Wang, Qirui and Sha, Youyang and Yu, Yongjun and Wang, Yinzhi and Ru, Peizhe and Yu, Xuanlong and Shen, Xi},
  journal={TMLR},
  year={2026}
}
```
