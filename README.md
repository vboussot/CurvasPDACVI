[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-CURVASPDACVI-blue)](https://huggingface.co/VBoussot/CurvasPDACVI)

# CURVAS2025: Bayesian Modeling of Inter-Annotator Variability in UNet-Based PDAC Tumor Segmentation (3rd place 🥈)

This repository provides the code, configurations, and models used for our submission to the **CURVAS 2025 Challenge**, focused on **uncertainty-aware segmentation** of **Pancreatic Ductal Adenocarcinoma (PDAC)** from CT scans.  
Our approach extends **pre-trained nnU-Net models** with **Adaptable Bayesian Neural Network (ABNN)** layers to model **inter-annotator variability** and generate **voxel-wise uncertainty maps**.

---

## 🏁 Official Challenge Results

Our method ranked **3rd overall** in the CURVAS2025 Challenge 🏅

| Rank | Team / Algorithm             | DSC (↑) | thresh-DSC (↑) | ECE (↓) | CRPS (↓) |
|------|------------------------------|---------|----------------|---------|----------|
| 🏅 1st | Tryzis    | 0.5576  | 0.5693         | 0.0296  | **5924** |
| 🥇 2nd | SHUGOSHA    | 0.5894  | 0.5801         | 0.0305  | 10792    |
| 🥈 3rd | **ours** | **0.7104** | **0.6401** | **0.0257** | 7320     |

✨ Highlights of our approach:
- 🥇 **Best Dice Score (DSC)**
- 🥇 **Best thresh-DSC**
- 🥇 **Best Calibration (ECE)**
- Robust and efficient Bayesian uncertainty estimation

🔗 [View the full CURVASPDACVI leaderboard](https://curvas-pdacvi.grand-challenge.org/evaluation/testing-phase/leaderboard/)

---

## 🚀 Inference Instructions

### 1. Download the models

Automatically download all required models from Hugging Face:

```bash
python download.py
```

This will create the following directory and files:

```
./resources/model/
├── FT_0.pt
├── FT_1.pt
├── FT_2.pt
├── FT_3.pt
├── FT_4.pt
└── M291.pt
```

- `FT_0.pt` – `FT_4.pt`: fine-tuned Bayesian checkpoints  
- `M291.pt`: auxiliary pancreas segmentation model used for ROI extraction  

🔗 [Model on Hugging Face](https://huggingface.co/VBoussot/Curvas2025)

---

### 2. Prepare the dataset

Expected structure:

```
./Dataset/
├── Case_001/
│   ├── IMAGE.mha   # input CT volume
├── Case_002/
│   ├── IMAGE.mha
└── ...
```

### 3. Run inference with uncertainty estimation

Run pancreas localization (ROI extraction) with the auxiliary model:

```bash
konfai PREDICTION -y \
  --gpu 0 \
  --num_workers 0 \
  --config ./resources/Prediction_TS.yml \
  --MODEL ./resources/model/M291.pt
```

Then run Bayesian inference with the fine-tuned models (ensemble of 5 checkpoints):

```bash
konfai PREDICTION -y \
  --gpu 0 \
  --num_workers 0 \
  --config ./resources/Prediction.yml \
  --MODEL ./resources/model/FT_0.pt:./resources/model/FT_1.pt:./resources/model/FT_2.pt:./resources/model/FT_3.pt:./resources/model/FT_4.pt
```

The predictions for each case are stored in:

```
./Predictions/Curvas/Dataset/Case_xxx/
├── Seg.mha   # Final Bayesian segmentation (after STAPLE aggregation)
└── Prob.mha  # Voxel-wise uncertainty map (values in [0, 1])
```


## 📚 References

- Franchi, G., Laurent, O., Leguéry, M., Bursuc, A., Pilzer, A., & Yao, A. (2023).  
  **Make Me a BNN: A Simple Strategy for Estimating Bayesian Uncertainty from Pre-trained Models**.  
  arXiv preprint [arXiv:2312.15297](https://arxiv.org/abs/2312.15297)

- Boussot, V., & Dillenseger, J.-L. (2025).  
  **KonfAI: A Modular and Fully Configurable Framework for Deep Learning in Medical Imaging**.  
  arXiv preprint [arXiv:2508.09823](https://arxiv.org/abs/2508.09823)
