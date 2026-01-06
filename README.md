# EPINET: Fully-Convolutional Neural Network for Depth from Light Field Images (TF2 / Keras 3)

**An updated and refactored implementation of EPINET using TensorFlow 2.x and Python 3.13.**

This repository contains a modern implementation of the paper:
> **EPINET: A Fully-Convolutional Neural Network using Epipolar Geometry for Depth from Light Field Images**
> Changha Shin, Hae-Gon Jeon, Youngjin Yoon, In So Kweon and Seon Joo Kim
> *IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018*
> [Paper Link](https://arxiv.org/pdf/1804.02379.pdf)

## Key Improvements in This Version

Unlike the original 2018 codebase, this implementation features:
* **Modern Stack:** Ported to **Python 3.13**, **TensorFlow 2.x**, and **Keras 3**.
* **Dynamic Input Shapes:** Fixed hardcoded dimensions to support rectangular Light Field images (e.g., Lytro dataset) alongside synthetic square ones.
* **Optimized Training:** Adjusted hyperparameters and learning rates to achieve convergence in just **3 epochs** (vs. original long training times).
* **Interactive Testing:** Added `EPINET_test.ipynb` for easy visualization of disparity maps using Jupyter Notebooks.
* **Refactored Codebase:** Modularized code structure following modern PEP 8 standards.

## 🛠️ Environment Setup

This project uses `uv` for fast and reliable dependency management.

1.  **Prerequisites:** Ensure you have Python 3.13 installed.
2.  **Install dependencies:**
    ```bash
    # Creates virtual environment and installs requirements
    uv sync
    ```

## 📂 Dataset Preparation

1.  Download the **HCI Light Field Dataset** from the [official website](https://lightfield-analysis.uni-konstanz.de/).
2.  Extract the dataset and organize the folders as follows inside the project root:

```text
project_root/
├── data/
│   └── hci_dataset/
│       ├── additional/
│       ├── training/
│       ├── test/
│       └── stratified/
│   └── lytro/
├── models/
├── results/
└── ...
```

## Training the Model

To train the model from scratch using the optimized settings:

```bash
uv run EPINET_train.py
```
Checkpoints: Saved automatically in `./models/checkpoints/`.

Logs: Training progress and MSE scores are logged to console and text file `./models/If_EPINET_train.txt`.

```text
Note: The training is set to run for 3 epochs, which is sufficient to reach an MSE of ~9.7 and Loss < 1.0.
```

## Testing & Visualization

### Option A: Jupyter Notebook (Recommended)

For an interactive experience where you can select specific images and visualize the output PFM files immediately:

1. Open the notebook `EPINET_test.ipynb` in Jupyter.

2. Configure the MODEL_WEIGHTS_PATH and INPUT_DIR variables in the first cell.

3. Run all cells to generate and view the disparity map.

### Option B: Python Script

To process a batch of images via terminal:

Edit `EPINET_test.py` to point to your specific checkpoint weight file:

```python
path_weight = 'epinet_checkpoints/EPINET_train_ckp/iter0002_trainmse9.728_bp38.80.keras'
```

Run the script:
```bash
uv run EPINET_test.py
```

Results are saved as .pfm files in `./results/`.

## References

If you use this code, please cite the original authors and the relevant literature:

> **EPINET: A Fully-Convolutional Neural Network using Epipolar Geometry for Depth from Light Field Images**
> Changha Shin, Hae-Gon Jeon, Youngjin Yoon, In So Kweon and Seon Joo Kim
> *IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018*
> [Paper Link](https://arxiv.org/pdf/1804.02379.pdf)

Original Author Contact (2018): changhashin@yonsei.ac.kr
