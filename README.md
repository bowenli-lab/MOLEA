# **MOLEA 🌿**

---
## **Prerequisites**

Before you begin, ensure you have **conda** installed on your system. The installation commands are tailored for a system with **CUDA 11.3**.

---
## **Installation**

Follow these steps to set up your environment and install all necessary dependencies.

1.  **Clone the Repository**
    
    ```bash
    git clone [https://github.com/bowenli-lab/MOLEA.git](https://github.com/bowenli-lab/MOLEA.git)
    cd MOLEA
    ```
    
2.  **Create and Activate the Conda Environment**
    
    ```bash
    conda create --name molea python=3.9 -y
    conda activate molea
    ```
    
3.  **Install Dependencies**
    
    First, install the specific PyTorch and PyTorch Geometric versions required.
    
    ```bash
    # Install PyTorch
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113  --extra-index-url https://download.pytorch.org/whl/cu113
    
    # Install PyTorch Geometric and its dependencies
    pip install torch-geometric==2.2.0 torch-sparse==0.6.16 torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
    ```
    
    Finally, install the remaining packages from `requirements.txt`.
    
    ```bash
    pip install -r requirements.txt
    ```
    
---
## **Usage**

The core workflow consists of pre-training, fine-tuning, and inference.

### **1. Dataset**

The datasets for both pre-training and fine-tuning stages are provided within the `./data` directory of this repository.

### **2. Pre-training**

A model checkpoint pre-trained on a 15M virtual library is already included in the `./ckpt` folder. This step is only necessary if you want to pre-train the model on your own dataset.

* **Note:** For a warm start, first download the pre-trained model weights from [MolCLR](https://github.com/yuyangw/MolCLR).
* Modify the `config_pretrain.yaml` file to match your dataset and requirements.

To start pre-training, run the following command:
```bash
python pretrain.py config_pretrain.yaml
```


### **3. Fine-tuning**

To fine-tune the model for a specific downstream task:

Modify the config_finetune.yaml file as needed.

To start fine-tuning, run the command:

```bash
python finetune.py config_finetune.yaml
```

### **4. Inference**

To run inference and generate predictions with your fine-tuned model, use the command below. The script also includes built-in visualization tools.

```bash
python infer.py <path_to_your_finetuned_model_folder>
```

## **5. Acknowledgement**
This project builds upon the work of these excellent open-source projects:
- **MolCLR**: [https://www.nature.com/articles/s42256-022-00447-x](https://www.nature.com/articles/s42256-022-00447-x)
- **Mordred**: [https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-018-0258-y)
- **AGILE**: [https://www.nature.com/articles/s41467-024-50619-z](https://www.nature.com/articles/s41467-024-50619-z)