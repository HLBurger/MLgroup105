# MLgroup105

## Datasets

<!-- **FairFace Dataset:**   -->
<!-- [FairFace on Hugging Face](https://huggingface.co/datasets/HuggingFaceM4/FairFace)   -->

**UTK Dataset:**  
[UTK on Kaggle](https://www.kaggle.com/datasets/roshan81/ageutk)  

## How to Use the Kaggle API
```python
import ipywidgets
import kagglehub

kagglehub.login()
```

Fill in your Kaggle username and API token.  
You can generate a new token by:

1. Going to Kaggle.
2. Navigating to **Settings**.
3. Clicking **Create new token** under API.

---

# Running an Experiment in MLGroup105

This guide explains how to set up and execute an experiment copying the `Experiment Hidde.ipynb` Jupyter notebook. You will learn how to modify different aspects of the experiment, such as changing the model architecture, adjusting hyperparameters, and testing different dataset splits.

## 1️ Prerequisites
Before running an experiment, ensure that you have the following dependencies installed:

### **Install Dependencies**
If you haven't installed the required packages, run:
```bash
pip install -r requirements.txt
```

Make sure you have Jupyter Notebook installed:
```bash
pip install notebook
```

### 1 **Ensure Correct Project Structure**
Your project directory should look like this:

```
MLGroup105/
│── experiments/
│   ├── Experiment Hidde.ipynb  # Jupyter Notebook you can copy and modify
│   ├── experiment_x.ipynb  # Jupyter Notebook for a specific experiment# Another experiment
│── src/
│   ├── data.py        # Manages dataset processing, including loading and transforming images.
│   ├── model.py       # Defines the convolutional neural network (CNN) architecture.
│   ├── train.py       # Handles model training, including data loading, optimization, and logging.
│   ├── evaluate.py    # Contains the script for evaluating the trained CNN model on the test dataset.
│   ├── utils.py       # Includes helper functions such as visualization and metric calculations.
│   ├── config.py      # Handles configuration settings such as device setup (CPU/GPU).
│── main.py           # Main script to run training and evaluation
│── requirements.txt  # Dependencies
│── README.md         # Documentation
│── .gitignore        # Excludes unnecessary files from GitHub
```


## 2️ Running an Experiment
To execute an experiment:

1. **Navigate to the `experiments/` directory**
   ```bash
   cd experiments
   ```

2. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

3. **Copy `Experiment Hidde.ipynb`**
   - Change name to `experiment_x.ipynb` for a specific experiment 

4. **Run the cells sequentially** to:
   - Load the dataset
   - Initialize the model
   - Train the model
   - Evaluate performance

## 3️ Customizing an Experiment
You can modify several parameters to test different configurations.

### **🔹 Changing Hyperparameters**
Modify the following parameters in the notebook:
```python
EPOCHS = 10  # Change number of training epochs
BATCH_SIZE = 16  # Adjust batch size
LEARNING_RATE = 0.001  # Modify learning rate
```

### **🔹 Changing the Model Architecture**
Modify `model.py` to change the CNN architecture. Example:
```python
class CustomCNN(nn.Module):
    def __init__(self, num_age_bins):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # New Layer
        self.fc1 = nn.Linear(256 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, num_age_bins)
```
To use this new model in the experiment:
```python
CNN_model = CustomCNN(num_age_bins=len(age_idx)).to(device)
```

### **🔹 Changing Data Augmentation**
Modify `train_transform` in the notebook to apply different augmentation techniques:
```python
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  # Change rotation angle
    transforms.ColorJitter(brightness=0.3, contrast=0.3),  # Modify brightness & contrast
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### **🔹 Using a Different Dataset Split**
Change the train-validation-test split in the notebook:
```python
train_loader, val_loader, test_loader = dataset_manager.get_data_loaders(
    train_size=0.6,  # Reduce training size
    val_size=0.2,  # Increase validation size
    test_size=0.2,  # Increase test set size
    batch_size=BATCH_SIZE,
    train_transform=train_transform,
    test_transform=test_transform
)
```

## 4️⃣ Saving and Logging Results
- **Document the experiment in Markdown**
  After running an experiment, write a structured report in Markdown at the end of the notebook. Include:
  ```markdown
  ## Experiment Documentation
  
  ### Hyperparameters:
  - Learning Rate: 0.001
  - Epochs: 10
  - Batch Size: 16
  
  ### Changes Tested:
  - Applied data augmentation (rotation, color jittering)
  - Used CrossEntropyLoss with class weights
  - Increased training dataset size
  
  ### Results:
  - Final Test Accuracy: 0.76%
  - Observed better performance with weighted loss
  - "Check ID" class is still underrepresented in classification report
  - Experiment showed improvement over previous model version
  ```
  
  This will help in tracking experiments and understanding what works best.


- **Save the Model**
To save (model/) experiment results:
- **Save model weights:**
  ```python
  torch.save(CNN_model.state_dict(), "experiment_model.pth")
  ```
- **Log results to a file:**
  ```python
  with open("experiment_results.txt", "w") as f:
      f.write(str(model_results))
  ```





