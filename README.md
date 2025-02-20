# Turbofan-Engine-RUL-Prediction using Transformers

# Project Overview
This project focuses on predicting the Remaining Useful Life (RUL) of turbofan engines using a Transformer-based deep learning model. The CMAPSS dataset is used for training and evaluation, and the workflow involves data preprocessing, model training, and hyperparameter tuning with Optuna.

# Features
Data Loading & Preprocessing: Efficient handling of CMAPSS dataset, including feature engineering and normalization.
Transformer-based Model: A deep learning model utilizing self-attention mechanisms for sequence prediction.
Hyperparameter Optimization: Uses Optuna to optimize key model parameters.
Training & Evaluation: Monitors loss curves and evaluates performance using metrics like RMSE.
Results Visualization: Graphical analysis of training loss and validation loss for different configurations.

# Project Structure
📂 RUL_Prediction
├── data_loader.py       # Handles CMAPSS dataset loading
├── preprocessing.py     # Data preprocessing steps
├── model.py             # Transformer model implementation
├── train.py             # Training and evaluation pipeline
├── hyperparameter.py    # Optuna-based hyperparameter tuning
├── visualization.py     # Generates training/validation loss graphs
├── README.md            # Project documentation

# Installation
1. Clone the repository
git clone https://github.com/your-username/rul-prediction.git
cd rul-prediction

2. Create a virtual environment and activate it
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3. Install dependencies
pip install -r requirements.txt

# Usage
1. Data Preprocessing
python preprocessing.py

3. Model Training
python train.py

3. Hyperparameter Tuning with Optuna
python hyperparameter.py

4. Visualization
python visualization.py

# Results
The project includes a comparative analysis of different batch sizes, number of trials, and epochs. Example training loss and validation loss curves are provided to analyze model performance.

# Future Improvements
Can experiment with alternative deep learning architectures.
Currently working to deploy the model as an API for real-time RUL predictions.

# License
MIT License

# Contact
For questions or contributions, feel free to open an issue or reach out at 'majeed.msaad98@gmail.com'.
