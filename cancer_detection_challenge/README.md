# Breast Cancer Detection Challenge
### Overview
This project analyzes the Breast Cancer Wisconsin dataset (breast-cancer-wisconsin.data.csv). The dataset contains clinical data used to classify if a pacient has cancer or not.

## Downloading the Code

To download the project from GitHub, follow these steps:

1. Open a terminal or command prompt.
2. Clone the repository using the following command:
   ```
   git clone https://github.com/ASantosMorales/progra2_mcd_challenges.git
   ```
3. Switch to `feature/cancer_challenge` branch.
4. Go to cancer_detection_challenge/ folder.

## Running the code
To execute the code, follow these steps:
1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Run the main script:
   ```
   python src/main.py
   ```

## MLflow Results Visualization
The project uses **mlflow** to track experiments and visualize results. Once the code has been executed, you can invoke from terminal the **mlflow** dashboard with:
   ```
   mlflow ui
   ```
Normally, the location is: **http://127.0.0.1:5000**

There, you will find:

+ Metrics and performance evaluation
+ Model used
+ Artifacts
+ Additional information about the experiment
