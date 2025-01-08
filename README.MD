# All tasks where done using tkinter as the visual interface to make usage, model training and inference direct and simple

# Task 1
## Views Prediction
- Model to predict Views for a given video from the video features using Scikit-Learn and tkinker for user interaction 
### Model Choice:
I chose the XGBoost (XGBRegressor) for several reasons:
- It handles non-linear relationships well
- It's robust against overfitting
- It can capture complex interactions between features
- It provides good feature importance metrics
- It typically performs well on numerical prediction tasks with moderate-sized datasets


### Data Preprocessing:
Cleaned missing values:
Converted time duration from HH:MM:SS format to total seconds
Standardized features using StandardScaler to ensure all features are on the same scale
Split the data into training and validation sets (80/20 split)

Added convert_duration_to_seconds: Ensures proper handling of invalid or missing Average view duration values.
Filled Remaining NaN Values: Replaced NaN with 0 in the prepare_data method to avoid issues during model training or scaling

### Model Metrics:
#### Root Mean Squared Error (RMSE):
RMSE is a measure of the average prediction error in the same units as the target (number of views). A lower RMSE indicates better performance.
The reported RMSE reflects the model’s ability to generalize and suggests how close the predictions are to the actual values.
#### R² Score:
The R² score indicates how much of the variance in the target variable (Views) is explained by the features.
A high R² (close to 1) implies the model captures most of the variance, while a low R² indicates room for improvement.
Significance of Features:
Using the feature importance provided by the XGBoost model:

#### Features
Features like Subscribers, Impressions click-through rate (%), and Average view duration often dominate as they directly impact engagement and video visibility.
Their high importance scores validate their predictive influence.
Other Features:
Features like Shares, Comments added, and Likes (vs. dislikes) (%) may have lower scores but still contribute meaningfully by indicating viewer engagement.

### Visualization:
The feature importance bar plot visually highlights the relative contributions of each feature, providing actionable insights for feature engineering.

### Predictions:
The trained model is used to predict the views for the videos in X_test.csv.
Estimation of Out-of-Sample Performance:
By evaluating metrics on the validation set during training, the out-of-sample performance (generalization) is estimated.

Feature Significance
Using XGBoost's built-in feature importance we easily identify which features contribute most to the model's predictions.
Visualize feature importance to offer insights into their relative influence.

### Train the Model:

Split the training dataset into a training set and a validation set.
Scale the features (if needed) to ensure the model performs optimally.
### Train the XGBoost regressor.
Evaluate Model:
Calculate metrics like RMSE and R² on the validation set to estimate out-of-sample performance.

Make Predictions:
Use the trained model to predict views for the videos in X_test.csv.

---


# TASK 2

# Audience Retention Prediction

## Overview
Audience retention prediction percentages for videos based on historical data and corresponding transcripts. It uses a trained machine learning model to generate retention predictions for specific video segments, particularly positions 0 to 5. The application displays results in a log and visualizes them using a scatter plot.

## Model Choice
The **Random Forest Regressor** was selected for this task because of its:
1. **Capability to Handle Nonlinear Relationships**: Audience retention data is influenced by various nonlinear factors such as content type, position in the video, and engagement levels.
2. **Robustness**: Random Forest avoids overfitting by averaging predictions across multiple decision trees, making it ideal for noisy data.
3. **Feature Importance**: It provides insights into the contributions of different features, aiding interpretability.

## Implementation Details
### Data Preparation
1. **Retention Data**: Loaded from `.xlsx` files, ensuring only numeric values are processed.
2. **Transcripts**: Extracted from `.docx` files, with word counts computed as features.

### Feature Engineering
Key features include:
- **Average Retention**: Overall engagement level of the audience.
- **Word Count**: Indicates the density of information in the transcript.
- **Position**: The segment of the video for which predictions are made.

### Model Training
- The model was trained using historical data, associating the above features with retention percentages.
- The training dataset comprised multiple videos and their corresponding retention graphs and transcripts.

### Prediction Process
1. **Sample Data Input**:
   - Retention data (`.xlsx`) with missing values is uploaded.
   - Corresponding transcript (`.docx`) is provided.
2. **Feature Extraction**:
   - Average retention and word count are calculated.
   - Position indices (0 to 5) are added as features.
3. **Model Inference**:
   - The trained model predicts retention values for positions 0 through 5 using the extracted features.

## Application Workflow
1. **Load Data Folder**:
   - Uploads a folder containing historical video data (`.xlsx`) and transcripts (`.docx`).
2. **Load Sample File**:
   - Uploads the sample `.xlsx` file with missing retention values for prediction.
3. **Predict Retention**:
   - Generates retention predictions for positions 0 to 5 and displays the results in the log and plot tabs.

## Output
### Log Tab
- Displays retention predictions for positions 0 through 5 in percentage values.

### Plot Tab
- Visualizes the predictions using a plot, showing retention percentages across video positions.

## Example Usage
1. Upload historical video data and transcripts via the `Load Data Folder` button.
2. Upload the sample `.xlsx` file via the `Load Sample File` button.
3. Click the `Predict Retention` button to generate predictions and view results in the log and plot tabs.

---

# Task 3

# Intro Optimization For Audience Retention 

## Overview
The **Intro Optimization script** is designed to improve audience retention percentages for video introductions. By analyzing historical transcript data and retention metrics, the application identifies the best-optimized introduction to maximize audience engagement.

This tool allows users to:
1. Train a predictive retention model based on existing data.
2. Generate and evaluate optimized introductions using advanced NLP techniques.
3. Display the best introduction with its predicted retention prominently.

## Features
- **Data Upload**: Upload video transcripts and their corresponding retention metrics.
- **Model Training**: Train a machine learning model to predict retention percentages.
- **Introduction Optimization**: Generate multiple optimized introductions using:
  - OpenAI's GPT-4 for advanced text rewriting.
  - SpaCy NLP techniques for keyword-based optimization.
- **Best Introduction Highlighting**: Displays the best introduction in green and bold font.
- **Clean and Ordered Output**: Clears previous logs when a new sample is uploaded.
- **Loading Indicator**: Shows a loading message during inference for better user experience.


## How It Works

### 1. Data Preparation
- Upload a folder containing:
  - Transcripts (`.docx`) of videos.
  - Retention metrics (`.xlsx`) corresponding to the videos.
- The application extracts the first three sentences of each transcript and computes average retention for the early positions (0–10%).

### 2. Model Training
- The application uses:
  - **TF-IDF Vectorizer** to convert textual introductions into numerical features.
  - **Random Forest Regressor** to predict retention percentages based on these features.

### 3. Introduction Optimization
- After uploading a sample transcript and retention file:
  1. The tool evaluates the original introduction's predicted retention.
  2. It generates optimized introductions using GPT-4 and SpaCy.
  3. Each optimized introduction is evaluated, and the one with the highest predicted retention is selected.

### 4. Display Results
- Logs details about the original and optimized introductions.
- Highlights the best introduction with its predicted retention prominently in the output area.

---

## User Guide

### Upload Data Folder
1. Click **Upload Data Folder**.
2. Select a folder containing `.docx` transcripts and `.xlsx` retention metrics.
3. The application processes the data and prepares it for training.

### Train the Model
1. Click **Train Model** after uploading the data.
2. The model will analyze the provided data and create a predictive pipeline.
3. A log entry confirms when training is complete.

### Optimize a Sample Introduction
1. Click **Upload Sample**.
2. Select a `.docx` file (transcript) and its corresponding `.xlsx` file (retention data).
3. The application will:
   - Clear previous logs.
   - Display a loading indicator while processing.
   - Generate and evaluate multiple optimized introductions.
   - Highlight the best introduction in bold green text.

---

## Implementation Details

### Key Components
1. **GUI**: Built with Tkinter for intuitive user interaction.
2. **Machine Learning Pipeline**:
   - **TF-IDF Vectorizer**: Extracts features from textual introductions.
   - **Random Forest Regressor**: Predicts retention percentages.
3. **NLP Optimization**:
   - **GPT-4**: Generates advanced text rewrites to improve engagement.
   - **SpaCy**: Extracts keywords for simpler optimizations.
4. **Logging and Display**:
   - Clears logs for clean outputs.
   - Highlights the best introduction prominently.
   - Includes a loading indicator during inference.

### Code Structure
- **`load_data_folder`**: Loads and processes data for model training.
- **`train_model`**: Trains the ML model to predict retention.
- **`optimize_intro_using_reps`**: Handles the optimization process, including:
  - Generating optimized introductions.
  - Evaluating and selecting the best introduction.
- **`display_best_intro`**: Displays the best introduction prominently.
- **`generate_optimized_intro_gpt`**: Uses OpenAI's GPT and SpacyNLP to generate text rewrites.

---

## Reasoning Behind Implementations

1. **Clear Output Box**:
   - Ensures logs are clean and organized for every new sample.

2. **Loading Indicator**:
   - Enhances user experience by providing feedback during lengthy inference processes.

3. **Highlight Best Introduction**:
   - Bold green text makes the best result stand out for better visibility.

4. **TF-IDF with Random Forest**:
   - TF-IDF effectively transforms text into numerical features.
   - Random Forest provides robust predictions for tabular data.

5. **GPT-4 for Optimization**:
   - Leverages advanced natural language generation for creative and engaging text rewrites.

6. **SpaCy for Simplicity**:
   - Extracts keywords to create concise and relevant introductions.

