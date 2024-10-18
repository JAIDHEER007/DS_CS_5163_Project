# Project Proposal Report

## 1. Problem Statement and Importance

### Problem Statement:
Tropical cyclones are among the most devastating natural disasters, characterized by strong winds and low atmospheric pressure. Predicting the **intensity** of tropical cyclones accurately is critical to minimizing the damage caused by these storms. Current models struggle with capturing the complex, non-linear relationships between meteorological variables, resulting in suboptimal forecasting.

### Importance:
- **Public safety**: Accurate predictions can help in issuing timely warnings, thereby saving lives and reducing casualties.
- **Infrastructure protection**: With accurate forecasts, governments and organizations can better protect critical infrastructure and reduce economic losses.
- **Improving preparedness**: Enhanced cyclone intensity forecasting allows for better resource allocation, including evacuations and emergency response plans.

### Potential Beneficiaries:
- Disaster management agencies.
- Coastal communities at risk from cyclones.
- Industries such as agriculture, insurance, logistics, and energy sectors, which depend on weather forecasts.

---

## 2. Literature Review

### 2.1 Paper 1: A Neural Network Regression Model for Tropical Cyclone Forecast
- **Problem**: Predicting the maximum potential intensity (MPI) of tropical cyclones in the Western North Pacific using meteorological data.
- **Methodology**: The authors applied a **competitive neural network classifier** and a **multiple linear regression** model to predict cyclone intensity. A variable selection procedure was used to identify the most significant variables (e.g., SST, pressure, and wind speed) to improve model accuracy.
- **Main Findings**: The model demonstrated improved accuracy in predicting the maximum potential intensity of cyclones, highlighting the importance of variable selection in machine learning models.

### 2.2 Paper 2: Comparative Analysis of Machine Learning Algorithms to Predict Tropical Cyclones
- **Problem**: Classifying the intensity of tropical cyclones based on various meteorological factors such as wind speed, pressure, and geographical coordinates.
- **Methodology**: The paper compared several machine learning algorithms, including **Random Forest, Decision Trees, Logistic Regression, and LSTM**. Five key features were used as inputs: latitude, longitude, maximum sustained wind speed (MSW), sea level pressure, and pressure drop at the cyclone's eye.
- **Main Findings**: The **Random Forest** algorithm achieved the highest accuracy (99.1%) in predicting cyclone intensity, followed closely by **C4.5 decision trees**. The study demonstrated that ensemble learning models tend to outperform other approaches.

### 2.3 Paper 3: Prediction of Tropical Cyclone Intensity via Deep Learning Techniques from Satellite Cloud Images
- **Problem**: Estimating the intensity of tropical cyclones using satellite cloud images.
- **Methodology**: The paper employed deep learning techniques such as **Vision Transformer (ViT)** and **Deep Convolutional Neural Network (DCNN)** to predict tropical cyclone intensity based on satellite imagery. The authors utilized data augmentation and smoothing techniques to enhance prediction accuracy.
- **Main Findings**: The hybrid model combining ViT and DCNN showed the best results, with a root mean square error (RMSE) of 9.81 knots, outperforming other image-based methods for cyclone intensity estimation.

---

## 3. Dataset

### Dataset(s) to be Used:
- **Source**: Best Track Data from the **India Meteorological Department (IMD)** and **NOAA** datasets for tropical cyclones.
- **Size**: The dataset consists of approximately **4,464 data points** from **183 tropical cyclones** over the North Indian Ocean, from 2001 to 2022.
  
### Input and Target Variables:
- **Input Variables**: 
  - Latitude (numerical)
  - Longitude (numerical)
  - Maximum sustained wind speed (MSW) (numerical)
  - Sea surface temperature (numerical)
  - Sea level pressure (numerical)
- **Target Variable**: 
  - Tropical cyclone intensity classification

  
  | Grade                  | Low Pressure System        | Maximum Sustained Wind Speed (MSWS) (Knots) |
  |------------------------|----------------------------|---------------------------------------------|
  | 0                      | Low Pressure Area (LP)      | < 17                                       |
  | 1                      | Depression (D)              | 17 - 27                                    |
  | 2                      | Deep Depression (DD)        | 28 - 33                                    |
  | 3                      | Cyclonic Storm (CS)         | 34 - 47                                    |
  | 4                      | Severe Cyclonic Storm (SCS) | 48 - 63                                    |
  | 5                      | Very Severe Cyclonic Storm (VSCS) | 64 - 89                              |
  | 6                      | Extremely Severe Cyclonic Storm (ESCS) | 90 - 119                        |
  | 7                      | Super Cyclonic Storm (SS)   | ≥ 120                                      |
 
---

## 4. Methodology and Success Measures
### 1. **Data Preprocessing**:
   - Handle missing values, erroneous records, and inconsistencies in the dataset.
   - Normalize numerical variables (e.g., wind speed, pressure, and SST) to ensure uniformity.
   - Encode categorical variables where necessary (e.g., cyclone intensity categories).

### 2. **Data Exploration and Visualization**:
   - Perform exploratory data analysis (EDA) to identify trends and patterns within the dataset.
   - Create visualizations such as heatmaps, scatter plots, and histograms to understand relationships between meteorological variables and cyclone intensity.
   - Use correlation analysis to identify the most relevant input variables affecting intensity.

### 3. **Data Modeling**:
   - Implement machine learning models including **Random Forest**, **C4.5 Decision Tree**, and **LSTM**.
   - Perform model tuning using cross-validation to improve performance and prevent overfitting.
   - Compare models based on accuracy, precision, and recall to choose the most effective algorithm.

### 4. **Model Evaluation and Success Metrics**:
   - **Accuracy**: The proportion of correct predictions among all predictions.
   - **Mean Absolute Error (MAE)**: A metric to measure the average magnitude of errors between predicted and actual cyclone intensities.
   - **F1 Score**: A harmonic mean of precision and recall, useful for assessing the model's performance, especially in imbalanced data.

### Why These Metrics?
- **Accuracy** is a fundamental measure to determine how well the model performs overall.
- **MAE** provides insight into the closeness of the predicted values to actual values for continuous predictions of cyclone intensity.
- **F1 Score** will be used to evaluate the balance between precision and recall, especially if the dataset has class imbalances (e.g., fewer severe cyclones).

---

## References
1. Liu, J. N. K., & Feng, B. (2005). A neural network regression model for tropical cyclone forecast. *Proceedings of the Fourth International Conference on Machine Learning and Cybernetics*.
2. Sundar, R., Varalakshmi, P., & Kumar, D. S. (2023). Comparative analysis of machine learning algorithms to predict the tropical cyclones. *2023 International Conference on Data Science, Agents, and Artificial Intelligence*.
3. Tong, B., Fu, J., Deng, Y., Huang, Y., Chan, P., & He, Y. (2023). Estimation of tropical cyclone intensity via deep learning techniques from satellite cloud images. *Remote Sensing, 15*, 4188.
