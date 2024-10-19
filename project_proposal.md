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
- **Problem**: The paper addresses the challenge of predicting the **maximum potential intensity (MPI)** of tropical cyclones in the Western North Pacific. Traditional models for intensity forecasting, such as statistical-dynamical models, often fail to capture the complex, non-linear relationships between meteorological variables that affect cyclone intensity.
  
- **Methodology**: The authors propose a **competitive neural network classifier** in combination with a **multiple linear regression** model. The data used spans a 10-year period of tropical cyclones in the Western North Pacific and is obtained from the **Hong Kong Observatory**. The model selects significant variables, including **sea surface temperature (SST)** and atmospheric pressure, to train the network. By incorporating variable selection, the model aims to improve both the speed and accuracy of the neural network’s learning process. A **Binary Trigger** is employed to adjust the structure of the network layers dynamically during training, optimizing the performance.

- **Main Findings**: The model outperformed traditional methods in terms of accuracy and efficiency, demonstrating the ability of neural networks to manage complex meteorological data. The study showed that using a neural network with properly selected features improves the prediction of the maximum potential intensity of cyclones, even in the presence of non-linear relationships between meteorological variables.
  
- **Main Experimental Results**: The neural network regression model achieved a mean absolute error (MAE) of **6.72 knots** in predicting maximum potential intensity, outperforming previous statistical models, which had errors of up to **10 knots**.

---

### 2.2 Paper 2: Comparative Analysis of Machine Learning Algorithms to Predict Tropical Cyclones
- **Problem**: This paper explores the problem of classifying tropical cyclones based on their intensity using machine learning algorithms. The primary challenge lies in finding the most effective machine learning approach for cyclone classification based on multiple meteorological features, including wind speed, sea level pressure, and geographical data (latitude, longitude). The unpredictability of cyclone behavior, coupled with the sheer volume of meteorological data, makes accurate classification a difficult task.

- **Methodology**: The authors employ a comparative study of various machine learning algorithms, such as **Random Forest, C4.5 Decision Tree, Logistic Regression, LSTM**, and others. The dataset is derived from **Best Track Data (BTD)** provided by the **India Meteorological Department (IMD)** and spans cyclone data from 2001 to 2022. The input features include **latitude**, **longitude**, **maximum sustained wind speed (MSW)**, **sea level pressure**, and the **pressure drop at the cyclone's eye**. The models are evaluated based on their ability to classify cyclones into categories such as **depression**, **cyclonic storm**, **severe cyclonic storm**, and **super cyclonic storm**. Various metrics, such as accuracy, precision, recall, and F1 score, were used to assess the performance of each model.

- **Main Findings**: The **Random Forest** algorithm achieved the highest accuracy, reaching **99.1%**, closely followed by the **C4.5 Decision Tree**, with an accuracy of **98.54%**. Other algorithms, such as Logistic Regression, LSTM, and Nearest Centroid Classifier, performed well but were not as accurate as the ensemble methods. The study also revealed that models like **Random Forest** are better suited for handling high-dimensional meteorological data, as they effectively manage both continuous and categorical variables and are robust to noise and missing data.

- **Main Experimental Results**: The Random Forest model achieved a precision of **98.87%** and recall of **99.21%** in cyclone intensity classification, demonstrating superior performance compared to other models like Logistic Regression, which achieved an accuracy of only **94.06%**.

---

### 2.3 Paper 3: Prediction of Landfall Intensity, Location, and Time of a Tropical Cyclone
- **Problem**: This paper addresses the problem of predicting the **landfall intensity, location, and time** of tropical cyclones. Accurate prediction of these parameters is crucial for reducing the impact of cyclones on coastal regions. Traditional methods face challenges due to the complex and dynamic nature of cyclones as they approach landfall, often resulting in inaccuracies in predicting the cyclone's landfall behavior.

- **Methodology**: The authors developed a **Long Short-Term Memory (LSTM)** based recurrent neural network model to predict landfall intensity (in terms of maximum sustained wind speed), location (latitude and longitude), and time (in hours after the observation period) of tropical cyclones in the **North Indian Ocean**. The input data included **best track data** from the **India Meteorological Department (IMD)**, including meteorological variables such as **pressure**, **sea surface temperature**, **latitude**, **longitude**, and **intensity** (measured as maximum sustained wind speed). The model was trained on cyclone data from 1982 to 2020 and tested on three recent cyclones: **Bulbul, Fani, and Gaja**.

- **Main Findings**: The LSTM-based model provided state-of-the-art results, significantly improving the accuracy of landfall predictions compared to existing models. The authors also compared their model's predictions with actual data from recent cyclones to validate its performance.

- **Main Experimental Results**: The LSTM model achieved a **mean absolute error (MAE)** of **4.24 knots** for landfall intensity, **0.24 degrees** for latitude, and **0.37 degrees** for longitude. The distance error in predicting the landfall location was **51.7 kilometers**, and the error in predicting landfall time was **4.5 hours**. These results showed a marked improvement over traditional prediction methods.

---

## 3. Dataset

### Dataset(s) to be Used:
- **Tropical cyclone best track data Hong Kong Weather Obeservatory**
  - data on post analysed position and intensity of tropical cyclones over the western North Pacific and the South China Sea from 1985 to 2023
  - **Size**: 2 MB
  - **Dataset Link**: [https://data.gov.hk/en-data/dataset/hk-hko-rss-tropical-cyclone-best-track-data](https://data.gov.hk/en-data/dataset/hk-hko-rss-tropical-cyclone-best-track-data)
- **Tropical cyclone best track data Hong Kong Weather Obeservatory**
  - data on post analysed position and intensity of tropical cyclones over the western North Pacific and the South China Sea from 1985 to 2023
  - **Size**: 2 MB
  - **Dataset Link**: [https://data.gov.hk/en-data/dataset/hk-hko-rss-tropical-cyclone-best-track-data](https://data.gov.hk/en-data/dataset/hk-hko-rss-tropical-cyclone-best-track-data)
- **Hurricanes and Typhoons, 1851-2014**
  - The NHC publishes the tropical cyclone historical database in a format known as HURDAT, short for HURricane DATabase. These databases (Atlantic HURDAT2 and NE/NC Pacific HURDAT2) contain six-hourly information on the location, maximum winds, central pressure, and (starting in 2004) size of all known tropical cyclones and subtropical cyclones
  - **Size**: 9.53 MB
  - **Dataset Link**: [Kaggle.com/Hurricanes and Typhoons, 1851-2014](https://www.kaggle.com/datasets/noaa/hurricane-database/data)

- **INSAT3D Infrared & Raw Cyclone Imagery (2012-2021)**
  - Image Dataset containing all INSAT3D captured INFRARED and RAW Cyclone Imagery over the Indian Ocean fronm 2012 to 2021 along with each Cyclone Image intensity in KNOTS
  - **Size**: 47.46 MB
  - **Dataset Link**: [Kaggle.com/INSAT3D Infrared & Raw Cyclone Imagery (2012-2021)](https://www.kaggle.com/datasets/sshubam/insat3d-infrared-raw-cyclone-images-20132021)
  
### Input and Target Variables:
- **Input Variables**: 
  - Latitude (numerical)
  - Longitude (numerical)
  - Maximum sustained wind speed (MSW) (numerical)
  - Sea surface temperature (numerical)
  - Sea level pressure (numerical)
- **Target Variable**: 
  - Tropical cyclone intensity classification

  
  | Grade                  | Low Pressure System        |
  |------------------------|----------------------------|
  | 0                      | Low Pressure Area (LP)      |
  | 1                      | Depression (D)              |
  | 2                      | Deep Depression (DD)        |
  | 3                      | Cyclonic Storm (CS)         |
  | 4                      | Severe Cyclonic Storm (SCS) |
  | 5                      | Very Severe Cyclonic Storm (VSCS) |
  | 6                      | Extremely Severe Cyclonic Storm (ESCS) |
  | 7                      | Super Cyclonic Storm (SS)   |
 
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

### 4. Model Evaluation and Success Metrics

#### 1. **Accuracy**:
Accuracy is the ratio of correctly predicted instances to the total instances in the dataset. It gives an overall measure of the model’s performance but can be misleading when the dataset is imbalanced.

**Why Useful**: Accuracy provides a general idea of how well the model is performing across all cyclone intensity classes. It is useful for a quick snapshot of performance, but for this project, it needs to be considered with other metrics, especially since the dataset may have imbalanced classes of cyclone intensities.

#### 2. **Mean Absolute Error (MAE)**:
MAE measures the average magnitude of prediction errors, showing how far the model's predictions are from the actual values. It works well for continuous variables, such as wind speed, because it gives the average error size in the same units as the predictions.

**Why Useful**: MAE is particularly relevant for this project because it directly tells us how close the predicted cyclone intensity (e.g., wind speed) is to the actual value. Lower MAE values indicate better performance, making it a useful measure of how well the model is estimating cyclone intensity.

#### 3. **Root Mean Squared Error (RMSE)**:
RMSE is similar to MAE but gives more weight to larger errors. It penalizes large errors more than MAE does, making it more sensitive to outliers or major deviations in predictions.

**Why Useful**: RMSE is beneficial in this project as it highlights large deviations in the model’s cyclone intensity predictions. If the model significantly over- or under-predicts the intensity of a severe cyclone, RMSE will reflect this, helping to identify whether the model struggles with extreme cases.

#### 4. **Relative Absolute Error (RAE)**:
RAE compares the total absolute error of the model to the error of a simple baseline model, typically the mean of the target variable. It shows how well the model performs relative to a naive prediction.

**Why Useful**: RAE is important for this project because it benchmarks the model's performance against a simple prediction (e.g., always predicting the mean cyclone intensity). A low RAE indicates that the model is significantly better than the baseline, which adds context to its predictive power.

---

## References
1. Liu, J. N. K., & Feng, B. (2005). A neural network regression model for tropical cyclone forecast. *Proceedings of the Fourth International Conference on Machine Learning and Cybernetics*.
2. Sundar, R., Varalakshmi, P., & Kumar, D. S. (2023). Comparative analysis of machine learning algorithms to predict the tropical cyclones. *2023 International Conference on Data Science, Agents, and Artificial Intelligence*.
3. Kumar, S., Biswas, K., & Pandey, A. K. (2021). Prediction of landfall intensity, location, and time of a tropical cyclone. *Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence*.
