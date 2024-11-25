# Product Sales Prediction

This project leverages historical sales data and machine learning techniques to predict whether a product will sell within the next six months. By analyzing product attributes, the model provides actionable insights for optimizing inventory, helping businesses decide which products to retain and which to remove.

---

## **Project Overview**

The dataset contains both historical sales data and active inventory records. For this project:
- **Objective:** Predict the `SoldFlag` (1 = sold, 0 = not sold) for historical products.
- **Use Case:** Aid inventory decision-making by identifying products likely to sell versus those with low sales potential.

Key challenges addressed in the project include:
- Handling class imbalance, as only 17% of the historical products are marked as sold (`SoldFlag=1`).
- Evaluating and improving the model's performance using techniques like k-fold cross-validation, class rebalancing, and feature engineering.

---

## **Steps Taken**

### **1. Initial Data Preparation**
- Filtered the dataset to include only historical records since active inventory does not contain sales information (`SoldFlag` and `SoldCount`).
- Removed irrelevant columns like `Order`, `SoldCount`, and `File_Type`.
- Shuffled the data to ensure it is evenly distributed for cross-validation.
- Split the data into features (`X`) and target (`y`) variables:
  - `X`: Contains product attributes used for prediction.
  - `y`: Contains the binary target variable, `SoldFlag`.

---

### **2. Addressing Class Imbalance**
- The dataset was imbalanced, with 63,000 records for `SoldFlag=0` (majority class) and only 12,996 records for `SoldFlag=1` (minority class). This imbalance led to:
  - A misleading accuracy of **83%**, as the model predominantly predicted the majority class.
  - A low F1 score of **24%**, indicating poor performance in predicting the minority class.

**Solution:**
- Combined **undersampling** and **oversampling**:
  1. **Undersampling** reduced the majority class to the average size of the two classes.
  2. **Oversampling** increased the minority class to match the reduced majority class using the `RandomOverSampler` from `imbalanced-learn`.

**Impact:**
- Improved performance with:
  - Accuracy: **76.63%**
  - F1 Score: **43.62%**

---

### **3. Feature Engineering**
- **Marketing Type:** One-hot encoded, as it is a binary categorical variable (`D` and `F`).
- **Release Number:** Initially treated as a numeric (ordinal) feature. Later, it was one-hot encoded to better capture its categorical nature.
  - **Why One-Hot Encode?**
    - The `ReleaseNumber` column likely represents distinct product lines or versions, not a progressive ordinal feature.
    - One-hot encoding eliminates the assumption that higher numbers (e.g., `10 > 5`) imply better or newer releases.
  - **Impact:** Further improved accuracy and F1 score:
    - Accuracy: **77.14%**
    - F1 Score: **43.70%**

---

## **Challenges Faced**

1. **Class Imbalance:**
   - Predicting the positive class (`SoldFlag=1`) was challenging due to the dataset's imbalance.
   - Balancing the classes using undersampling and oversampling significantly improved the F1 score.

2. **Feature Representation:**
   - Deciding how to encode features like `ReleaseNumber` involved assumptions about the data's nature (ordinal vs. categorical).
   - Experimentation showed that treating `ReleaseNumber` as categorical (via one-hot encoding) yielded better results.

3. **Model Evaluation:**
   - High accuracy in initial tests masked the model's inability to predict the minority class accurately.
   - Using the F1 score as the evaluation metric provided a more balanced view of the model's performance.

---

## **Model Evaluation**

- **Metrics Used:**
  - **Accuracy:** Measures the overall percentage of correct predictions but can be misleading in imbalanced datasets.
  - **F1 Score:** Balances precision and recall, focusing on the positive class (`SoldFlag=1`).

- **Results:**
  - After addressing class imbalance and refining feature engineering:
    - Accuracy: **77.14%**
    - F1 Score: **43.70%**

---

## **Code Structure**

1. **Data Preprocessing:**
   - Filtered historical data.
   - Dropped irrelevant columns.
   - Applied one-hot encoding for categorical features (`MarketingType` and `ReleaseNumber`).
   - Addressed class imbalance using undersampling and oversampling.

2. **Pipeline Construction:**
   - Used `ColumnTransformer` to preprocess categorical columns.
   - Built a pipeline combining preprocessing steps and a `RandomForestClassifier`.

3. **Model Training and Validation:**
   - Used **k-fold cross-validation** (5 folds) to:
     - Split data into training and testing sets.
     - Train the model on each fold and evaluate performance metrics.
   - Ensured robust evaluation by averaging scores across folds.

---

## **Conclusion**

This project successfully demonstrates the use of machine learning to predict product sales likelihood based on historical data. By addressing key challenges such as class imbalance and feature engineering, the final model achieved improved performance metrics. 

While the results are promising, further steps, such as experimenting with more advanced models or collecting additional data, could enhance predictions.

---

## **How to Run**

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:
3. run the script:
    python productsold.py SalesKaggle3.csv
