# CNN-Plastic-Waste-Classification
Shell-Edunet Skills4Future AICTE Internship presented by Edunet Foundation, in collaboration with AICTE &amp; Shell, focusing on Green Skills using AI technologies.

---

## Overview  
This project focuses on building a Convolutional Neural Network (CNN) model to classify images of plastic waste into various categories. The primary goal is to enhance waste management systems by improving the segregation and recycling process using deep learning technologies.  

---

## Table of Contents  
- [Project Description](#project-description)  
- [Dataset](#dataset)  
- [Model Architecture](#model-architecture)  
- [Training](#training)  
- [Weekly Progress](#weekly-progress)   
- [Technologies Used](#technologies-used)  
- [Future Scope](#future-scope)
- [Sample Outputs](#sample-outputs)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Description  
Plastic pollution is a growing concern globally, and effective waste segregation is critical to tackling this issue. This project employs a CNN model to classify plastic waste into distinct categories, facilitating automated waste management.  

## Dataset  
The dataset used for this project is the **Waste Classification Data** by Sashaank Sekar. It contains a total of 25,077 labeled images, divided into two categories: **Organic** and **Recyclable**. This dataset is designed to facilitate waste classification tasks using machine learning techniques.  


### Key Details:
- **Total Images**: 25,077  
  - **Training Data**: 22,564 images (85%)  
  - **Test Data**: 2,513 images (15%)  
- **Classes**: Organic and Recyclable  
- **Purpose**: To aid in automating waste management and reducing the environmental impact of improper waste disposal.
  
### Approach:  
- Studied waste management strategies and white papers.  
- Analyzed the composition of household waste.  
- Segregated waste into two categories (Organic and Recyclable).  
- Leveraged IoT and machine learning to automate waste classification.  

### Dataset Link:  
You can access the dataset here: [Waste Classification Data](https://www.kaggle.com/datasets/techsash/waste-classification-data).  

*Note: Ensure appropriate dataset licensing and usage guidelines are followed.*  


## Model Architecture  
The CNN architecture includes:  
- **Convolutional Layers:** Feature extraction  
- **Pooling Layers:** Dimensionality reduction  
- **Fully Connected Layers:** Classification  
- **Activation Functions:** ReLU and Softmax  

### Basic CNN Architecture  
Below is a visual representation of the CNN architecture used in this project:  

<p align="center">
  <img src="https://github.com/tarangver/CNN-Plastic-Waste-Classification/blob/main/Images/CNN-Architecture.jpg" style="width:80%;">
</p>

## Training  
- **Optimizer:** Adam  
- **Loss Function:** Categorical Crossentropy  
- **Epochs:** Configurable (default: 25)  
- **Batch Size:** Configurable (default: 32)  

Data augmentation techniques were utilized to enhance model performance and generalizability.  

## Weekly Progress  
This section will be updated weekly with progress details and corresponding Jupyter Notebooks.  

### **Week 1: Libraries, Data Import, and Setup**  
- **Date:** 21st January 2025 - 24th January 2025  
- **Activities:**  
  - Imported the required libraries and frameworks.  
  - Set up the project environment.  
  - Explored the dataset structure.    

- **Notebooks:**  
  - [Week1-Libraries-Importing-Data-Setup.ipynb](Week1-Libraries-Importing-Data-Setup.ipynb)  
   
### **Week 2: Model Training, Evaluation, and Predictions**  
- **Date:** 28th January 2025 - 31st January 2025  
- **Activities:**  
  - Trained the CNN model on the dataset.  
  - Optimized hyperparameters to improve accuracy.  
  - Evaluated model performance using accuracy and loss metrics.  
  - Performed predictions on test images.  
  - Visualized classification results with a confusion matrix.  

- **Notebooks:**  
  - [Week2-Model-Training-Evaluation-Predictions.ipynb](Week2-Fitting-CNN-Model.ipynb)  

### **Week 3: Model Deployment with Streamlit App**  
- **Date:** 4th February 2025 - 7th February 2025  
- **Activities:**  
  - Developed a **Streamlit web application** for real-time waste classification.  
  - Uploaded the trained model to **Kaggle and GitHub** for public access.  
  - Finalized the **project documentation and README formatting**.

- **Notebooks:**  
  - [Week2-Week3-Combined-CNN-Model.ipynb](Week2-Week3-Combined-CNN-Model.ipynb)  


### **📌 Conclusion & Summary of Model Performance**  
#### **1️⃣ Overview of the Model**  
The trained **Convolutional Neural Network (CNN)** model was designed to classify waste into two categories:  
- **O (Organic Waste)**  
- **R (Recyclable Waste)**  

It was trained on a dataset of **training images** using convolutional layers, max-pooling, batch normalization, and fully connected dense layers. The model was optimized using **categorical cross-entropy loss** and evaluated based on **accuracy**.

---

#### **2️⃣ Model Evaluation on Test Data**  
After training, the model was evaluated on a separate test dataset. The results are as follows:  

✅ **Test Accuracy**: **85.32%**  
✅ **Test Loss**: **0.3997**  

🔹 This means the model correctly classifies waste in **85 out of 100 cases** on unseen data. The **low test loss** indicates that the model has learned meaningful patterns and is **not overfitting** significantly.

---

#### **3️⃣ Predictions and Sample Results**  
The model made predictions on test images, converting probability outputs into class labels. Here’s a sample of predicted vs. actual results:

- **Predicted Classes**: `['O', 'O', 'O', 'O', 'R', 'O', 'O', 'O', 'O', 'O']`
- **Actual Classes**: `['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']`

🔹 The model mostly predicted correctly, but **one case (index 5) was classified as "R" instead of "O"**, indicating a potential misclassification.

---

#### **4️⃣ Classification Report Analysis**  
The **classification report** provides deeper insights into model performance for each class:

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **O (Organic)** | 0.89 | 0.84 | 0.86 | 1401 |
| **R (Recyclable)** | 0.81 | 0.87 | 0.84 | 1112 |
| **Overall Accuracy** | **85%** |  
| **Macro Avg (Balanced Score for Classes)** | **85%** |
| **Weighted Avg (Adjusted for Imbalance)** | **85%** |

📌 **Key Observations:**  
🔹 **Precision for Organic (O) is higher**: The model is more confident when predicting **Organic waste** than **Recyclable waste**.  
🔹 **Recall for Recyclable (R) is higher**: The model captures more actual **Recyclable waste** but sometimes **mislabels Organic waste as Recyclable**.  
🔹 **Balanced F1-score**: The model performs **well for both classes**, with a small **bias towards predicting Organic waste correctly**.

---

#### **5️⃣ Confusion Matrix Insights**  
The **confusion matrix** helps visualize the model’s errors:  

- **1401 Organic waste samples**:  
  - **1180 were correctly classified as Organic (True Positives)**  
  - **221 were misclassified as Recyclable (False Negatives)**  

- **1112 Recyclable waste samples**:  
  - **968 were correctly classified as Recyclable (True Positives)**  
  - **144 were misclassified as Organic (False Positives)**  

📌 **Key Takeaways from Confusion Matrix:**  
✅ The model performs well overall but **struggles slightly more with distinguishing Recyclable waste** from Organic waste.  
✅ **221 Organic samples were wrongly classified as Recyclable waste**, which might be due to overlapping features (e.g., **food-contaminated paper/cardboard**).  

---

### **🚀 Final Conclusion:**
- The **CNN model** has achieved **85.32% accuracy**, which is a strong performance for waste classification.  
- The **model performs slightly better for Organic waste** but can sometimes **misclassify Recyclable waste**.  
- **Possible Improvements:**
  - Adding **more diverse training data** to reduce misclassification.
  - Applying **data augmentation** to expose the model to **more variations**.
  - Experimenting with **fine-tuning on pre-trained CNN models** (like ResNet, VGG16) to further boost accuracy.  
  - Tweaking **hyperparameters** (learning rate, dropout rate) to optimize performance.  

✅ **Overall, the model is highly effective and can be used for real-world waste classification tasks with further refinements!** 🔥♻️

---

## Technologies Used  
- Python  
- TensorFlow/Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib  
- Streamlit

## Future Scope  
- Expanding the dataset to include more plastic waste categories.  
- Deploying the model as a web or mobile application for real-time use.  
- Integration with IoT-enabled waste management systems.  

---

## Sample outputs

<p align="center">
  <img src="https://github.com/tarangver/CNN-Plastic-Waste-Classification/blob/main/Images/HomePage.png">
  <img src="https://github.com/tarangver/CNN-Plastic-Waste-Classification/blob/main/Images/Organic.png">
  <img src="https://github.com/tarangver/CNN-Plastic-Waste-Classification/blob/main/Images/Recyclable.png">
</p>

---

## Contributing  
Contributions are welcome! If you would like to contribute, please open an issue or submit a pull request.  

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details. 


---
