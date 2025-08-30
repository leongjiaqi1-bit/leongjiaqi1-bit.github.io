---
layout: post
author: Name
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background
Our team's project focused on exploring **early detection of mental health conditions through a data-centric approach**. Mental health issues such as depression, anxiety, and burnout are prevalent among young adults, but stigma often prevents open discussions and early intervention. By leveraging data, the project aimed to identify **early warning signs** and **key contributing factors** to mental health conditions. 

Business Goal:
To develop a data-driven solution that identifies early warning signs and key contributing factors of mental health challenges among adults, with the aim of enabling timely intervention and reducing the risk of long-term mental health deterioration.

The sub-objective I worked on is investigating how lifestyle factors (sleep, exercise, screen time etc.) contribute to mental health conditions. 

## Work Accomplished
The dataset I used contained 3,000 entries across 12 columns, with 11 of the columns complete and one column, Mental Health Condition, containing 595 missing values. 

### Data Preparation
Initial preparation involved cleaning categorical data to ensure consistency, for example verifying that categories like “Exercise Level” did not contain duplicates or misspellings. I then created a binary target variable representing whether a person had a mental health condition (1) or not (0), and removed the original multi-class target column. Because the dataset was imbalanced, with significantly more people recorded as having a condition, I applied resampling methods such as SMOTE and SMOTEENN to create a balanced dataset for training. To enrich the dataset, I also engineered features such as interaction terms (for example, low sleep combined with high stress) and binned continuous variables like sleep hours into categories of low, normal, and high.

### Modelling
For the modelling stage, I split the dataset into 70% training and 30% testing, with stratification to preserve class ratios. Two classification models were implemented: Logistic Regression and Random Forest. Logistic Regression served as a baseline model that provided interpretability, while Random Forest was chosen for its ability to capture non-linear relationships and variable interactions. To further improve model performance, I conducted hyperparameter tuning on the Random Forest using GridSearchCV with 3-fold cross-validation. A total of 144 parameter combinations were tested, and the best configuration was found with 100 estimators, a maximum depth of 20, maximum features set to ‘sqrt,’ a minimum of one sample per leaf, and a minimum of five samples per split.

### Evaluation
Evaluation results showed clear differences between the models. Logistic Regression achieved an accuracy of 0.48, with a precision of 0.80, recall of 0.47, F1-score of 0.59, and ROC-AUC of 0.514. While it was somewhat effective at identifying individuals with no condition, it struggled with detecting those with conditions, resulting in 382 false negatives and only 339 true positives. The baseline Random Forest improved on this, achieving an accuracy of 0.59, precision of 0.81, recall of 0.64, F1-score of 0.71, and ROC-AUC of 0.5096. It correctly identified 459 true positives while reducing false negatives to 262. The tuned Random Forest performed slightly better still, with an accuracy of 0.60, precision of 0.81, recall of 0.64, F1-score of 0.72, and ROC-AUC of 0.5172. Importantly, the tuned model further reduced false negatives, capturing more individuals with actual conditions.

In summary, Random Forest consistently outperformed Logistic Regression, especially in recall and F1-score. This made it the stronger model for supporting our business objective of identifying individuals at risk based on lifestyle factors.

## Recommendation and Analysis

The analysis confirmed that lifestyle features such as stress level, sleep hours, exercise level, work hours, social interaction, and happiness score contribute meaningfully to mental health outcomes. However, the correlation between individual features and the target variable was low, suggesting that no single factor strongly determined mental health conditions in isolation. Instead, predictive power emerged when lifestyle features were combined in machine learning models.

Based on these findings, I recommend enriching the dataset with additional data sources such as medical history, digital screen-time monitoring, and social support measures to improve predictive power. For preventive health programmes, organisations and policymakers should place greater emphasis on stress management, sleep hygiene, and exercise promotion, as these lifestyle factors consistently showed measurable links to mental health in our models. From a technical standpoint, the Random Forest model should be prioritised for deployment, as it provides a better balance of precision and recall compared to Logistic Regression and is more effective at identifying individuals with conditions.

## AI Ethics 

Applying machine learning to sensitive topics like mental health inevitably raises ethical considerations. Privacy is paramount, as mental health and lifestyle data are highly personal and must be anonymised to avoid misuse or discrimination. Fairness is another concern; the original dataset was imbalanced, which could bias predictions toward over-detecting conditions. To mitigate this, I applied SMOTE and SMOTEENN resampling, but continuous monitoring is needed to ensure fairness across different demographic groups.

Accuracy also poses an ethical challenge, since the models achieved only modest ROC-AUC scores of around 0.51, suggesting limited separability between classes. Deploying such models without improvement could lead to harmful misclassifications. Accountability must also be considered: predictions should never be used in isolation but rather accompanied by human oversight, particularly when informing mental health decisions. Finally, transparency is essential for trust. Logistic Regression provided interpretable coefficients that highlighted the direction of lifestyle factors (e.g., higher screen time and work hours increased risk, while better sleep and exercise reduced it). Such interpretability is critical for explaining predictions to stakeholders and ensuring ethical adoption.

## Source Codes and Datasets
Upload your model files and dataset into a GitHub repo and add the link here. 
