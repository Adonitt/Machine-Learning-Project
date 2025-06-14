﻿# Machine-Learning-Project

1. Dataset Description
•	Source: Kaggle - Official FIFA 23 dataset
https://www.kaggle.com/datasets/bryanb/fifa-player-stats-database?select=FIFA23_official_data.csv
•	Columns Used:
o	Age
o	Overall
o	Potential
o	Club
o	Value
o	Special
o	Preferred Foot
o	International Reputation
o	Weak Foot
o	Skill Moves
o	Work Rate
o	Body Type
o	Position
o	Height
o	Weight
o	Release Clause

•	Target Variables:
o	Model 1: Player "Value" (Regression)
o	Model 2: "Preferred Foot" (Classification)
________________________________________
2. EDA Summary (Key Insights & Charts)
•	Most players are right-footed (~75%)
•	Player value is correlated with Overall and Potential ratings
•	Certain positions have higher average values (e.g., ST, CAM)
•	Distribution plots, heatmaps, and pairplots were used to identify trends and correlation
________________________________________


3. Modeling Process
Model 1: Linear Regression
•	Goal: Predict player's market value in euros
•	Why LR: Simple, interpretable, and widely used for numerical prediction
•	Preprocessing:
o	Converted Value, Height, Weight, and Release Clause to numerical values
o	Handled missing data (e.g., filled with 0 or mean values)
o	Label encoded categorical columns
•	Pros: Fast, interpretable
•	Cons: Sensitive to outliers, assumes linearity
Model 2: K-Nearest Neighbors (KNN)
•	Goal: Predict whether a player is left-footed or right-footed
•	Why KNN: Simple, intuitive, good for classification tasks with mixed features
•	Preprocessing:
o	Encoded "Preferred Foot" as target (0=Left, 1=Right)
o	Normalized features (optional but recommended)
o	Label encoded categorical columns like Club, Position
•	Pros: Non-parametric, easy to implement
•	Cons: Slow for large datasets, affected by feature scale and irrelevant features
•	Hyperparameters:
o	n_neighbors=5 (default)
________________________________________
4. Model Evaluation
Linear Regression (Player Value Prediction):
•	MAE: ~453,000 EUR
•	MSE: ~1.2e+12
•	RMSE: ~1,095,000 EUR
•	Interpretation: Average error in prediction is around 1 million EUR
KNN (Preferred Foot Prediction):
•	Accuracy: ~73%
•	Precision & Recall: Used classification report
•	Comment: Model performed fairly well despite class imbalance
________________________________________

5. Conclusion
•	Findings:
o	Value prediction is feasible but sensitive to data quality and outliers
o	Preferred Foot can be classified with decent accuracy using simple KNN
•	Applications:
o	Useful for scouts, game designers, and sports analysts
•	Limitations:
o	Missing or noisy data
o	Linear model’s simplicity may miss complex patterns
•	Future Work:
o	Try advanced models like XGBoost or Random Forests
o	Include more features like match stats, player position history
