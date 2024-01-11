# Final Project Analysis Plan for [CPSC-392 Course](https://github.com/cmparlettpelleriti/CPSC392ParlettPelleriti)
Analysis performed on a life expectancy dataset that comprises immunization, mortality, economic, and social factors. Data was collected through The Global Health Observatory (GHO) data repository under World Health Organization (WHO) from years 2000-2015. For this analysis, we addressed the following questions:
1. When predicting life expectancy based on health and economic indicators, do regularization techniques improve the model’s performance? If so, which does more, Lasso or Ridge?
2. Would performing PCA on the whole data set improve K-Means clustering more than performing PCA on select features? When focusing on life expectancy, which variables are highly correlated and would benefit from PCA, and which ones would be better off without PCA, if any?
3. Can countries be clustered based on a combination of variables like life expectancy, adult mortality, schooling, total expenditure, and gdp?
4. How effectively can schooling and income composition of resources predict a country's life expectancy?
5. Can Supervised Learning Models such as Random Forest Regression, effectively predict life expectancy based on a set of health and socio-economic indicators (like Adult Mortality, Infant Deaths, Alcohol, Percentage Expenditure, Hepatitis B, Measles, BMI, Under-Five Deaths, Polio, Total Expenditure, Diphtheria, HIV/AIDS, GDP, Population, Thinness 1-19 years, Thinness 5-9 years, Income Composition of Resources, and Schooling), and determine the key factors that influence life expectancy?
6. Is it possible for clustering models, such as K-Means, to effectively group countries based on their health(like Life Expectancy, Adult Mortality, Alcohol, BMI, and HIV/AIDS)  and socio-economic indicators (like GDP, Income composition of resources, percentage expenditure, and Total expenditure), to provide insights into the relationship between life expectancy and socio-economic status?

## Table of Contents
- [Introduction](https://github.com/cbellerob/LifeExpectancy-CPSC392#introduction)
- [Analysis (Question 1)](https://github.com/cbellerob/LifeExpectancy-CPSC392#question-1-when-predicting-life-expectancy-based-on-health-and-economic-indicators-do-regularization-techniques-improve-the-models-performance-if-so-which-does-more-lasso-or-ridge)
- [Analysis (Question 2)](https://github.com/cbellerob/LifeExpectancy-CPSC392#question-2-would-performing-pca-on-the-whole-data-set-improve-k-means-clustering-more-than-performing-pca-on-select-features-when-focusing-on-life-expectancy-which-variables-are-highly-correlated-and-would-benefit-from-pca-and-which-ones-would-be-better-off-without-pca-if-any)
- [Contributors to Full Project](https://github.com/cbellerob/LifeExpectancy-CPSC392#contributors-to-full-project)

## Introduction  
The Life Expectancy dataset, comprising 2,938 observations and 22 features (2 categorical and 20 continuous) encompasses immunization, mortality, economic, and social aspects, and offers a holistic view of factors influencing life expectancy. These features are the country, year of collection, developed/developing status, life expectancy in age, adult mortality rates, number of infant deaths, alcohol consumption, Hepatitis B immunization, measles reports, BMI, under-five deaths, Polio immunization, Diphtheria immunization, HIV/AIDS deaths, GDP in USD, country population, thinness prevalence for 1-19 and 5-9 years of age, income composition of resources, years of schooling, and total and percentage expenditure. Employing advanced techniques such as LASSO and Ridge regression enhanced predictive accuracy, while Principal Component Analysis (PCA) simplified complex data patterns. K-Means clustering facilitated effective grouping based on health and socio-economic indicators, unveiling inherent relationships. Random Forest Regression predicted life expectancy and unraveled complex factors. Scree Plots in PCA ensured focused analysis, and our approach aims to provide actionable insights for informed decision-making. This comprehensive exploration empowers stakeholders and policymakers to make informed choices for improved public health outcomes and societal well-being. 

## Analysis for Questions 1 and 2
### Question 1: When predicting life expectancy based on health and economic indicators, do regularization techniques improve the model’s performance? If so, which does more, Lasso or Ridge?  

Linear Regression models are useful when predicting continuous values for a data set. So, for predicting the continuous value of an individual’s life expectancy, this supervised model was ideal. There are various ways to improve a Linear Regression model’s performance, one being regularization to avoid overfitting. Lasso and Ridge are common regularization techniques that penalize the coefficients that are too large.

To perform this analysis, initial preparation of the data was necessary. 14 out of 22 features in the data set had missing values, some features even missing over 500. Since dropping all of these values would result in a great loss of data, they needed to be handled differently. Instead, mean imputation was performed, where the mean was calculated and the missing values were substituted with it. This is a well-known approach for preserving the shape of the data as well as its distribution. Outliers also needed to be detected and handled so that they wouldn’t negatively affect the performance of the models. This was completed by calculating the z-score for each column and removing any values that were greater than 3 standard deviations away from the mean, as these would be considered extreme values. 

The predictors chosen for this analysis were potential health and economic indicators, which included Polio immunization, Diphtheria tetanus immunization, Hepatitis B immunization, measles cases, HIV/AIDS deaths, alcohol consumption, BMI, GDP in USD, total and percentage expenditures, and income composition of resources. These are all continuous variables, so they were standardized to the same unit scale during preprocessing. Train test split was performed, making 10% of the data be the test set. Three Linear Regression models were created: standard, lasso regularization, and ridge regularization. Before creating the regularization models, “.alpha_” was used to determine the optimal regularization strength; the alphas value was 0.01 for lasso and 10.0 for ridge. Pipelines for the preprocessing and models were built and fit on the training set. Then, predictions were made on the train and test data. Metrics such as MSE, MAE, MAPE, and R^2 were evaluated. Coefficient values and residual plots were visualized for each model.

It can be concluded that these models performed well, as MSE, MAE, and MAPE values were low (for example, MSE was ~13 for each model) and R^2 was high at ~0.81-0.82 for the models. They also weren’t overfit, which is supported by the metrics like R^2, where the test predictions weren’t less than the training predictions. Residuals also clustered around 0, where closer to 0 means more accurate predictions. When comparing the three models, regularization did improve performance. For instance, when looking at the R^2 metric, it was 0.8198 without regularization, 0.8120 with ridge regularization, and 0.8201 with lasso regularization. While this improvement isn’t significant, the change still exists and represents how regularization can adjust coefficients to affect predictor influence. There are a few reasons why the improvement wasn’t drastic. First, the predictors chosen for this model already had good correlation (see Figure for Question 4 section). Regularization tends to favor predictors with good correlation, so it’s not surprising that they didn’t adjust the coefficients too much. Second, this model has low complexity since there were only 9 predictors included and the number of examples wasn’t significantly large. Overall, though a small change, lasso and ridge did improve model performance with lasso increasing it the most. For a problem involving bigger data sets and higher complexity, regularization is an efficient technique for adjusting predictors, especially lasso if there is a stronger need for variable selection.

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/LR_coeff.jpg" 
  width="400" 
  height="300">  
**Figure 1: Coefficient values for Linear Regression Model**

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/LR_residual.jpg" 
  width="400" 
  height="300">  
**Figure 2: Residual Plot for Linear Regression Model**
<br></br>

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/Lasso_coeff.jpg" 
  width="400" 
  height="300">  
**Figure 3: Coefficient values for Lasso Regularization**

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/Lasso_residual.jpg" 
  width="400" 
  height="300">  
**Figure 4: Residual Plot for Lasso Regularization**
<br></br>

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/Ridge_coeff.jpg" 
  width="400" 
  height="300">  
**Figure 5: Coefficient values for Ridge Regularization**

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/Ridge_residual.jpg" 
  width="400" 
  height="300">  
**Figure 6: Residual Plot for Ridge Regularization**
<br></br>

### Question 2: Would performing PCA on the whole data set improve K-Means clustering more than performing PCA on select features? When focusing on life expectancy, which variables are highly correlated and would benefit from PCA, and which ones would be better off without PCA, if any?  

Principal Component Analysis is a dimensionality reduction technique to present data more concisely and efficiently. This model creates linear combinations of the data by creating components that contain a certain amount of the original data. PCA is beneficial to model performance, but the way it is used must be carefully considered. For instance, using it when data has good correlation and correctly choosing the number of components.

The data preparation for this analysis was consistent with the cleaning for question 3: mean imputation and z-scoring handled missing values and outliers respectively. Two PCA models were created, one containing all of the dataset’s features and a second having a select number of features. A correlation matrix was generated to determine which features had good correlation with life expectancy. Through observing the life expectancy row, most of the features had good correlation except for adult mortality, infant deaths, under-five deaths, HIV/AIDS, and thinness 1-9 and 5-9 years. All but these six features were selected for the second PCA model. Before creating the two PCA models, scree plots were generated for each group of variables to determine how many principal components to keep. Scree plots demonstrate the explained variance for each component (how much of the data is in each), so choosing the number of components at the inflection point is a good method as most of the data will be represented at this amount. 2 and 3 principal components were chosen for the model with all variables and the model with select features, respectively. Preprocessing was performed, where variables for each model were standardized, and then put into a pipeline with the models. The pipelines were fit to their variable data and a new representation of the original data was created.

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/corr_matr.jpg" 
  width="600" 
  height="500" 
  alt="Correlation Matrix for Life Expectancy">   
**Figure 7: Correlation Matrix for all features in Life Expectancy Dataset**
<br></br>

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/scree_all.jpg" 
  width="400" 
  height="300" >   
**Figure 8: Scree Plot for PCA model containing all continuous features**
<br></br>

<img src="https://github.com/cbellerob/LifeExpectancy-CPSC392/blob/main/Visualization/scree_selected.jpg" 
  width="400" 
  height="300" >   
**Figure 9: Scree Plot for PCA model containing features selected with high correlation**
<br></br>

Two K-Means clustering models were created with 3 clusters in each, which were chosen by the inflection point after visualizing the within cluster SSE for different K clusters. One K-Means model was fit to the all-variable principal components and the second K-Means model was fit to the selected-variable principal components. Silhouette scores were computed for each model and proved efficient clustering, with a 0.747745 score for all variables and a 0.747746 score for the selected variables.

Visualizing the correlation matrix and the scree plots were important steps for this analysis, to observe the variables that have good correlation and verify that correlation with the components’ ability to represent data. If a scree plot line is close to diagonal, then there is low correlation and the principal components aren’t combining the data well. However, the scree plots for both feature groups in this analysis had a clear inflection point, which makes sense due to the consistent correlations the variables had with life expectancy. Calculating the silhouette scores for the K-Means clusters was also necessary to see how well the models are at creating clusters. A high silhouette score indicates that data points are cohesive with others in its cluster and separate from those in other clusters; the silhouette scores for this analysis were high, meaning that K-Means were effective at clustering. While the difference between the two models wasn’t significant, this analysis shows how selecting features to undergo dimensionality reduction can improve a model. If there were way more features with low correlation, the difference between the all-variable and selected variable models would be noticeable because performing PCA with low correlation may result in a large loss of data.

## Contributors to Full Project
* Anika Nguyenkhoa
* Bandhavya Parvathaneni
* Caroline Robinson
