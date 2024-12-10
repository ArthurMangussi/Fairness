# Exploring the Influence of Missing Data Imputation in Group Fairness Metrics

This repository contains the codebase for the paper: *Exploring the Influence of Missing Data Imputation in Group Fairness Metrics*

## Paper Details
- Authors: Arthur Dantas Mangussi, Ricardo Cardoso Pereira, Miriam Seone Satnos, Ana Carolina Lorena, Mykola Pechenizkiy, and Pedro Henriques Abreu
- Abtract:Missing data is a common problem in real-world datasets and can be characterized as the lack of information on one or multiple variables 
in a dataset. The most frequent technique for handling this issue is imputation, which consists in the replacement of the missing values according 
to a predefined criterion. Since missing values are often imputed based on the known values in the dataset, existing data issues can be propagated 
during the imputation process. One such issue is fairness, a concept integral to responsible Artificial Intelligence practices. This work investigates 
the impact of the imputation process on system fairness by examining how imputation affects the fairness of predictions in Machine Learning models. 
It provides a comprehensive analysis covering thirteen unfair benchmark datasets with six state-of-the-art imputation strategies under synthetic
 Missing Not At Random and Missing At Random mechanisms in a multivariate scenario with 10\%, 20\%, 40\%, and 60\% of missing rates. Fairness was 
 measured by the following metrics: Statistical Parity, Equalized Odds, Equality of Opportunity, Predictive Equality, Equality of Positive, and 
 Negative Predicted Values. The results demonstrate that the missing mechanism, the classifier choice, and the imputation strategy decisively 
 influence the fairness of the predictions obtained by the Machine Learning models.
- Keywords: Missing Data, Fairness, Responsible Artificial Intelligence
- Year: 2024
- Contact: mangussiarthur@gmail.com

## Installation
```bash
git clone https://github.com/ArthurMangussi/Fairness.git
cd Fairness
pip install -r requirements.txt
```

## Acknowledgements
The authors gratefully acknowledge the Brazilian funding agencies FAPESP (Fundação Amparo à Pesquisa do Estado de São Paulo) under 
grants 2021/06870-3, 2022/10553-6, and 2023/13688-2. Moreover, this research was supported in part by Portuguese Recovery and Resilience Plan (PRR) 
through project C645008882-00000055 Center for Responsable AI.
