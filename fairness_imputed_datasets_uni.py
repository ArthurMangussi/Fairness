# -*- coding: utf-8 -*
import os
import pandas as pd
from supervised.automl import AutoML
from utils.MyFairness import Fairness
from utils.MyUtils import MyPipeline
from utils.MeLogSingle import MeLogger

from sklearn.metrics import f1_score

def processa_imputados_fairness(path:str, model_impt:str, nome_datasets:list, mecanismo:str):

    _logger = MeLogger()
    results_metrics = {}
    f1_metrics = {}
    for nome in nome_datasets:
        
        for mr in [10,20,40,60]:
            features_protected_list = MyPipeline.retorna_featuresFairness(nome)
            for x_miss in features_protected_list:
              _logger.info("="*40)
              _logger.info(f"{model_impt} = {nome} com md = {mr} na feature {x_miss}")
              imputed_dataset = pd.read_csv(path + f"/{nome}_{model_impt}_md{mr}_{x_miss}.csv")
              df = imputed_dataset.copy()
              features_protect = None
              X = df.drop(columns='target')
              y_imputed = df['target'].values
  
              if nome == "adult":
                  features_protect = X[["age","race","sex"]]
              elif nome == "ricci":
                  features_protect = X[["Race"]]
              elif nome == "german_credit":
                  features_protect = X[["age", "personal-status-and-sex"]]
              elif nome == "bank":
                  features_protect = X[["age", "marital"]]
              elif nome == "credit_card":
                  features_protect = X[["sex", "education","marriage"]]
              elif nome == "student_math" or nome == "student_port":
                  features_protect= X [["sex", "age"]]
              elif nome == "compass_7k" or nome == "compass_4k":
                  features_protect = X[["race", "sex"]]  
              elif nome == "law":
                  features_protect = X[["race","gender"]]
              elif nome == "diabetes":
                  features_protect = X[["gender"]]
              elif nome == "kdd":
                  features_protect = X[["sex","race"]]
              elif nome == "dutch":
                  features_protect = X[["sex"]]                  
              
  
              os.makedirs(f"./Fairness/ClassificaImputed/{mecanismo}/{nome}_{model_impt}_md{mr}_{x_miss}", exist_ok=True)
              automl = AutoML(results_path=f"./Fairness/ClassificaImputed/{mecanismo}/{nome}_{model_impt}_md{mr}_{x_miss}",
                              algorithms=["Xgboost"],
                              mode="Optuna",
                              model_time_limit=None,
                              total_time_limit = 10,
                              train_ensemble=False,
                              n_jobs=-1,
                              eval_metric="f1",
                              optuna_time_budget=60,
                              validation_strategy={"validation_type": "kfold",
                                                  "k_folds": 5,
                                                  "shuffle": True,
                                                  "stratify": True,
                                                  "random_seed": 123},
                              random_state=123,
                              fairness_metric="demographic_parity_ratio",
                              fairness_threshold=0.8,
                      )
              _logger.info("Treinamento do modelo")
              automl.fit(X,y_imputed)
  
              y_pred_baseline = automl.predict(X)
              
              fairness_class = Fairness()
              metricas_imputed = fairness_class.calcula_metricas_fairness(y_original=y_imputed,
                                                                      y_predito=y_pred_baseline,
                                                                      val_protecte=features_protect)
              
              _logger.info("Calcular metricas:")
              f1 = f1_score(y_true=y_imputed,
                    y_pred=y_pred_baseline)
      
              f1_metrics[f"{nome}_{model_impt}_md{mr}_{x_miss}"] = f1
              results_metrics[f"{nome}_{model_impt}_md{mr}_{x_miss}"] = metricas_imputed

    results_final_tab = pd.DataFrame(results_metrics).T
    results_final_tab.to_csv(f"./Fairness/classification_{mecanismo}fairness_{nome}_{model_impt}.csv")
    f1_final_tab = pd.DataFrame(list(f1_metrics.items()), columns=['Dataset', 'F1-score'])
    f1_final_tab.to_csv(f"./Fairness/f1{mecanismo}_score_{nome}_{model_impt}.csv")
    
    return _logger.info(f"Resultados salvos com sucesso: {model_impt}")

if __name__ == "__main__":
    path = "/home/cruncher/Desktop/@MestradoArthur/Fairness/Datasets/MNAR_Univariado"
    
    datasets = ["adult",
                "kdd",
                "german_credit", 
                "dutch",
                "bank",                          
                "credit_card",
                "compass_7k",
                "compass_4k",
                "diabetes",
                "ricci",
                "student_math",
                "student_port",
                "law"              
                ]

    mecanismo = "MNAR_Univariado-XGB"
    #processa_imputados_fairness(path,"mean",datasets,mecanismo)
    #processa_imputados_fairness(path,"softImpute",datasets,mecanismo)
    #processa_imputados_fairness(path,"customKNN",datasets,mecanismo)
    #processa_imputados_fairness(path,"gain",datasets,mecanismo)
    processa_imputados_fairness(path,"mice",datasets,mecanismo)
    processa_imputados_fairness(path,"pmivae",datasets,mecanismo)


                