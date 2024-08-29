import os 
import pandas as pd
from utils.MyUtils import MyPipeline

#from utils.MyPreprocessing import PreprocessingDatasets

#path = "/home/cruncher/Desktop/@MestradoArthur/Fairness/Datasets/MAR-random_Multivariado"
#arqs = os.listdir(path)
#le = PreprocessingDatasets()

#for arq in arqs:
#    nome = arq.split("_")[0]
#    if nome == "german":
#        df = pd.read_csv(os.path.join(path, arq))
#        df = le.label_encoder(df, ["target"])
#        df.to_csv(os.path.join(path,arq))

#######################################################################################################
# Salva as folds unificadas

path = "/home/cruncher/Desktop/@MestradoArthur/Fairness/Datasets/MAR-random_MultivariadoAll"


for name_dataset in ["german_credit",
                    "adult",
                    "bank",
                    "credit_card",
                    "diabetes",
                    "dutch",
                    "law",
                    "ricci",
                    "compass_7k",
                    "compass_4k",
                    "student_math",
                    "student_port",
                    "kdd"
                      ]:

    for model_impt in ["mean",
                    "customKNN",
                    "mice",
                    "pmivae",
                    #"saei",
                    "gain",
                    "softImpute"
                      ]:
        
        for mr in [10,20,40,60]:

            folds = []
            #features_protected = MyPipeline.retorna_featuresFairness(name_dataset)
            #for x_miss in features_protected:
            for fold in range (5):
              arq = f"{name_dataset}_{model_impt}_fold{fold}_md{mr}.csv"
                  #arq = f"{name_dataset}_{model_impt}_fold{fold}_md{mr}_{x_miss}.csv"
              df = pd.read_csv(os.path.join(path,arq))
              folds.append(df)
                  
              df_unificado = pd.concat(folds, ignore_index=True)
              #df_unificado.to_csv(f"/home/cruncher/Desktop/@MestradoArthur/Fairness/Datasets/{name_dataset}_{model_impt}_md{mr}_{x_miss}.csv", index=False)
              df_unificado.to_csv(f"/home/cruncher/Desktop/@MestradoArthur/Fairness/Datasets/{name_dataset}_{model_impt}_md{mr}.csv", index=False)        
print("done")