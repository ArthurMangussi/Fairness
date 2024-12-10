# -*- coding: utf-8 -*-
# Bibliotecas
import pandas as pd
import numpy as np 

from sklearn.model_selection import StratifiedKFold

from utils.MyModels import ModelsImputation
from utils.MyUtils import MyPipeline
from utils.MyPreprocessing import PreprocessingDatasets
from utils.MyFairness import Fairness
from utils.MyResults import AnalysisResults
from utils.MeLogSingle import MeLogger

from mdatagen.multivariate.mMAR import mMAR

import multiprocessing

from time import perf_counter
import os 

dict_types_datasets = MyPipeline.retorna_dict_types_datasetsFairness()

def pipeline_fairness(model_impt:str, mecanismo:str, tabela_resultados:dict):
    _logger = MeLogger()
    with open(f'./Fairness/Tempos/{mecanismo}_Multivariado/tempo_{model_impt}.txt','w') as file:
        # Gerando resultados para os mecanismo
        for dados, nome in zip(tabela_resultados['datasets'], tabela_resultados['nome_datasets']):
            for md in tabela_resultados['missing_rate']:
                df = dados.copy()
                file.write(f'Dataset = {nome} com MD = {md}\n')
                _logger.info(f'Dataset = {nome} com MD = {md} no {model_impt}\n')
                fold = 0

                cv = StratifiedKFold()
                X = df.drop(columns='target')
                y = df['target'].values
                x_cv = X.values

                results_cv = {}
                for train_index, test_index in cv.split(x_cv, y):
                    _logger.info(f'Fold = {fold}')
                    # self._logger.info(f'Fold = {fold}')
                    x_treino, x_teste = x_cv[train_index], x_cv[test_index]
                    y_treino, y_teste = y[train_index], y[test_index]
                
                    X_treino = pd.DataFrame(x_treino, columns=X.columns)
                    X_teste = pd.DataFrame(x_teste, columns=X.columns)

                    # Inicializando o normalizador (scaler)
                    scaler = PreprocessingDatasets.inicializa_normalizacao(X_treino)

                    # Normalizando os dados
                    X_treino_norm = PreprocessingDatasets.normaliza_dados(
                        scaler, X_treino
                    )
                    X_teste_norm = PreprocessingDatasets.normaliza_dados(scaler, X_teste)

                    # Gera��o dos missing values em cada conjunto de forma independente
                    impt_md_train = mMAR(X=X_treino_norm, 
                                         y=y_treino, 
                                         n_xmiss=X_treino.shape[1]+1
                                        )
                    X_treino_norm_md = impt_md_train.random(missing_rate=md)
                    X_treino_norm_md = X_treino_norm_md.drop(columns='target')

                    impt_md_test = mMAR(X=X_teste_norm, 
                                        y=y_teste,
                                        n_xmiss=X_teste_norm.shape[1]+1
                                        )
                    X_teste_norm_md = impt_md_test.random(missing_rate=md)
                    X_teste_norm_md = X_teste_norm_md.drop(columns='target')

                    inicio_imputation = perf_counter()
                    # Inicializando e treinando o modelo
                    model_selected = ModelsImputation()
                    if model_impt == 'saei':
                        # SAEI
                        model = model_selected.choose_model(
                            model=model_impt,
                            x_train=X_treino_norm,
                            x_test=X_teste_norm,
                            x_train_md=X_treino_norm_md,
                            x_test_md=X_teste_norm_md,
                            input_shape=X.shape[1],
                        )

                    # KNN, MICE, PMIVAE, MEAN, SoftImpute, GAIN, missForest
                    else:
                        model = model_selected.choose_model(
                            model=model_impt,
                            x_train=X_treino_norm_md,
                            x_test = X_teste_norm_md,
                            x_test_complete = X_teste_norm,
                            listTypes = dict_types_datasets[nome] #KNN with HEOM
                        )
                        

                    fim_imputation = perf_counter()
                    file.write(
                        f'Tempo de treinamento para fold = {fold} foi = {fim_imputation-inicio_imputation:.3f} s\n'
                    )

                    # Imputa��o dos missing values nos conjuntos de treino e teste
                    try:
                        
                        output_md_test = model.transform(
                            X_teste_norm_md.iloc[:, :].values
                        )
                    except AttributeError:
                        
                        fatores_latentes_test = model.fit(X_teste_norm_md.iloc[:, :].values)
                        output_md_test = model.predict(X_teste_norm_md.iloc[:, :].values)

                    _logger.info(np.isnan(output_md_test))
                    #Encode das vari�veis categ�ricas
                    df_output_md_teste = pd.DataFrame(output_md_test, columns=X.columns)
                    output_md_test = MyPipeline.encode_features_categoricas(nome, df_output_md_teste)
                    _logger.info(output_md_test.isna().sum())
                        
                    # Calculando MAE para a imputa��o no conjunto de teste
                    (
                        mae_teste_mean,
                        mae_teste_std,
                    ) = AnalysisResults.gera_resultado_multiva(
                        resposta=output_md_test.iloc[:,:].values,
                        dataset_normalizado_md=X_teste_norm_md,
                        dataset_normalizado_original=X_teste_norm,
                    )

                    tabela_resultados[
                        f'{model_impt}/{nome}/{md}/{fold}/MAE'
                    ] = {'teste': round(mae_teste_mean,3)}
                    
                    # Dataset imputado
                    data_imputed = pd.DataFrame(output_md_test.copy(), columns=X.columns)
                    data_imputed['target'] = y_teste

                    data_imputed.to_csv(f"./Fairness/Datasets/{mecanismo}_Multivariado/{nome}_{model_impt}_fold{fold}_md{md}.csv", index=False)
                    fold += 1

        resultados_final = AnalysisResults.extrai_resultados(tabela_resultados)

        # Resultados da imputa��o
        resultados_mecanismo = (
            AnalysisResults.calcula_metricas_estatisticas_resultados(
                resultados_final, 1, fold
            )
        )
        resultados_mecanismo.to_csv(
            f'./Fairness/Resultados/{mecanismo}_Multivariado/{model_impt}_{mecanismo}.csv',
            
        )    

    return None

if __name__ == "__main__":

    diretorio = "./DatasetsFairness"
    datasets = MyPipeline.carrega_datasets(diretorio)

    fairness = Fairness()
    tabela_resultados = fairness.cria_tabela_fairness(datasets)

    mecanismo = "MAR-random"

    # Cria diret�rios para salvar os resultados do experimento
    os.makedirs(f"./Fairness/Tempos/{mecanismo}_Multivariado", exist_ok=True)
    os.makedirs(f"./Fairness/Datasets/{mecanismo}_Multivariado", exist_ok=True)
    os.makedirs(f"./Fairness/Resultados/{mecanismo}_Multivariado", exist_ok=True)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

        args_list = [("mean",mecanismo,tabela_resultados),
                     ("customKNN",mecanismo,tabela_resultados),
                     ("mice",mecanismo,tabela_resultados),
                     ("pmivae",mecanismo,tabela_resultados),
                     ("saei",mecanismo,tabela_resultados),
                     ("softImpute",mecanismo,tabela_resultados),
                     ("gain",mecanismo,tabela_resultados)
                     ]
        
        pool.starmap(pipeline_fairness,args_list)
