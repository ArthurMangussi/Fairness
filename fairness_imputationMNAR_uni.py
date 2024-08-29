# Bibliotecas
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils.MyModels import ModelsImputation
from utils.MyUtils import MyPipeline
from utils.MyPreprocessing import PreprocessingDatasets
from utils.MyFairness import Fairness
from utils.MyResults import AnalysisResults
from utils.MeLogSingle import MeLogger

from mdatagen.univariate.uMNAR import uMNAR
import multiprocessing

from time import perf_counter
import os 

# Para todas as features protegidas de cada vez, quando for mais de uma 
dict_types_datasets = MyPipeline.retorna_dict_types_datasetsFairness()

def pipeline_fairness(model_impt:str, mecanismo:str, tabela_resultados:dict):
    _logger = MeLogger()

    with open(f'./Fairness/Tempos/{mecanismo}_Univariado/tempo_{model_impt}.txt','w') as file:
        # Gerando resultados para os mecanismo
        for dados, nome in zip(tabela_resultados['datasets'], tabela_resultados['nome_datasets']):
            # Para todos os Missing Rates e Folds, a feature selecionada precisa ser a mesma.
            features_protected = MyPipeline.retorna_featuresFairness(nome)
            for x_miss in features_protected:
                for md in tabela_resultados['missing_rate']:
                    df = dados.copy()
                    file.write(f'Dataset = {nome} com MD = {md}\n')
                    fold = 0

                    cv = StratifiedKFold()
                    X = df.drop(columns='target')
                    y = df['target'].values
                    x_cv = X.values

                    for train_index, test_index in cv.split(x_cv, y):
                        _logger.info(f'{nome} com MD = {md} no Fold = {fold} imputed by {model_impt} na feature = {x_miss}')
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

                        # Geração dos missing values em cada conjunto de forma independente
                        impt_md_train = uMNAR(X=X_treino_norm, 
                                                y=y_treino, 
                                                missing_rate=md,
                                                x_miss=x_miss,
                                                threshold=0.6
                                                )
                        X_treino_norm_md = impt_md_train.run(deterministic=False)
                        X_treino_norm_md = X_treino_norm_md.drop(columns='target')

                        impt_md_test = uMNAR(X=X_teste_norm, 
                                            y=y_teste,
                                            missing_rate=md,
                                            x_miss=x_miss,
                                            threshold=0.6
                                            )
                        X_teste_norm_md = impt_md_test.run(deterministic=False)
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

                        # Imputação dos missing values nos conjuntos de treino e teste
                        try:
                            output_md_test = model.transform(
                                X_teste_norm_md.iloc[:, :].values
                            )
                        except AttributeError:
                            fatores_latentes_test = model.fit(X_teste_norm_md.iloc[:, :].values)
                            output_md_test = model.predict(X_teste_norm_md.iloc[:, :].values)

                        #Encode das variáveis categóricas
                        df_output_md_teste = pd.DataFrame(output_md_test, columns=X.columns)
                        output_md_test = MyPipeline.encode_features_categoricas(nome, df_output_md_teste)
                        
                        # Calculando MAE para a imputação no conjunto de teste
                        id_miss = list(X_teste_norm_md.columns).index(x_miss)
                        result_teste = AnalysisResults.gera_resultado_univa(
                            resposta=output_md_test,
                            dataset_normalizado_md=X_teste_norm_md,
                            dataset_normalizado_original=X_teste_norm,
                            missing_id=id_miss,
                            flag=False
                        )

                        tabela_resultados[
                            f'{model_impt}/{nome}-{x_miss}/{md}/{fold}/MAE'
                        ] = {'teste': round(result_teste,3)}
                        
                        # Dataset imputado                   
                        data_imputed = pd.DataFrame(output_md_test.copy(), columns=X.columns)
                        data_imputed['target'] = y_teste

                        data_imputed.to_csv(f"./Fairness/Datasets/{mecanismo}_Univariado/{nome}_{model_impt}_fold{fold}_md{md}_{x_miss}.csv", index=False)
                        fold += 1

        resultados_final = AnalysisResults.extrai_resultados(tabela_resultados)

        # Resultados da imputação
        resultados_mecanismo = (
            AnalysisResults.calcula_metricas_estatisticas_resultados(
                resultados_final, 1, fold
            )
        )
        resultados_mecanismo.to_csv(
            f'./Fairness/Resultados/{mecanismo}_Univariado/{model_impt}_{mecanismo}.csv', 
        )    
    return _logger.info(f"{model_impt} done!")

if __name__ == "__main__":

    diretorio = "/home/cruncher/Desktop/@MestradoArthur/DatasetsFairness"
    datasets = MyPipeline.carrega_datasets(diretorio)

    fairness = Fairness()
    tabela_resultados = fairness.cria_tabela_fairness(datasets)

    mecanismo = "MNAR"

    # Cria diretórios para salvar os resultados do experimento
    os.makedirs(f"./Fairness/Tempos/{mecanismo}_Univariado", exist_ok=True)
    os.makedirs(f"./Fairness/Datasets/{mecanismo}_Univariado", exist_ok=True)
    os.makedirs(f"./Fairness/Resultados/{mecanismo}_Univariado", exist_ok=True)

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:

        args_list = [#("mean",mecanismo,tabela_resultados),
                    #("customKNN",mecanismo,tabela_resultados),
                    #("mice",mecanismo,tabela_resultados),
                    ("pmivae",mecanismo,tabela_resultados),
                    #("saei",mecanismo,tabela_resultados),
                    #("softImpute",mecanismo,tabela_resultados),
                    ("gain",mecanismo,tabela_resultados)
                    ]
        
        pool.starmap(pipeline_fairness,args_list)
