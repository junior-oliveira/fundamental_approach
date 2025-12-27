import pandas as pd
import os
import subprocess
import logging
import threading

class Experimento():
    """
        Executa um conjunto de experimentos no MOA e retorna os resultados.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger('ExperimentoLogger')
        self.file_lock = threading.Lock()
    
    def _salvar_base_individual(self, df, modelo):
        """Salva os resultados de uma base individual imediatamente no CSV"""
        nome_arquivo = f'resultados/resultados_{modelo}.csv'
        
        with self.file_lock:
            try:
                              
                # Verifica se o arquivo existe e tem conteúdo
                arquivo_existe = os.path.exists(nome_arquivo) and os.path.getsize(nome_arquivo) > 0
                
                if arquivo_existe:
                    # Adiciona ao arquivo existente SEM cabeçalho
                    df.to_csv(nome_arquivo, mode='a', header=False, index=False)
                else:
                    # Cria um novo arquivo COM cabeçalho
                    df.to_csv(nome_arquivo, mode='w', header=True, index=False)
                    
                self.logger.debug(f"Base salva em {nome_arquivo}")
            except Exception as e:
                self.logger.error(f"Erro ao salvar base individual no CSV: {str(e)}")

    def executar(self, lag=1, l=350, k=450, v='WeightedVoteStrategy', w=4.0, evaluatorOption='Basic', model='ECS', atraso=27):
        """
            Executa um conjunto de experimentos.

            :param int lag: Dias de espera até começar a testar.
            :return: DataFrame contendo os resultados dos experimentos.
            :rtype: DataFrame
        """
        learners = {                    
                    'NB' :  f'bayes.NaiveBayes', 
                    'NB_PSR' : 'bayes.NaiveBayes',
                    'NB_S_VOL' : 'bayes.NaiveBayes',
                    'NB_LAG_27_profit3' : 'bayes.NaiveBayes',
                    'NB_LAG_60' : 'bayes.NaiveBayes',
                    'NB_inv_pred' :  f'bayes.NaiveBayes', 
                    'NBProfit' :  f'bayes.NaiveBayes', 
                    'NBProfit2' :  f'bayes.NaiveBayes', # Pode negociar a qualquer momento
                    'NBProfit3' :  f'bayes.NaiveBayes', # Pode negociar a qualquer momento
                    'HT' :  f'trees.HoeffdingTree',
                    'HT_S_VOL' :  f'trees.HoeffdingTree',
                    'HT_LAG_60' :  f'trees.HoeffdingTree',
                    'HT_PSR' :  f'trees.HoeffdingTree',
                    'HT_inv_pred' :  f'trees.HoeffdingTree',
                    'HTProfit' :  f'trees.HoeffdingTree',
                    'HTProfit2' :  f'trees.HoeffdingTree', # Pode negociar a qualquer momento
                    'HTProfit3' :  f'trees.HoeffdingTree', # Pode negociar a qualquer momento
                    'ECSHT2' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 290 -k 3 -e 0.28 -m Entropy -s 10',
                    'ECSHT3' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 290 -k 3 -e 0.4 -m Entropy -s 10',
                    'ECSHT4' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 290 -k 3 -e 0.5 -m Entropy -s 10',
                    'ECSHT5' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 290 -k 3 -e 0.6 -m Entropy -s 10',
                    'ECSHT6' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 100 -k 3 -e 0.5 -m Entropy -s 10',
                    'ECSHT7' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 400 -k 3 -e 0.5 -m Entropy -s 10',
                    'ECSHT8' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 400 -k 3 -e 0.5 -m Entropy -s 10 -v SingleVoteStrategy',
                    'ECSHTEspecialized' : f'meta.EntropyClassifierSelectionEspecialized -l trees.HoeffdingTree -j 40 -k 3 -e 0.5 -m Entropy -s 10',
                    'ECSHTSingle' : f'meta.EntropyClassifierSelection -l trees.HoeffdingTree -j 290 -k 30 -e 0.1 -m Entropy -s 10 -v SingleVoteStrategy',
                    'ECSHTSingle2' : f'meta.EntropyClassifierSelection -l bayes.NaiveBayes -j 290 -k 30 -e 0.28 -m Entropy -s 10 -v SingleVoteStrategy',
                    'BOLE' : f'meta.BOLE',
                    'OZABAG' : 'meta.OzaBag'
                    }
        detectores = {
            'N_DET' : 'moa.classifiers.rules.core.changedetection.NoChangeDetection',
            }

        if evaluatorOption == 'AUC':
            evaluator = 'BasicAUCImbalancedPerformanceEvaluatorTest -a'
        elif evaluatorOption == 'Basic':
            evaluator = 'BasicClassificationPerformanceEvaluator'


        cat_vol = ['fund', 'nao_fund']
        resultado = pd.DataFrame()
        
        for k in detectores:
            detector = detectores[k]
            res_modelo = pd.DataFrame()
            
            modelo = learners[model]
            self.logger.info(f"Modelo configurado: {model} | Configuração MOA: {modelo}")
            
            for cat in cat_vol:
                out_file = f'/home/jjos/fundamental_approach/_temp/temp_{lag}_{model}_{cat}_{atraso}.csv'
                base_path = f'/home/jjos/databases/b3/lags/atraso_{atraso}/{cat}/{lag}/'
                if cat == 'nao_fund':
                    base_path = f'/home/jjos/databases/b3/lags/atraso_{atraso}/{cat}/{lag}/'
                bases_names = os.listdir(f'{base_path}') 
                bases_names = [item for item in bases_names if len(item) == 10]
                
                total_bases = len(bases_names)
                self.logger.info(f"Processando {total_bases} bases para categoria: {cat}, lag: {lag}")

                # Percorre cada base de dados
                
                for idx, base_name in enumerate(bases_names, 1):
                    self.logger.info(f"[{idx}/{total_bases}] Iniciando base: {base_name} | "
                                   f"Modelo: {model} | Lag: {lag}")
                    
                    stream = f'(moa.streams.ArffFileStream -f ({base_path}{base_name}))' 
                    
                    # Define a porcentagem de treinamento
                    trainingPercentage = 1.0
                    
                    # Monta o argumento completo do DoTask
                    dotask_arg = f'EvaluatePrequentialUFPEProfit -l ({modelo}) -s {stream} -f 10000 -L {lag} -d ({out_file}) -e ({evaluator}) -T {trainingPercentage}'

                    self.logger.info(f'Comando DoTask: java -cp moa.jar moa.DoTask \ "{dotask_arg}"')
                    # Comando como lista de argumentos com caminho completo
                    cmd = ['java', '-cp', 'moa.jar', 'moa.DoTask', dotask_arg]
                    
                    self.logger.debug(f"Executando comando: {' '.join(cmd[:4])}...")

                    try:
                        # Executa o comando
                        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd='/home/jjos/fundamental_approach/bin/')

                        df = pd.read_csv(out_file)
                        df['categoria'] = cat
                        df['modelo'] = model
                        df['base'] = base_name.replace('.arff', '')
                        df['fh'] = lag
                        df['parametros'] = f'lag_{lag}_{model}_{cat}'
                        df['trainingPercentage'] = trainingPercentage

                        # Salvar imediatamente esta base no CSV do modelo
                        self._salvar_base_individual(df, f'{model}_atraso_{atraso}')
                        
                        res_modelo = pd.concat([res_modelo, df])
                        
                        # Remove o arquivo temporário
                        if os.path.exists(out_file):
                            os.remove(out_file)
                        
                        self.logger.info(f"[{idx}/{total_bases}] ✓ Base concluída: {base_name} | "
                                       f"{len(df)} linhas geradas | Salvo em resultados_{model}.csv")
                        
                    except subprocess.CalledProcessError as e:
                        error_msg = e.output.decode('utf-8') if hasattr(e.output, 'decode') else str(e.output)
                        self.logger.error(f"[{idx}/{total_bases}] ✗ Erro ao processar base: {base_name}")
                        self.logger.error(f"Saída do erro: {error_msg[:500]}")  # Primeiros 500 caracteres
                    except FileNotFoundError as e:
                        self.logger.error(f"[{idx}/{total_bases}] ✗ Arquivo não encontrado: {base_name} | "
                                        f"Erro: {str(e)}")
                    except Exception as e:
                        self.logger.error(f"[{idx}/{total_bases}] ✗ Erro inesperado na base: {base_name} | "
                                        f"Erro: {str(e)}")
                        
            resultado = pd.concat([resultado, res_modelo])
            
        self.logger.info(f"✓ Todas as {total_bases} bases processadas para modelo: {model}, lag: {lag}")
        
        return resultado
    
if __name__ == '__main__':
    # Configurar logging básico para teste
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    experimento = Experimento()
    resultado = experimento.executar(lag=1, l=350, k=450, v='WeightedVoteStrategy', 
                                     w=4.0, evaluatorOption='Basic', model='ECS')
    print(resultado)