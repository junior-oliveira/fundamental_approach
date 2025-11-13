import os
from Experimento2 import Experimento
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import time
import threading
import logging
from datetime import datetime

# Lock para evitar problemas de concorrência
file_lock = threading.Lock()
log_lock = threading.Lock()

# Configurar logging
def setup_logging():
    """Configura o sistema de logging com arquivo e console"""
    log_filename = f'logs/experimento_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Criar formatador
    formatter = logging.Formatter(
        '%(asctime)s | PID:%(process)d | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Handler para arquivo
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Handler para console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configurar logger
    logger = logging.getLogger('ExperimentoLogger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def log_status(logger, status, modelo, lag, l, k, w, v, e, base=None, mensagem_extra=""):
    """
    Registra o status do experimento
    
    status: INICIANDO, PROCESSANDO, CONCLUIDO, ERRO
    """
    with log_lock:
        msg_parts = [
            f"[{status}]",
            f"Modelo: {modelo}",
            f"Lag: {lag}",
            f"Params: l={l}, k={k}, w={w}, v={v}, e={e}"
        ]
        
        if base:
            msg_parts.append(f"Base: {base}")
        
        if mensagem_extra:
            msg_parts.append(f"| {mensagem_extra}")
        
        logger.info(" | ".join(msg_parts))

def salvar_resultado(resultado, modelo, lag, l, k, w, v, e, evaluator, logger):
    """Salva o resultado em um arquivo CSV específico do modelo"""
    nome_arquivo = f'resultados_{modelo}.csv'
    
    try:
        with file_lock:
            if os.path.exists(nome_arquivo):
                resultado.to_csv(nome_arquivo, mode='a', header=False, index=False)
            else:
                resultado.to_csv(nome_arquivo, mode='w', header=True, index=False)
        
        num_linhas = len(resultado)
        log_status(logger, "SALVO", modelo, lag, l, k, w, v, e, 
                   mensagem_extra=f"{num_linhas} linhas salvas em {nome_arquivo}")
        return True
    except Exception as e:
        log_status(logger, "ERRO_SALVAR", modelo, lag, l, k, w, v, e, 
                   mensagem_extra=f"Erro ao salvar: {str(e)}")
        return False

def tarefa(args):
    l, k, w, v, lag, e, evaluator, modelo = args
    
    # Criar logger para este processo
    logger = logging.getLogger('ExperimentoLogger')
    
    log_status(logger, "INICIANDO", modelo, lag, l, k, w, v, e)
    
    try:
        experimento = Experimento()
        
        # Log antes de executar
        log_status(logger, "EXECUTANDO", modelo, lag, l, k, w, v, e, 
                   mensagem_extra="Iniciando otimização com algoritmo evolucionário")
        
        inicio_tarefa = time.time()
        # As bases serão salvas individualmente dentro do método executar()
        resultado = experimento.executar(lag, l, k, v, w, e, evaluator, modelo)
        tempo_decorrido = time.time() - inicio_tarefa
        
        # Log após execução
        log_status(logger, "CONCLUIDO", modelo, lag, l, k, w, v, e, 
                   mensagem_extra=f"Tempo: {tempo_decorrido/60:.2f} min | {len(resultado)} linhas totais")
        
        return resultado
        
    except Exception as e:
        log_status(logger, "ERRO", modelo, lag, l, k, w, v, e, 
                   mensagem_extra=f"Exceção: {str(e)}")
        raise

def criar_arquivo_status(argumentos, log_filename):
    """Cria arquivo com lista de todos os experimentos a executar"""
    status_file = f'resultados/status_experimentos_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    with open(status_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("STATUS DOS EXPERIMENTOS\n")
        f.write(f"Arquivo de log: {log_filename}\n")
        f.write(f"Total de experimentos: {len(argumentos)}\n")
        f.write(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENTOS A EXECUTAR:\n")
        f.write("-"*80 + "\n")
        
        for i, (l, k, w, v, lag, e, evaluator, modelo) in enumerate(argumentos, 1):
            f.write(f"{i:3d}. Modelo: {modelo:15s} | Lag: {lag:3d} | "
                   f"l={l:3d}, k={k}, w={w}, v={v}, e={e}\n")
    
    return status_file

if __name__ == '__main__':
    
    # Configurar logging
    logger, log_filename = setup_logging()
    logger.info("="*80)
    logger.info("INICIANDO EXPERIMENTOS")
    logger.info("="*80)
    
    inicio = time.time()
    config = {
        'l' : [280],
        'k' : [2],
        'w' : [4.0],
        'e' : ['0.28'],
        'v' : ['WeightedVoteStrategy'],
        'lag' : [1, 30, 60, 90, 120, 150, 180, 210, 240],
        'evaluator' : ['AUC'],
        # 'modelos' : ['ECS', 'AWE', 'BLAST', 'BOLE', 'DACC', 'DWM', 'OZABAG', 
        #              'OZABAG_ADWIN', 'RCD', 'PLkNN', 'ADOB', 'OzaBoost', 'LeveragingBag']
        'modelos' : ['NB']
    }

    # Gerando todas as combinações
    argumentos = list(product(config['l'], config['k'], config['w'], config['v'], 
                             config['lag'], config['e'], config['evaluator'], config['modelos']))
    
    logger.info(f"Total de experimentos: {len(argumentos)}")
    logger.info(f"CPUs disponíveis: {os.cpu_count()}")
    logger.info(f"Workers paralelos: {os.cpu_count() - 1}")
    logger.info("-"*80)
    
    # Criar arquivo de status
    status_file = criar_arquivo_status(argumentos, log_filename)
    logger.info(f"Arquivo de status criado: {status_file}")
    logger.info("-"*80)

    # Executar em paralelo
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = {executor.submit(tarefa, args): args for args in argumentos}
        
        resultados_completos = []
        experimentos_com_erro = []
        
        for i, future in enumerate(as_completed(futures), 1):
            args = futures[future]
            modelo = args[7]
            lag = args[4]
            
            try:
                resultado = future.result()
                resultados_completos.append(resultado)
                logger.info(f"✓ Progresso: {i}/{len(argumentos)} experimentos concluídos "
                           f"({(i/len(argumentos)*100):.1f}%)")
            except Exception as e:
                experimentos_com_erro.append((args, str(e)))
                logger.error(f"✗ Experimento {i}/{len(argumentos)} FALHOU: {args}")

    # Consolidar resultados
    if resultados_completos:
        logger.info("-"*80)
        logger.info("Consolidando resultados finais...")
        df_final = pd.concat(resultados_completos, ignore_index=True)
        arquivo_final = f'resultados/resultados_consolidados_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df_final.to_csv(arquivo_final, index=False)
        logger.info(f"✓ Arquivo consolidado salvo: {arquivo_final}")

    # Resumo final
    fim = time.time()
    tempo_total = (fim - inicio) / 60
    
    logger.info("="*80)
    logger.info("RESUMO FINAL")
    logger.info("="*80)
    logger.info(f"Tempo total: {tempo_total:.2f} minutos ({tempo_total/60:.2f} horas)")
    logger.info(f"Experimentos bem-sucedidos: {len(resultados_completos)}/{len(argumentos)}")
    logger.info(f"Experimentos com erro: {len(experimentos_com_erro)}/{len(argumentos)}")
    
    if experimentos_com_erro:
        logger.info("-"*80)
        logger.info("EXPERIMENTOS COM ERRO:")
        for args, erro in experimentos_com_erro:
            logger.error(f"  - {args}: {erro}")
    
    logger.info("="*80)
    logger.info("FINALIZADO")
    logger.info("="*80)