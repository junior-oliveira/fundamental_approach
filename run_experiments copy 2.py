import os
from experiment import Experiment 
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import time
import threading
import logging
from datetime import datetime

# Lock to prevent concurrency issues
file_lock = threading.Lock()
log_lock = threading.Lock()

# Setup logging
def setup_logging():
    """Configures the logging system with file and console output"""
    log_filename = f'logs/experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s | PID:%(process)d | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger('ExperimentLogger')
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

def log_status(logger, status, model, lag, evaluator, delay, dataset=None, extra_message=""):
    """
    Logs the status of the experiment
    
    status: STARTING, EXECUTING, COMPLETED, ERROR
    """
    with log_lock:
        msg_parts = [
            f"[{status}]",
            f"Model: {model}",
            f"Lag: {lag}",
            f"Evaluator: {evaluator}",
            f"Delay: {delay}"
        ]
        
        if dataset:
            msg_parts.append(f"Dataset: {dataset}")
        
        if extra_message:
            msg_parts.append(f"| {extra_message}")
        
        logger.info(" | ".join(msg_parts))

def save_result(result, model, lag, evaluator, delay, logger):
    """Saves the result in a specific CSV file for the model"""
    filename = f'results_{model}.csv'
    
    try:
        with file_lock:
            if os.path.exists(filename):
                result.to_csv(filename, mode='a', header=False, index=False)
            else:
                result.to_csv(filename, mode='w', header=True, index=False)
        
        num_rows = len(result)
        log_status(logger, "SAVED", model, lag, evaluator, delay, 
                   extra_message=f"{num_rows} rows saved in {filename}")
        return True
    except Exception as e:
        log_status(logger, "SAVE_ERROR", model, lag, evaluator, delay, 
                   extra_message=f"Error saving: {str(e)}")
        return False

def task(args):
    # Unpacking updated arguments
    lag, evaluator, model, delay = args
    
    # Create logger for this process
    logger = logging.getLogger('ExperimentLogger')
    
    log_status(logger, "STARTING", model, lag, evaluator, delay)
    
    try:
        experiment = Experiment()
        
        # Log before execution
        log_status(logger, "EXECUTING", model, lag, evaluator, delay,
                   extra_message="Starting MOA execution")
        
        task_start = time.time()
        # The datasets will be saved individually inside the execute() method
        result = experiment.execute(lag=lag, evaluator_option=evaluator, model=model, delay=delay)
        elapsed_time = time.time() - task_start
        
        # Log after execution
        log_status(logger, "COMPLETED", model, lag, evaluator, delay,
                   extra_message=f"Time: {elapsed_time/60:.2f} min | {len(result)} total rows")
        
        return result
        
    except Exception as e:
        log_status(logger, "ERROR", model, lag, evaluator, delay,
                   extra_message=f"Exception: {str(e)}")
        raise

def create_status_file(arguments, log_filename):
    """Creates a file with a list of all experiments to execute"""
    status_file = f'results/experiment_status_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
    
    # Ensure the results directory exists
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    with open(status_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERIMENT STATUS\n")
        f.write(f"Log file: {log_filename}\n")
        f.write(f"Total experiments: {len(arguments)}\n")
        f.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENTS TO EXECUTE:\n")
        f.write("-"*80 + "\n")
        
        for i, (lag, evaluator, model, delay) in enumerate(arguments, 1):
            f.write(f"{i:3d}. Model: {model:15s} | Lag: {lag:3d} | Delay: {delay:2d} | "
                   f"Evaluator={evaluator}\n")
    
    return status_file

if __name__ == '__main__':
    
    # Ensure directories exist
    os.makedirs('results', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    # Setup logging
    logger, log_filename = setup_logging()
    logger.info("="*80)
    logger.info("STARTING EXPERIMENTS")
    logger.info("="*80)
    
    start_time = time.time()
    config = {
        'lag' : [1, 30, 60, 90, 120, 150, 180, 210, 240],
        'evaluator' : ['AUC'],
        'delays' : [1],
        'models' : ['HT']
    }

    # Generating all combinations (Removed l, k, w, v)
    arguments = list(product(config['lag'], config['evaluator'], config['models'], config['delays']))
    
    logger.info(f"Total experiments: {len(arguments)}")
    logger.info(f"Available CPUs: {os.cpu_count()}")
    logger.info(f"Parallel workers: {os.cpu_count() - 1}")
    logger.info("-"*80)
    
    # Create status file
    status_file = create_status_file(arguments, log_filename)
    logger.info(f"Status file created: {status_file}")
    logger.info("-"*80)

    # Execute in parallel
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = {executor.submit(task, args): args for args in arguments}
        
        completed_results = []
        failed_experiments = []
        
        for i, future in enumerate(as_completed(futures), 1):
            args = futures[future]
            lag = args[0]
            model = args[2] 
            
            try:
                result = future.result()
                completed_results.append(result)
                logger.info(f"✓ Progress: {i}/{len(arguments)} experiments completed "
                           f"({(i/len(arguments)*100):.1f}%)")
            except Exception as e:
                failed_experiments.append((args, str(e)))
                logger.error(f"✗ Experiment {i}/{len(arguments)} FAILED: {args}")

    # Consolidate results
    if completed_results:
        logger.info("-"*80)
        logger.info("Consolidating final results...")
        final_df = pd.concat(completed_results, ignore_index=True)
        final_file = f'results/consolidated_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        final_df.to_csv(final_file, index=False)
        logger.info(f"✓ Consolidated file saved: {final_file}")

    # Final summary
    end_time = time.time()
    total_time = (end_time - start_time) / 60
    
    logger.info("="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Total time: {total_time:.2f} minutes ({total_time/60:.2f} hours)")
    logger.info(f"Successful experiments: {len(completed_results)}/{len(arguments)}")
    logger.info(f"Failed experiments: {len(failed_experiments)}/{len(arguments)}")
    
    if failed_experiments:
        logger.info("-"*80)
        logger.info("FAILED EXPERIMENTS:")
        for args, error in failed_experiments:
            logger.error(f"  - {args}: {error}")
    
    logger.info("="*80)
    logger.info("FINISHED")
    logger.info("="*80)