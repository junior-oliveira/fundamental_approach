import pandas as pd
import os
import subprocess
import logging
import threading

class Experiment():
    """
    Executes a set of prequential evaluation experiments in MOA and returns the results.
    """
    
    def __init__(self) -> None:
        self.logger = logging.getLogger('ExperimentLogger')
        self.file_lock = threading.Lock()
    
    def _save_individual_dataset(self, df, model):
        """Saves the results of an individual dataset immediately to a CSV file."""
        filename = f'results/results_{model}.csv'
        
        with self.file_lock:
            try:
                file_exists = os.path.exists(filename) and os.path.getsize(filename) > 0
                
                if file_exists:
                    df.to_csv(filename, mode='a', header=False, index=False)
                else:
                    df.to_csv(filename, mode='w', header=True, index=False)
                    
                self.logger.debug(f"Dataset saved to {filename}")
            except Exception as e:
                self.logger.error(f"Error saving individual dataset to CSV: {str(e)}")

    def execute(self, horizon=1, evaluator_option='Basic', model='HT', lag=1):
        """
        Executes a set of experiments across the specified forecast horizons.

        :param int horizon: The forecast horizon representing the verification latency step.
        :param str evaluator_option: Metric evaluator choice (e.g., 'AUC', 'Basic').
        :param str model: The selected learner model name.
        :param int lag: The lag configuration for the data streams.
        :return: DataFrame containing the empirical experiment results.
        :rtype: pd.DataFrame
        """
        learners = {                    
            'HT' : 'trees.HoeffdingTree',
            'ARF' : 'meta.AdaptiveRandomForest',
        }

        if evaluator_option == 'AUC':
            evaluator = 'BasicAUCImbalancedPerformanceEvaluatorUFPE -a'
        elif evaluator_option == 'Basic':
            evaluator = 'BasicClassificationPerformanceEvaluator'

        # Updated categories reflecting the correct scientific taxonomy
        categories = ['fundamental', 'technical']
        final_result = pd.DataFrame()
        
        moa_learner = learners[model]
        self.logger.info(f"Model configured: {model} | MOA Configuration: {moa_learner}")
        
        temp_dir = f'_temp/'

        # Ensure the temporary directory exists before any I/O operations
        os.makedirs(temp_dir, exist_ok=True)
        
        current_script_dir = os.path.dirname(os.path.abspath(__file__))

        for cat in categories:

            out_file = os.path.join(current_script_dir, f'_temp/temp_{horizon}_{model}_{cat}_{lag}.csv')

            database_root = os.path.join(current_script_dir, 'databases', 'lags')             

            base_path = f'../databases/lags/lag_{lag}/{cat}/{horizon}/'

            relative_base_path = f'lag_{lag}/{cat}/{horizon}/'
            base_path = os.path.join(database_root, relative_base_path)

            
            dataset_names = os.listdir(f'{base_path}') 
            dataset_names = [item for item in dataset_names if len(item) == 10]
            
            total_datasets = len(dataset_names)
            self.logger.info(f"Processing {total_datasets} datasets for category: {cat}, horizon: {horizon}")

            for idx, dataset_name in enumerate(dataset_names, 1):
                self.logger.info(f"[{idx}/{total_datasets}] Starting dataset: {dataset_name} | "
                               f"Model: {model} | Horizon: {horizon}")
                
                stream = f'(moa.streams.ArffFileStream -f ({base_path}{dataset_name}))' 
                
                training_percentage = 1.0
                
                # The -L parameter in the custom EvaluatePrequentialUFPEDelayed task receives the horizon size
                dotask_arg = f'EvaluatePrequentialUFPEDelayed -l ({moa_learner}) -s {stream} -f 10000 -L {horizon} -d ({out_file}) -e ({evaluator})'

                self.logger.info(f'DoTask Command: java -cp moa.jar moa.DoTask \ "{dotask_arg}"')
                cmd = ['java', '-cp', 'moa.jar', 'moa.DoTask', dotask_arg]
                
                self.logger.debug(f"Executing command: {' '.join(cmd[:4])}...")

                try:
                    if os.path.exists(out_file):
                        os.remove(out_file)
                    
                    # Implementation Note Context: Execution strictly depends on the bin/ directory containing MOA 2021.0
                    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, cwd='/home/jjos/fundamental_approach/bin/')

                    df = pd.read_csv(out_file)
                    df['category'] = cat
                    df['model'] = model
                    df['dataset'] = dataset_name.replace('.arff', '')
                    df['horizon'] = horizon
                    df['parameters'] = f'horizon_{horizon}_{model}_{cat}'
                    df['trainingPercentage'] = training_percentage

                    self._save_individual_dataset(df, f'{model}_lag_{lag}')
                    
                    final_result = pd.concat([final_result, df])
                    
                    if os.path.exists(out_file):
                        os.remove(out_file)
                    
                    self.logger.info(f"[{idx}/{total_datasets}] ✓ Dataset completed: {dataset_name} | "
                                   f"{len(df)} rows generated | Saved in results_{model}.csv")
                    
                except subprocess.CalledProcessError as e:
                    error_msg = e.output.decode('utf-8') if hasattr(e.output, 'decode') else str(e.output)
                    self.logger.error(f"[{idx}/{total_datasets}] ✗ Error processing dataset: {dataset_name}")
                    self.logger.error(f"Error output: {error_msg[:500]}")  
                except FileNotFoundError as e:
                    self.logger.error(f"[{idx}/{total_datasets}] ✗ File not found: {dataset_name} | "
                                    f"Error: {str(e)}")
                except Exception as e:
                    self.logger.error(f"[{idx}/{total_datasets}] ✗ Unexpected error in dataset: {dataset_name} | "
                                    f"Error: {str(e)}")
                    
        self.logger.info(f"✓ All {total_datasets * 2} datasets processed for model: {model}, horizon: {horizon}") 
        
        return final_result
    
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s'
    )
    
    experiment = Experiment()
    result = experiment.execute(horizon=1, evaluator_option='Basic', model='HT', lag=1)
    print(result)