import torch
from catalyst import dl
from torch.utils.data import DataLoader
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from SegmentationCustomRunner import SegmentationCustomRunner

class Experiment:
    def __init__(self, init_config):
        self.name = init_config['name']
        self.log_dir = f'./logs_{self.name}'

        self.model = init_config['model']

        self.runner_config = init_config['runner_config']

        self.metrics_to_plot = init_config['metrics_to_plot']

        self.eval_config = init_config['eval_config']

    def __init_runner(self):
        self.runner = SegmentationCustomRunner(input_key="image", output_key="mask_pred", target_key="mask_target", loss_key="loss")
    
    def __plot_and_save_graphs(self):
        df_train_hist = pd.read_csv(os.path.join(self.log_dir, 'csv_logger', 'train.csv'))
        df_valid_hist = pd.read_csv(os.path.join(self.log_dir, 'csv_logger', 'valid.csv'))

        metrics_num = len(self.metrics_to_plot)
        n_rows = metrics_num // 3
        n_cols = 3
        fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 8))
        for i in range(metrics_num):
            key = self.metrics_to_plot[i]
            ax[i].set_title(key)
            ax[i].plot(df_train_hist['step'], df_train_hist[key], label='Train', color='r')
            ax[i].plot(df_valid_hist['step'], df_valid_hist[key], label='Valid', color='g')
            ax[i].grid()
            ax[i].legend(loc="upper right")
        plt.show()
        fig.savefig(os.path.join(f'{self.log_dir}', 'results.png'))

    def __validate_after_train(self):
        print("Best model metrics:")
        print("-"*50)
        metrics = self.runner.evaluate_loader(
            loader=self.runner_config["loaders"]["valid"],
            callbacks=self.eval_config["callbacks"]
        )
        #print(metrics)

    def run_experiment(self):
        model = self.model
        self.__init_runner()
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        torch.cuda.empty_cache()
        self.runner.train(
            model=model,
            logdir=self.log_dir,
            **self.runner_config
        )
        self.__plot_and_save_graphs()
        self.__validate_after_train()

        
        