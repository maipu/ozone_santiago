# Forecasting Ozone Pollution using Recurrent Neural Nets and Multiple Quantile Regression

## Folders and Files:
* **all_scores:** All scores obtained from the models in the same folder.
* **data:** Datasets used, containing only Summer data.
* **models:** Cache of the trained models.
* **model.backup:** When a loading is interrupted, the model try to train again, so if that new training is canceled, the cache will be overwrited, leaving slight variations in the scores.
* **precalcs:** Precalculate data from the dataset, this 'precalcs' are only for test and is not used in this study, but it's necessary to the properly work of the code.
* **tuning.done:** Tuning done with Hyperas. For each model have 3 files:
  * .py: Code with the configuration to start the tuning. the arguments <HyperLSTM | preSQP> < Number of runs >
    * i.e: file.py HyperLSTM 35
  * .temp.txt: The output of the corresponding .py
  * .txt: The tail of the corresponding .temp.txt
* **work:** This is the main folder. All .py files must be in this folder to work properly, also the ozone_forecasting_multi-task notebook.


## For all models:
* Optimizer: Adam.
* Learning Rate: $0.001$
* B<sub>1</sub>: $0.9$
* B<sub>2</sub>: $0.999$
