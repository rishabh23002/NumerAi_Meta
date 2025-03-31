**1. Download Dataset**
    To download the dataset run the pre-installs.py using the command ```python pre-installs.py```

**2. Train Model**
    For training the meta model run the command ```python train.py```

**3. Generating Predictions**
    For generating the predictions.csv on the live data use the command ```python predict.py```


**4. Validation**
    For validating the model on the validation.parquet use the command ```python validation.py```
    
The Validation file will generate the Result graphs and a board.txt file in the Experiment_name folder from the config file.
The generated graph(s) shows the correlation between the predicted values and the targets same can be seen in the board.txt. 

**5. Autosubmission**
For generating the model.pkl file to upload on the numerai for autosubmissions run the command ```python live.py```