
#pip install optuna-dashboard

import optuna

# Define the SQLite database to save the study
storage = "sqlite:///optuna_study.db"  # This will create a file called optuna_study.db in your current directory

# Create a study object with storage
study = optuna.create_study(study_name="survival_analysis_study", direction='maximize', storage=storage, load_if_exists=True)

# Optimize the objective function
study.optimize(objective, n_trials=50, timeout=3600)


optuna-dashboard sqlite:///optuna_study.db

#Optional: Running the Dashboard in Jupyter Notebook
!optuna-dashboard sqlite:///optuna_study.db --port 8080 --runserver
