import optuna

storage = "mysql://root:zm6148mz@34.134.200.45/mzlearn_webapp_DB"
study_name = 'distributed-example'


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True)

    study.optimize(objective, n_trials=100)