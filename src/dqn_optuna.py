import os
import json
import numpy as np
import torch
import gymnasium as gym
import optuna
import os

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

from utils_dqn import train_dqn, evaluate_dqn_model


def make_objective(
    env_id: str,
    experiment_folder: str,
    eval_episodes: int = 50,
    seed: int = 42,
    fixed: dict | None = None,
    successful_trials_path: str = "successful_trials.jsonl",
):
    
    fixed = fixed or {}

    def objective(trial: optuna.Trial) -> float:
        hparams = {
            "episodes":          fixed.get("episodes", 200),
            "buffer_size":       fixed.get("buffer_size",       trial.suggest_int("buffer_size", 1000, 50_000, log=True)),
            "max_steps":         fixed.get("max_steps", 500),
            "gamma":             fixed.get("gamma",             trial.suggest_float("gamma", 0.90, 0.999)),
            "lr":                fixed.get("lr",                trial.suggest_float("lr", 1e-4, 1e-2, log=True)),
            "target_update_freq":fixed.get("target_update_freq",trial.suggest_int("target_update_freq", 50, 700, step=50)),
            "min_epsilon":       fixed.get("min_epsilon",       trial.suggest_float("min_epsilon", 0.01, 0.10)),
            "max_epsilon":       fixed.get("max_epsilon",       1.0),
            "decay_rate":        fixed.get("decay_rate",        trial.suggest_float("decay_rate", 1e-3, 1e-1, log=True)),
            "batch_size":        fixed.get("batch_size",        trial.suggest_categorical("batch_size", [32, 64, 128])),
        }

        # Carpeta propia para cada trial, así no se pisan los modelos
        trial_folder = os.path.join(experiment_folder, f"trial_{trial.number}")
        os.makedirs(trial_folder, exist_ok=True)

        env = gym.make(env_id)

        try:
            model_path, model_params = train_dqn(
                experiment_folder=trial_folder,
                env=env,
                seed=seed,
                save=False,
                log_q_values=False,
                **hparams,
            )

            eval_env = gym.make(env_id)
            success_rate, mean_reward = evaluate_dqn_model(
                path=model_path,
                env=eval_env,
                model_params=model_params,
                episodes=eval_episodes,
                seed=seed,
            )
            eval_env.close()

        except Exception as e:
            print(f"Trial {trial.number} falló: {e}")
            return -np.inf
        finally:
            env.close()

        trial.report(success_rate, step=hparams["episodes"])

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        # Guardamos información útil dentro del trial
        trial.set_user_attr("success_rate", success_rate)
        trial.set_user_attr("mean_reward", mean_reward)
        trial.set_user_attr("model_path", model_path)
        trial.set_user_attr("hparams", hparams)

        # Guardar inmediatamente todos los que convergieron
        if success_rate == 1.0:
            successful_trial = {
                "trial_number": trial.number,
                "success_rate": success_rate,
                "mean_reward": mean_reward,
                "model_path": model_path,
                "hparams": hparams,
            }

            with open(successful_trials_path, "a") as f:
                f.write(json.dumps(successful_trial, indent=None) + "\n")

            print(f"Trial exitoso guardado: {trial.number}")

        if success_rate < 1.0:
            return -np.inf

        return success_rate

    return objective

def hyperparameter_search(
    env_id: str,
    experiment_folder: str,
    n_trials: int = 50,
    eval_episodes: int = 50,
    seed: int = 42,
    fixed: dict | None = None,
    study_name: str | None = None,
    storage: str | None = None,         # ej. "sqlite:///optuna.db" para persistencia
    n_jobs: int = 1,                    # paralelismo (cuidado con envs no thread-safe)
    show_plots: bool = True,
    results_path: str = "hparam_results.json",
    successful_trials_path: str = "successful_trials.jsonl",
) -> optuna.Study:
    """
    Busca hiperparámetros óptimos para DQN con Optuna.

    Returns
    -------
    study : optuna.Study
        Objeto con todos los trials y el mejor resultado.

    Ejemplo de uso
    --------------
    >>> study = hyperparameter_search(
    ...     env_id="CartPole-v1",
    ...     experiment_folder="cartpole_search",
    ...     n_trials=30,
    ...     fixed={"max_steps": 500, "max_epsilon": 1.0},
    ... )
    >>> print(study.best_params)
    """
    study_name = study_name or f"dqn_{env_id.lower().replace('-', '_')}"

    sampler = TPESampler(seed=seed)           # Bayesiano eficiente (Tree-structured Parzen Estimator)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,              # reanuda si ya existe en storage
    )

    objective = make_objective(
        env_id=env_id,
        experiment_folder=experiment_folder,
        eval_episodes=eval_episodes,
        seed=seed,
        fixed=fixed,
        successful_trials_path=successful_trials_path,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    print("\n" + "="*60)
    print(f"  Mejores hiperparámetros ({study_name})")
    print("="*60)
    print(f"  Valor (success_rate): {study.best_value:.4f}")
    for k, v in study.best_params.items():
        print(f"  {k:25s}: {v}")
    print("="*60 + "\n")

    results = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "env_id": env_id,
    }
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en: {results_path}")

    if show_plots:
        try:
            plot_optimization_history(study).show()
            plot_param_importances(study).show()
            plot_parallel_coordinate(study).show()
        except Exception as e:
            print(f"No se pudieron generar los plots: {e}")

    return study

# if __name__ == "__main__":
#     study = hyperparameter_search(
#         env_id="CartPole-v1",
#         experiment_folder="cartpole_optuna",
#         n_trials=30,
#         eval_episodes=50,
#         seed=42,
#         fixed={
#             "max_steps": 500,
#             "max_epsilon": 1.0,
#         },
#             # storage="sqlite:///optuna_dqn.db",  # descomentá para persistencia entre runs
#         show_plots=True,
#     )