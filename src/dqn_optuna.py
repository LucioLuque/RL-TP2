import os
import json
import numpy as np
import torch
import gymnasium as gym
import optuna

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate,
)

from utils_dqn import train_dqn

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.use_deterministic_algorithms(True)

def make_objective(
    env_id: str,
    experiment_folder: str,
    seed: int = 42,
    fixed: dict | None = None,
    all_trials_path: str = "all_trials.jsonl",
    converged_trials_path: str = "converged_trials.jsonl",
):
    fixed = fixed or {}

    def objective(trial: optuna.Trial) -> float:
        hparams = {
            "episodes": fixed.get("episodes", 200),

            "buffer_size": fixed.get("buffer_size", 15000),

            "max_steps": fixed.get("max_steps", 500),

            "gamma": fixed.get("gamma", 0.99),

            "lr": fixed.get("lr", trial.suggest_float("lr", 1e-4, 1e-2, log=True)),

            "target_update_freq": trial.suggest_categorical("target_update_freq", [100, 200, 500]),

            "min_epsilon": trial.suggest_categorical("min_epsilon", [0.0, 0.005, 0.01, 0.02]),

            "max_epsilon": fixed.get("max_epsilon", 1.0),

            "decay_rate": trial.suggest_float("decay_rate", 0.005, 0.03, log=True),

            "batch_size": fixed.get("batch_size", 64)
        }

        env = gym.make(env_id)

        try:

            model_path, model_params, training_stats = train_dqn(
                experiment_folder=experiment_folder,
                env=env,
                seed=seed,
                save=False,
                log_q_values=False,
                trial=trial,
                **hparams,
            )

        except optuna.exceptions.TrialPruned:

            raise

        except Exception as e:

            print(f"Trial {trial.number} falló: {e}")

            failed_trial = {
                "trial_number": trial.number,
                "status": "failed",
                "error": str(e),
                "hparams": hparams,
            }

            with open(all_trials_path, "a") as f:
                f.write(json.dumps(failed_trial) + "\n")

            return -np.inf

        finally:
            env.close()

        final_eval = training_stats["final_eval_reward"]
        best_eval = training_stats["best_eval_reward"]
        final_eval_std = training_stats["final_eval_std"]

        first_solved_episode = training_stats["first_solved_episode"]

        if first_solved_episode is None:
            score = -np.inf
        else:
            speed_score = max(0.0, 160 - first_solved_episode)
            score = (0.6 * final_eval + 0.4 * best_eval - 0.3 * final_eval_std + 0.5 * speed_score)

        trial_data = {
            "trial_number": trial.number,
            "status": "completed",
            "final_moving_avg": float(final_eval),
            "best_moving_avg": float(best_eval),
            "final_std": float(final_eval_std),
            "score": float(score),
            "hparams": hparams,
        }

        with open(all_trials_path, "a") as f:
            f.write(json.dumps(trial_data) + "\n")

        if (
            best_eval >= 495
            and final_eval_std <= 10
        ):

            converged_data = {
                "trial_number": trial.number,
                "score": float(score),
                "final_moving_avg": float(final_eval),
                "best_moving_avg": float(best_eval),
                "final_std": float(final_eval_std),
                "hparams": hparams,
            }

            with open(converged_trials_path, "a") as f:
                f.write(json.dumps(converged_data) + "\n")

        print(
            f"Trial {trial.number} | "
            f"Score={score:.2f} | "
            f"FinalAvg={final_eval:.2f} | "
            f"Std={final_eval_std:.2f}"
        )

        return score


    return objective

def hyperparameter_search(
    env_id: str,
    experiment_folder: str,

    n_trials: int = 50,

    seed: int = 42,
    fixed: dict | None = None,

    study_name: str | None = None,

    storage: str | None = "sqlite:///optuna_dqn.db",

    n_jobs: int = 1,
    show_plots: bool = True,

    results_path: str = "hparam_results.json",

    all_trials_path: str = "all_trials.jsonl",
    converged_trials_path: str = "converged_trials.jsonl"
    ) -> optuna.Study:

    study_name = study_name or f"dqn_{env_id.lower().replace('-', '_')}"

    sampler = TPESampler(seed=seed)

    pruner = MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=40,
        interval_steps=10,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(
        env_id=env_id,
        experiment_folder=experiment_folder,
        seed=seed,
        fixed=fixed,
        all_trials_path=all_trials_path,
        converged_trials_path = converged_trials_path,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
    )

    print("\n" + "=" * 60)
    print(f"  Mejores hiperparámetros ({study_name})")
    print("=" * 60)

    for k, v in study.best_params.items():
        print(f"  {k:25s}: {v}")

    print("=" * 60 + "\n")

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
            print(
                "No se pudieron generar los plots "
                f"(instala plotly): {e}"
            )

    return study