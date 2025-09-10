import optuna
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from optuna.samplers import (
    TPESampler,
    RandomSampler,
    CmaEsSampler,
    GridSampler,
    NSGAIISampler,
    QMCSampler
)
from abc_solver import MDMSMCVRP_ABC
from problem import MDMSMCVRPProblem


class OptimizationTracker:
    """Enhanced tracking with statistical analysis and visualization"""

    def __init__(self):
        self.trial_results = []
        self.best_params = None
        self.best_fitness = float('inf')
        self.start_time = None
        self.convergence_data = []

    def log_trial(self, trial_number: int, params: Dict, fitness: float,
                  execution_time: float, additional_metrics: Dict = None):
        """Enhanced trial logging with additional metrics"""
        result = {
            'trial': trial_number,
            'params': params,
            'fitness': fitness,
            'execution_time': execution_time,
            'timestamp': datetime.now().isoformat(),
            'additional_metrics': additional_metrics or {}
        }
        self.trial_results.append(result)

        if fitness < self.best_fitness:
            improvement = self.best_fitness - fitness
            self.best_fitness = fitness
            self.best_params = params.copy()
            print(f"  ✅ New best found in trial {trial_number}: {fitness:.2f} "
                  f"(improvement: {improvement:.2f})")

    def get_summary(self) -> Dict:
        """Enhanced summary with statistical analysis"""
        if not self.trial_results:
            return {}

        fitnesses = [r['fitness'] for r in self.trial_results if r['fitness'] != float('inf')]
        times = [r['execution_time'] for r in self.trial_results]

        summary = {
            'total_trials': len(self.trial_results),
            'valid_trials': len(fitnesses),
            'best_fitness': self.best_fitness,
            'best_params': self.best_params,
            'fitness_stats': {
                'mean': np.mean(fitnesses) if fitnesses else float('inf'),
                'std': np.std(fitnesses) if len(fitnesses) > 1 else 0,
                'min': np.min(fitnesses) if fitnesses else float('inf'),
                'max': np.max(fitnesses) if fitnesses else float('inf'),
                'median': np.median(fitnesses) if fitnesses else float('inf'),
                'q25': np.percentile(fitnesses, 25) if fitnesses else float('inf'),
                'q75': np.percentile(fitnesses, 75) if fitnesses else float('inf')
            },
            'time_stats': {
                'mean': np.mean(times),
                'std': np.std(times),
                'total': np.sum(times),
                'min': np.min(times),
                'max': np.max(times)
            }
        }

        # Calculate improvement rate
        if len(fitnesses) > 1:
            summary['improvement_rate'] = (fitnesses[0] - fitnesses[-1]) / fitnesses[0] * 100

        return summary

    def plot_optimization_progress(self, save_plot: bool = True):
        """Plot optimization progress"""
        if not self.trial_results:
            print("No trial data to plot!")
            return

        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. Fitness over trials
            trials = [r['trial'] for r in self.trial_results]
            fitnesses = [r['fitness'] for r in self.trial_results if r['fitness'] != float('inf')]
            valid_trials = [r['trial'] for r in self.trial_results if r['fitness'] != float('inf')]

            ax1.plot(valid_trials, fitnesses, 'b-', alpha=0.7)
            ax1.scatter(valid_trials, fitnesses, c='blue', s=20, alpha=0.6)
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('Fitness Value')
            ax1.set_title('Fitness Progress Over Trials')
            ax1.grid(True, alpha=0.3)

            # Highlight best trial
            if self.best_fitness != float('inf'):
                best_trial = next(r['trial'] for r in self.trial_results if r['fitness'] == self.best_fitness)
                ax1.scatter([best_trial], [self.best_fitness], c='red', s=100, marker='*',
                            label=f'Best: {self.best_fitness:.2f}')
                ax1.legend()

            # 2. Execution time distribution
            times = [r['execution_time'] for r in self.trial_results]
            ax2.hist(times, bins=min(20, len(times)), alpha=0.7, color='green')
            ax2.set_xlabel('Execution Time (seconds)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Execution Time Distribution')
            ax2.grid(True, alpha=0.3)

            # 3. Parameter correlation heatmap (if enough data) - FIXED
            if len(self.trial_results) > 5:
                param_data = []
                for result in self.trial_results:
                    if result['fitness'] != float('inf'):
                        row = list(result['params'].values()) + [result['fitness']]
                        param_data.append(row)

                if param_data:  # ✅ FIXED - correct condition
                    param_names = list(self.trial_results[0]['params'].keys()) + ['fitness']
                    correlation_matrix = np.corrcoef(np.array(param_data).T)

                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                                xticklabels=param_names, yticklabels=param_names, ax=ax3)
                    ax3.set_title('Parameter Correlation Matrix')

            # 4. Fitness distribution
            if fitnesses:
                ax4.hist(fitnesses, bins=min(20, len(fitnesses)), alpha=0.7, color='orange')
                ax4.axvline(self.best_fitness, color='red', linestyle='--',
                            label=f'Best: {self.best_fitness:.2f}')
                ax4.set_xlabel('Fitness Value')
                ax4.set_ylabel('Frequency')
                ax4.set_title('Fitness Distribution')
                ax4.legend()
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_plot:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'optimization_analysis_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✅ Optimization analysis saved as '{filename}'")

            plt.show()

        except ImportError:
            print("❌ Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")
        except Exception as e:
            print(f"Error creating plots: {e}")


def get_search_spaces(mode: str = "comprehensive") -> Dict:
    """Enhanced search spaces optimized for our upgraded algorithm"""

    if mode == "quick":
        return {
            # "swarm_size": [30, 50, 80],
            # "max_iterations": [100, 200, 300],
            # "employed_ratio": [0.4, 0.5, 0.6],
            # "onlooker_ratio": [0.2, 0.3, 0.4],
            # "limit": [10, 15, 20]

            "swarm_size": [40, 60, 80, 100, 120],  # 5 values
            "max_iterations": [250, 350, 450, 550],  # 4 values
            "employed_ratio": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6],  # 6 values
            "onlooker_ratio": [0.2, 0.25, 0.3, 0.35, 0.4],  # 5 values
            "limit": [20, 25, 30, 35]  # 4 values

        # Total combinations: 5 × 4 × 6 × 5 × 4 = 2,400 combinations
        }

    elif mode == "comprehensive":
        return {
            # "swarm_size": [50, 80, 100, 150],
            # "max_iterations": [ 100, 200, 300],
            # "employed_ratio": [0.3,  0.5, 0.7],
            # "onlooker_ratio": [0.15, 0.2, 0.4, 0.5],
            # "limit": [ 18, 20, 25]

            "swarm_size": [40, 50, 60, 70, 80, 90, 100, 120, 150],  # 9 values
            "max_iterations": [200, 250, 300, 350, 400, 450, 500, 600, 700],  # 9 values
            "employed_ratio": [0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7],  # 10 values
            "onlooker_ratio": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],  # 9 values
            "limit": [10, 15, 20, 25, 30, 35, 40, 45, 50]  # 9 values

        # Total combinations: 9 × 9 × 10 × 9 × 9 = 65,610 combinations

        }

    elif mode == "exhaustive":
        # MAXIMUM coverage for research-grade optimization (very long runtime)
        return {
            "swarm_size": [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 150, 180, 200],  # 14 values
            "max_iterations": [150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 800, 1000],  # 14 values
            "employed_ratio": [0.2, 0.25, 0.3, 0.32, 0.35, 0.37, 0.4, 0.42, 0.45, 0.47, 0.5, 0.52, 0.55, 0.57, 0.6,
                               0.62, 0.65, 0.7],  # 18 values
            "onlooker_ratio": [0.05, 0.1, 0.12, 0.15, 0.17, 0.2, 0.22, 0.25, 0.27, 0.3, 0.32, 0.35, 0.37, 0.4, 0.42,
                               0.45, 0.47, 0.5],  # 18 values
            "limit": [5, 8, 10, 12, 15, 17, 20, 22, 25, 27, 30, 32, 35, 37, 40, 42, 45, 50, 60]  # 19 values
        }
        # Total combinations: 14 × 14 × 18 × 18 × 19 = 1,205,064 combinations


    elif mode == "fine_tune":
        return {
            # "swarm_size": [ 70, 80],
            # "max_iterations": [150, 200],
            # "employed_ratio": [0.4, 0.6],
            # "onlooker_ratio": [0.2, 0.25, 0.4],
            # "limit": [10, 20]

            "swarm_size": [65, 70, 75, 80, 85, 90, 95, 100, 105, 110],  # 10 values
            "max_iterations": [320, 350, 380, 400, 420, 450, 480, 500, 520, 550],  # 10 values
            "employed_ratio": [0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58],  # 9 values
            "onlooker_ratio": [0.22, 0.25, 0.27, 0.3, 0.32, 0.35, 0.37, 0.4],  # 8 values
            "limit": [22, 25, 27, 30, 32, 35, 37, 40]  # 8 values

        # Total combinations: 10 × 10 × 9 × 8 × 8 = 57,600 combinations

        }

    elif mode == "ultra_comprehensive":
        # RESEARCH-GRADE exhaustive search with maximum precision
        return {
            "swarm_size": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 110, 120, 130, 150, 170,
                           200],  # 22 values
            "max_iterations": [100, 150, 200, 220, 250, 280, 300, 320, 350, 380, 400, 420, 450, 480, 500, 520, 550, 600,
                               650, 700, 800, 1000],  # 22 values
            "employed_ratio": [0.15, 0.2, 0.22, 0.25, 0.27, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48,
                               0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7],  # 26 values
            "onlooker_ratio": [0.05, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34,
                               0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5],  # 23 values
            "limit": [5, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 55, 60]
            # 25 values
        }
        # Total combinations: 22 × 22 × 26 × 23 × 25 = 7,210,600 combinations

    else:  # default balanced configuration
        return {
            "swarm_size": [50],
            "max_iterations": [200],
            "employed_ratio": [0.5],
            "onlooker_ratio": [0.3],
            "limit": [15]
        }


def validate_parameters(params: Dict) -> Tuple[bool, List[str]]:
    """Enhanced parameter validation with detailed error messages"""
    errors = []

    # Check ratio constraints
    total_ratio = params['employed_ratio'] + params['onlooker_ratio']
    if total_ratio > 1.0:
        errors.append(f"employed_ratio + onlooker_ratio = {total_ratio:.2f} > 1.0")

    if total_ratio < 0.5:
        errors.append(f"employed_ratio + onlooker_ratio = {total_ratio:.2f} < 0.5 (may be inefficient)")

    # Check parameter ranges
    if not (10 <= params['swarm_size'] <= 500):
        errors.append(f"swarm_size = {params['swarm_size']} not in range [10, 500]")

    if not (20 <= params['max_iterations'] <= 1000):
        errors.append(f"max_iterations = {params['max_iterations']} not in range [20, 1000]")

    if not (0.1 <= params['employed_ratio'] <= 0.9):
        errors.append(f"employed_ratio = {params['employed_ratio']} not in range [0.1, 0.9]")

    if not (0.1 <= params['onlooker_ratio'] <= 0.9):
        errors.append(f"onlooker_ratio = {params['onlooker_ratio']} not in range [0.1, 0.9]")

    if not (1 <= params['limit'] <= 100):
        errors.append(f"limit = {params['limit']} not in range [1, 100]")

    # Check for efficient parameter combinations
    if params['limit'] > params['swarm_size']:
        errors.append(f"limit ({params['limit']}) > swarm_size ({params['swarm_size']}) may reduce exploration")

    return len(errors) == 0, errors


def run_multiple_trials(problem: MDMSMCVRPProblem, params: Dict, n_runs: int = 3) -> Tuple[
    float, float, List[float], Dict]:
    """Enhanced multiple trials with detailed metrics collection"""
    results = []
    additional_metrics = {
        'convergence_curves': [],
        'improvement_counts': [],
        'final_feasibility': []
    }

    print(f"    Running {n_runs} trials with parameters: {params}")

    for run in range(n_runs):
        try:
            # ✅ FIXED - removed verbose parameter
            abc = MDMSMCVRP_ABC(
                problem=problem,
                **params
            )

            best_solution, convergence_curve = abc.optimize()

            # Collect metrics
            results.append(best_solution.fitness)
            additional_metrics['convergence_curves'].append(convergence_curve)
            additional_metrics['improvement_counts'].append(abc.improvements_count)
            additional_metrics['final_feasibility'].append(best_solution.is_feasible)

            print(f"      Run {run + 1}: {best_solution.fitness:.2f}")

        except Exception as e:
            print(f"      Run {run + 1} failed: {e}")
            results.append(float('inf'))

    # Filter out infinite values
    valid_results = [r for r in results if r != float('inf')]

    if not valid_results:
        return float('inf'), float('inf'), results, additional_metrics

    mean_fitness = np.mean(valid_results)
    std_fitness = np.std(valid_results) if len(valid_results) > 1 else 0.0

    # Calculate additional statistics
    additional_metrics['success_rate'] = len(valid_results) / len(results)
    additional_metrics['mean_improvements'] = np.mean(additional_metrics['improvement_counts'])
    additional_metrics['feasibility_rate'] = np.mean(additional_metrics['final_feasibility'])

    return mean_fitness, std_fitness, results, additional_metrics


def optimize_with_optuna(problem: MDMSMCVRPProblem,
                         n_trials: int = 50,
                         sampler_name: str = "grid", # replace with it 'tpe' for bayesian ooptimization
                         search_mode: str = "quick", # change here for quick , comprehensive etc.
                         n_runs_per_trial: int = 1,
                         save_results: bool = True,
                         study_name: Optional[str] = None,
                         timeout: Optional[int] = None) -> Dict:
    """
    Enhanced Optuna optimization with comprehensive analysis and early stopping.

    Args:
        problem: MDMSMCVRPProblem instance
        n_trials: Number of optimization trials
        sampler_name: Sampler to use ('tpe', 'random', 'cmaes', 'grid', 'nsga2', 'qmc')
        search_mode: Search space mode ('quick', 'comprehensive', 'fine_tune', 'default')
        n_runs_per_trial: Number of runs per parameter combination (for statistical validation)
        save_results: Whether to save results to file
        study_name: Optional study name for tracking
        timeout: Optional timeout in seconds

    Returns:
        Dictionary containing best parameters and comprehensive optimization summary
    """

    print(f"\n{'=' * 60}")
    print(f"ENHANCED OPTUNA HYPERPARAMETER OPTIMIZATION")
    print(f"{'=' * 60}")
    print(f"Sampler: {sampler_name.upper()}")
    print(f"Search Mode: {search_mode}")
    print(f"Trials: {n_trials}")
    print(f"Runs per trial: {n_runs_per_trial}")
    if timeout:
        print(f"Timeout: {timeout}s")

    tracker = OptimizationTracker()
    tracker.start_time = time.time()

    # Get search space
    search_space = get_search_spaces(search_mode)

    def objective(trial):
        trial_start = time.time()

        try:
            # ✅ FIXED - improved parameter generation to avoid invalid ratios
            if sampler_name.lower() == "grid":
                params = {
                    'swarm_size': trial.suggest_categorical('swarm_size', search_space['swarm_size']),
                    'max_iterations': trial.suggest_categorical('max_iterations', search_space['max_iterations']),
                    'employed_ratio': trial.suggest_categorical('employed_ratio', search_space['employed_ratio']),
                    'onlooker_ratio': trial.suggest_categorical('onlooker_ratio', search_space['onlooker_ratio']),
                    'limit': trial.suggest_categorical('limit', search_space['limit'])
                }
            else:
                # ✅ FIXED - Smart parameter generation to ensure employed + onlooker <= 1.0
                employed_ratio = trial.suggest_float('employed_ratio', 0.3, 0.7, step=0.05)
                max_onlooker = min(0.6, 0.95 - employed_ratio)  # Ensure total ≤ 0.95
                onlooker_ratio = trial.suggest_float('onlooker_ratio', 0.1, max_onlooker, step=0.05)

                params = {
                    'swarm_size': trial.suggest_int('swarm_size', 30, 150, step=10),
                    'max_iterations': trial.suggest_int('max_iterations', 100, 400, step=25),
                    'employed_ratio': employed_ratio,
                    'onlooker_ratio': onlooker_ratio,
                    'limit': trial.suggest_int('limit', 5, 25, step=2)
                }

            # Validate parameters
            is_valid, errors = validate_parameters(params)
            if not is_valid:
                print(f"  ❌ Invalid parameters in trial {trial.number}: {errors[0]}")
                return float('inf')

            # Run multiple trials for statistical validation
            if n_runs_per_trial > 1:
                mean_fitness, std_fitness, all_results, additional_metrics = run_multiple_trials(
                    problem, params, n_runs_per_trial)

                # Store additional metrics
                trial.set_user_attr("std_fitness", std_fitness)
                trial.set_user_attr("all_results", all_results)
                trial.set_user_attr("success_rate", additional_metrics['success_rate'])
                trial.set_user_attr("mean_improvements", additional_metrics['mean_improvements'])
                trial.set_user_attr("feasibility_rate", additional_metrics['feasibility_rate'])

                fitness = mean_fitness
            else:
                # ✅ FIXED - removed verbose parameter
                abc = MDMSMCVRP_ABC(problem=problem, **params)
                best_solution, convergence_curve = abc.optimize()
                fitness = best_solution.fitness

                additional_metrics = {
                    'improvements_count': abc.improvements_count,
                    'is_feasible': best_solution.is_feasible,
                    'convergence_curve': convergence_curve
                }

            # Log trial results
            trial_time = time.time() - trial_start
            tracker.log_trial(trial.number, params, fitness, trial_time, additional_metrics)

            # Print progress with more details
            print(f"Trial {trial.number + 1}/{n_trials} - Fitness: {fitness:.2f} - "
                  f"Time: {trial_time:.1f}s - Valid: {fitness != float('inf')}")

            return fitness

        except Exception as e:
            print(f"❌ Error in trial {trial.number}: {e}")
            return float('inf')

    # Configure sampler
    samplers = {
        "random": RandomSampler(seed=42),
        "cmaes": CmaEsSampler(seed=42),
        "grid": GridSampler(search_space) if search_mode != "default" else GridSampler(get_search_spaces("quick")),
        "nsga2": NSGAIISampler(seed=42),
        "qmc": QMCSampler(seed=42),
        "tpe": TPESampler(seed=42, n_startup_trials=10, n_ei_candidates=24)
    }

    sampler = samplers.get(sampler_name.lower(), GridSampler(get_search_spaces("quick")))  # ✅ Changed fallback

    #sampler = samplers.get(sampler_name.lower(), TPESampler(seed=42))

    # Adjust n_trials for grid search
    if sampler_name.lower() == "grid":
        total_combinations = np.prod([len(values) for values in search_space.values()])
        n_trials = min(n_trials, total_combinations)
        print(f"Grid Search: Testing up to {total_combinations} parameter combinations")

    # Create and run study
    study_name = study_name or f"mdms_mcvrp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name
        )

        # Run optimization with optional timeout
        if timeout:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        # Get results
        if study.trials:
            best_params = study.best_params
            best_fitness = study.best_value
        else:
            print("❌ No successful trials completed!")
            return {}

        total_time = time.time() - tracker.start_time

        print(f"\n{'=' * 60}")
        print(f"OPTIMIZATION COMPLETED")
        print(f"{'=' * 60}")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Best fitness: {best_fitness:.2f}")
        print(f"Best parameters:")
        for key, value in best_params.items():
            print(f"  {key:>15}: {value}")

        # Prepare comprehensive results
        results = {
            'study_name': study_name,
            'optimization_summary': {
                'sampler': sampler_name,
                'search_mode': search_mode,
                'n_trials': n_trials,
                'n_runs_per_trial': n_runs_per_trial,
                'total_time': total_time,
                'best_fitness': float(best_fitness),
                'best_params': best_params,
                'timeout_used': timeout is not None
            },
            'trial_history': tracker.trial_results,
            'statistics': tracker.get_summary(),
            'study_summary': {
                'n_trials': len(study.trials),
                'n_complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'n_failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                'n_pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
            }
        }

        # Add parameter importance analysis if available
        try:
            if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) > 10:
                importance = optuna.importance.get_param_importances(study)
                results['parameter_importance'] = importance

                print(f"\nParameter Importance:")
                for param, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {param:>15}: {imp:.4f}")
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")

        # Generate visualization
        print(f"\nGenerating optimization analysis plots...")
        tracker.plot_optimization_progress(save_results)

        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"optuna_results_{timestamp}.json"
            try:
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                print(f"✅ Results saved to {filename}")
            except Exception as e:
                print(f"❌ Error saving results: {e}")

        return best_params

    except Exception as e:
        print(f"❌ Error in Optuna optimization: {e}")
        return {}


def quick_parameter_search(problem: MDMSMCVRPProblem, save_results: bool = True) -> Dict:
    """Enhanced quick parameter search with statistical validation"""

    print(f"\n{'=' * 60}")
    print(f"ENHANCED QUICK PARAMETER SEARCH")
    print(f"{'=' * 60}")

    # Enhanced configurations based on our upgraded algorithm
    configurations = [
        {'name': 'Balanced Standard',
         'params': {'swarm_size': 50, 'max_iterations': 200, 'employed_ratio': 0.5, 'onlooker_ratio': 0.3,
                    'limit': 15}},
        {'name': 'High Exploration',
         'params': {'swarm_size': 80, 'max_iterations': 300, 'employed_ratio': 0.6, 'onlooker_ratio': 0.2,
                    'limit': 20}},
        {'name': 'High Exploitation',
         'params': {'swarm_size': 40, 'max_iterations': 150, 'employed_ratio': 0.4, 'onlooker_ratio': 0.4,
                    'limit': 10}},
        {'name': 'Large Scale',
         'params': {'swarm_size': 100, 'max_iterations': 400, 'employed_ratio': 0.45, 'onlooker_ratio': 0.35,
                    'limit': 25}},
        {'name': 'Fast Convergence',
         'params': {'swarm_size': 30, 'max_iterations': 100, 'employed_ratio': 0.3, 'onlooker_ratio': 0.5, 'limit': 8}},
        {'name': 'Adaptive Focus',  # New configuration
         'params': {'swarm_size': 60, 'max_iterations': 250, 'employed_ratio': 0.45, 'onlooker_ratio': 0.35,
                    'limit': 18}}
    ]

    best_config = None
    best_fitness = float('inf')
    results = {}

    for config in configurations:
        print(f"\n🔄 Testing configuration: {config['name']}")
        start_time = time.time()

        try:
            mean_fitness, std_fitness, all_results, additional_metrics = run_multiple_trials(
                problem, config['params'], n_runs=3)
            execution_time = time.time() - start_time

            results[config['name']] = {
                'params': config['params'],
                'mean_fitness': mean_fitness,
                'std_fitness': std_fitness,
                'all_results': all_results,
                'execution_time': execution_time,
                'success_rate': additional_metrics['success_rate'],
                'mean_improvements': additional_metrics['mean_improvements'],
                'feasibility_rate': additional_metrics['feasibility_rate']
            }

            print(f"  📊 Results: Mean={mean_fitness:.2f} (±{std_fitness:.2f}), "
                  f"Success={additional_metrics['success_rate']:.1%}, "
                  f"Feasible={additional_metrics['feasibility_rate']:.1%}")
            print(f"  ⏱️ Time: {execution_time:.2f}s")

            if mean_fitness < best_fitness:
                best_fitness = mean_fitness
                best_config = config['params'].copy()
                print(f"  ✅ New best configuration!")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[config['name']] = {'error': str(e)}

    print(f"\n{'=' * 60}")
    print(f"QUICK SEARCH RESULTS")
    print(f"{'=' * 60}")
    print(f"🏆 Best configuration fitness: {best_fitness:.2f}")
    print(f"📋 Best parameters:")
    if best_config:
        for key, value in best_config.items():
            print(f"  {key:>15}: {value}")

    # Create comparison visualization
    try:
        valid_configs = {name: data for name, data in results.items() if 'mean_fitness' in data}

        if valid_configs:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Fitness comparison
            names = list(valid_configs.keys())
            fitnesses = [data['mean_fitness'] for data in valid_configs.values()]
            stds = [data['std_fitness'] for data in valid_configs.values()]

            bars = ax1.bar(range(len(names)), fitnesses, yerr=stds, capsize=5, alpha=0.7)
            ax1.set_xlabel('Configuration')
            ax1.set_ylabel('Mean Fitness')
            ax1.set_title('Configuration Comparison')
            ax1.set_xticks(range(len(names)))
            ax1.set_xticklabels(names, rotation=45, ha='right')
            ax1.grid(True, alpha=0.3)

            # Highlight best configuration
            best_idx = fitnesses.index(min(fitnesses))
            bars[best_idx].set_color('gold')

            # Success rate comparison
            success_rates = [data['success_rate'] * 100 for data in valid_configs.values()]
            ax2.bar(range(len(names)), success_rates, alpha=0.7, color='green')
            ax2.set_xlabel('Configuration')
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Success Rate Comparison')
            ax2.set_xticks(range(len(names)))
            ax2.set_xticklabels(names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                plot_filename = f'quick_search_comparison_{timestamp}.png'
                plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"✅ Comparison plot saved as '{plot_filename}'")

            plt.show()

    except ImportError:
        print("❌ Matplotlib not available for visualization")
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")

    # Save results
    if save_results:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"quick_search_results_{timestamp}.json"
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✅ Results saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

    return best_config


# Enhanced convenience functions
def optimize_for_quality(problem: MDMSMCVRPProblem, n_trials: int = 100, timeout: int = 3600) -> Dict:
    """Optimize for solution quality with statistical validation"""
    return optimize_with_optuna(
        problem=problem,
        n_trials=n_trials,
        sampler_name="tpe", # replace it with 'tpe' for bayesian optimization
        search_mode="comprehensive",
        n_runs_per_trial=3,
        timeout=timeout
    )


def optimize_for_speed(problem: MDMSMCVRPProblem, n_trials: int = 20, timeout: int = 600) -> Dict:
    """Optimize for quick results"""
    return optimize_with_optuna(
        problem=problem,
        n_trials=n_trials,
        sampler_name="random",
        search_mode="quick",
        n_runs_per_trial=1,
        timeout=timeout
    )


def fine_tune_parameters(problem: MDMSMCVRPProblem, base_params: Dict, n_trials: int = 30) -> Dict:
    """Fine-tune around known good parameters"""
    print(f"Fine-tuning around base parameters: {base_params}")
    return optimize_with_optuna(
        problem=problem,
        n_trials=n_trials,
        sampler_name="grid", # replace it with 'tpe' for bayesian optimization
        search_mode="fine_tune",
        n_runs_per_trial=2
    )


def compare_samplers(problem: MDMSMCVRPProblem, n_trials: int = 30) -> Dict:
    """Compare different Optuna samplers"""
    samplers_to_test = ["tpe", "random", "cmaes", "qmc"]
    results = {}

    print(f"\n{'=' * 60}")
    print(f"SAMPLER COMPARISON")
    print(f"{'=' * 60}")

    for sampler in samplers_to_test:
        print(f"\n🔄 Testing {sampler.upper()} sampler...")
        try:
            best_params = optimize_with_optuna(
                problem=problem,
                n_trials=n_trials,
                sampler_name=sampler,
                search_mode="quick",
                n_runs_per_trial=1,
                save_results=False
            )
            results[sampler] = best_params
            print(f"✅ {sampler.upper()} completed successfully")
        except Exception as e:
            print(f"❌ {sampler.upper()} failed: {e}")
            results[sampler] = None

    return results
