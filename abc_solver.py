import numpy as np
import random
from typing import List, Tuple
import pickle
import json
from datetime import datetime
from solution import MDMSMCVRPSolution
import time


class MDMSMCVRP_ABC:
    def __init__(self, problem, swarm_size=50, max_iterations=200,
                 employed_ratio=0.5, onlooker_ratio=0.3, limit=15, verbose=False):
        """Enhanced ABC solver for MDMS-MCVRP problem with route display"""
        self.problem = problem
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.employed_ratio = employed_ratio
        self.onlooker_ratio = onlooker_ratio
        self.limit = limit
        self.verbose = verbose  # ✅ NEW: Control route display

        # Core components
        self.swarm: List[MDMSMCVRPSolution] = []
        self.fitness_values: List[float] = []
        self.trials: List[int] = []
        self.best_solution: MDMSMCVRPSolution = None
        self.best_fitness = float('inf')
        self.convergence_curve: List[float] = []

        # Enhanced tracking
        self.improvements_count = 0
        self.improvement_iterations: List[int] = []
        self.best_solutions_history = []  # ✅ NEW: Store all best solutions

    def initialize_swarm(self):
        """Initialize the swarm with random solutions"""
        print(f"Initializing swarm with {self.swarm_size} solutions...")

        self.swarm = []
        for i in range(self.swarm_size):
            solution = MDMSMCVRPSolution(self.problem)
            solution.initialize_random()
            self.swarm.append(solution)

        self.fitness_values = [sol.fitness for sol in self.swarm]
        self.trials = [0] * self.swarm_size

        # Find best initial solution
        valid_solutions = [sol for sol in self.swarm if sol.fitness != float('inf')]
        if valid_solutions:
            self.best_solution = min(valid_solutions, key=lambda x: x.fitness)
            self.best_fitness = self.best_solution.fitness
        else:
            self.best_solution = self.swarm[0]
            self.best_fitness = self.swarm[0].fitness

        print(f"Initial best fitness: {self.best_fitness:.2f}")

        # ✅ NEW: Show initial routes if verbose
        if self.verbose:
            print("🎯 Initial Best Routes:")
            self.print_solution_routes(self.best_solution)

    def optimize(self) -> Tuple[MDMSMCVRPSolution, List[float]]:
        """Run the ABC optimization algorithm with enhanced route tracking"""
        self.initialize_swarm()

        print(f"Starting optimization with {self.max_iterations} iterations...")
        start_time = time.time()

        for iteration in range(self.max_iterations):
            # Print progress every 20 iterations
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {self.best_fitness:.2f}")

            prev_best = self.best_fitness

            # ABC phases
            self._employed_bee_phase()
            self._onlooker_bee_phase()
            self._scout_bee_phase()

            # Track convergence
            self.convergence_curve.append(self.best_fitness)

            # ✅ ENHANCED: Track improvements with route display
            if self.best_fitness < prev_best:
                self.improvements_count += 1
                self.improvement_iterations.append(iteration)
                print(f"  ✅ NEW BEST at iteration {iteration}: {self.best_fitness:.2f}")

                # Store this best solution in history
                self.best_solutions_history.append({
                    'iteration': iteration,
                    'fitness': self.best_fitness,
                    'solution': self.best_solution.copy(),
                    'timestamp': datetime.now().isoformat()
                })

                # ✅ NEW: Display routes when improvement found
                if self.verbose:
                    self.print_solution_routes(self.best_solution)

        total_time = time.time() - start_time
        print(f"\nOptimization completed in {total_time:.2f}s")
        print(f"Final best fitness: {self.best_fitness:.2f}")
        print(f"Total improvements: {self.improvements_count}")

        # ✅ NEW: Always show final best routes
        print(f"\n🏆 FINAL BEST SOLUTION ROUTES:")
        self.print_solution_routes(self.best_solution)

        return self.best_solution, self.convergence_curve

    def print_solution_routes(self, solution):
        """✅ NEW: Print detailed routes for the current solution"""
        if solution is None:
            print("    No solution to display!")
            return

        print(f"    🚛 Primary Vehicle Routes (Depots → Satellites):")
        for depot, route in solution.routes['E1'].items():
            if len(route) > 2:  # Only show non-empty routes
                satellites_count = len([s for s in route if s in self.problem.VS])
                route_str = ' → '.join(route)
                print(f"      Vehicle {depot}: {route_str} ({satellites_count} satellites)")

        print(f"    🚐 Secondary Vehicle Routes (Satellites → Customers):")
        for satellite, route in solution.routes['E2'].items():
            if len(route) > 2:  # Only show non-empty routes
                customers_count = len([c for c in route if c in self.problem.VC])
                route_str = ' → '.join(route)
                print(f"      Vehicle {satellite}: {route_str} ({customers_count} customers)")
        print()  # Empty line for readability

    def _employed_bee_phase(self):
        """Employed bees phase"""
        employed_count = int(self.employed_ratio * self.swarm_size)

        for i in range(employed_count):
            new_solution = self._modify_solution(self.swarm[i])
            new_fitness = new_solution.calculate_fitness()

            if new_fitness < self.fitness_values[i]:
                self.swarm[i] = new_solution
                self.fitness_values[i] = new_fitness
                self.trials[i] = 0

                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_fitness
            else:
                self.trials[i] += 1

    def _onlooker_bee_phase(self):
        """Onlooker bees phase"""
        employed_count = int(self.employed_ratio * self.swarm_size)
        onlooker_count = int(self.onlooker_ratio * self.swarm_size)

        # Calculate selection probabilities
        max_fit = max(self.fitness_values[:employed_count]) if self.fitness_values[:employed_count] else 1
        fitnesses = [max_fit - fit + 1 for fit in self.fitness_values[:employed_count]]
        total_fit = sum(fitnesses)

        if total_fit > 0:
            probabilities = [f / total_fit for f in fitnesses]

            for _ in range(onlooker_count):
                idx = np.random.choice(range(employed_count), p=probabilities)

                new_solution = self._modify_solution(self.swarm[idx])
                new_fitness = new_solution.calculate_fitness()

                if new_fitness < self.fitness_values[idx]:
                    self.swarm[idx] = new_solution
                    self.fitness_values[idx] = new_fitness
                    self.trials[idx] = 0

                    if new_fitness < self.best_fitness:
                        self.best_solution = new_solution.copy()
                        self.best_fitness = new_fitness
                else:
                    self.trials[idx] += 1

    def _scout_bee_phase(self):
        """Scout bees phase"""
        employed_count = int(self.employed_ratio * self.swarm_size)

        for i in range(employed_count):
            if self.trials[i] >= self.limit:
                new_solution = MDMSMCVRPSolution(self.problem)
                new_solution.initialize_random()

                self.swarm[i] = new_solution
                self.fitness_values[i] = new_solution.fitness
                self.trials[i] = 0

                if new_solution.fitness < self.best_fitness:
                    self.best_solution = new_solution.copy()
                    self.best_fitness = new_solution.fitness

    def _modify_solution(self, solution: MDMSMCVRPSolution) -> MDMSMCVRPSolution:
        """Apply simple modification operators"""
        new_solution = solution.copy()

        # Simple operators
        modification = random.choice([
            'swap_customers',
            'swap_satellites',
            'change_route_order'
        ])

        if modification == 'swap_customers' and len(self.problem.VS) > 1:
            k1, k2 = random.sample(self.problem.VS, 2)
            route1, route2 = new_solution.routes['E2'][k1], new_solution.routes['E2'][k2]

            if len(route1) > 2 and len(route2) > 2:
                i, j = random.randint(1, len(route1) - 2), random.randint(1, len(route2) - 2)
                route1[i], route2[j] = route2[j], route1[i]

        elif modification == 'swap_satellites':
            l = random.choice(self.problem.VD)
            route = new_solution.routes['E1'][l]

            if len(route) > 4:  # Has satellites to swap
                satellites = [i for i, node in enumerate(route) if node in self.problem.VS]
                if len(satellites) >= 2:
                    i, j = random.sample(satellites, 2)
                    route[i], route[j] = route[j], route[i]

        elif modification == 'change_route_order':
            k = random.choice(self.problem.VS)
            route = new_solution.routes['E2'][k]

            if len(route) > 4:  # Has customers to shuffle
                customers = [i for i, node in enumerate(route) if node in self.problem.VC]
                if len(customers) >= 2:
                    i, j = random.sample(customers, 2)
                    route[i], route[j] = route[j], route[i]

        new_solution._calculate_flows()
        return new_solution

    def plot_convergence(self, save_plot=True, show_plot=True):
        """Enhanced convergence plot with improvement markers"""
        if not self.convergence_curve:
            print("No convergence data available!")
            return

        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            plt.plot(self.convergence_curve, 'b-', linewidth=2, marker='o', markersize=3)
            plt.title('ABC Algorithm Convergence for MDMS-MCVRP', fontsize=14, fontweight='bold')
            plt.xlabel('Iteration', fontsize=12)
            plt.ylabel('Best Fitness', fontsize=12)
            plt.grid(True, alpha=0.3)

            # Add improvement points
            if self.improvement_iterations:
                improvement_fitness = [self.convergence_curve[i] for i in self.improvement_iterations]
                plt.scatter(self.improvement_iterations, improvement_fitness,
                            color='red', s=60, marker='*', label='Improvements', zorder=5)
                plt.legend()

            # Add best fitness annotation
            best_idx = self.convergence_curve.index(min(self.convergence_curve))
            best_fitness = min(self.convergence_curve)
            plt.annotate(f'Best: {best_fitness:.2f}\nIteration: {best_idx}',
                         xy=(best_idx, best_fitness),
                         xytext=(best_idx + len(self.convergence_curve) * 0.1,
                                 best_fitness + (max(self.convergence_curve) - min(self.convergence_curve)) * 0.1),
                         arrowprops=dict(arrowstyle='->', color='red'),
                         fontsize=10, color='red')

            plt.tight_layout()

            if save_plot:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'abc_convergence_{timestamp}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"✅ Convergence plot saved as '{filename}'")

            if show_plot:
                plt.show()

        except ImportError:
            print("❌ Matplotlib not available. Install with: pip install matplotlib")

    def print_best_solution(self):
        """Print comprehensive details of the best solution found"""
        if self.best_solution is None:
            print("No solution found!")
            return

        print(f"\n=== COMPREHENSIVE BEST SOLUTION ANALYSIS ===")
        print(f"🏆 Best Fitness: {self.best_fitness:.2f}")
        print(f"📈 Total Improvements: {self.improvements_count}")
        print(f"🔄 Improvement Iterations: {self.improvement_iterations}")

        # Enhanced route display
        self.print_solution_routes(self.best_solution)

        # Solution statistics if available
        try:
            stats = self.best_solution.get_route_statistics()
            print(f"📊 Solution Statistics:")
            print(f"   Customers Served: {stats['customers_served']}/{stats['total_customers']}")
            print(f"   Service Rate: {stats['service_rate']:.1%}")
        except AttributeError:
            print("📊 Basic solution metrics only available")

    def save_best_solution(self, filename_base='best_solution'):
        """✅ ENHANCED: Save the best solution in multiple formats"""
        if self.best_solution is None:
            print("No solution to save!")
            return

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save as pickle (original format)
        pickle_filename = f"{filename_base}_{timestamp}.pkl"
        with open(pickle_filename, 'wb') as f:
            pickle.dump(self.best_solution, f)
        print(f"✅ Solution saved as pickle: {pickle_filename}")

        # ✅ NEW: Save routes as JSON for easy inspection
        json_filename = f"{filename_base}_routes_{timestamp}.json"
        route_data = {
            'timestamp': datetime.now().isoformat(),
            'fitness': float(self.best_fitness),
            'improvements_count': self.improvements_count,
            'primary_routes': {},
            'secondary_routes': {},
            'optimization_summary': {
                'total_iterations': len(self.convergence_curve),
                'improvement_iterations': self.improvement_iterations,
                'final_fitness': float(self.best_fitness)
            }
        }

        # Extract route information
        for depot, route in self.best_solution.routes['E1'].items():
            satellites_count = len([s for s in route if s in self.problem.VS])
            route_data['primary_routes'][depot] = {
                'route': route,
                'satellites_count': satellites_count
            }

        for satellite, route in self.best_solution.routes['E2'].items():
            customers_count = len([c for c in route if c in self.problem.VC])
            route_data['secondary_routes'][satellite] = {
                'route': route,
                'customers_count': customers_count
            }

        with open(json_filename, 'w') as f:
            json.dump(route_data, f, indent=2)
        print(f"✅ Routes saved as JSON: {json_filename}")

        # ✅ NEW: Save optimization history
        history_filename = f"{filename_base}_history_{timestamp}.json"
        with open(history_filename, 'w') as f:
            # Convert solutions to route data for JSON serialization
            history_data = []
            for entry in self.best_solutions_history:
                history_entry = {
                    'iteration': entry['iteration'],
                    'fitness': entry['fitness'],
                    'timestamp': entry['timestamp'],
                    'routes': {
                        'E1': entry['solution'].routes['E1'],
                        'E2': entry['solution'].routes['E2']
                    }
                }
                history_data.append(history_entry)

            json.dump({
                'optimization_history': history_data,
                'convergence_curve': [float(x) for x in self.convergence_curve]
            }, f, indent=2)
        print(f"✅ Optimization history saved: {history_filename}")

    @staticmethod
    def load_solution(filename):
        """Load a previously saved solution"""
        try:
            with open(filename, 'rb') as f:
                solution = pickle.load(f)
            print(f"✅ Solution loaded from {filename}")
            return solution
        except Exception as e:
            print(f"❌ Error loading solution: {e}")
            return None
