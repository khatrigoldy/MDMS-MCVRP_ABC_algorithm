from collections import defaultdict
import random
import json
import copy
from typing import Dict, List, Optional


# Add this function at the module level (outside the class)
def create_nested_defaultdict():
    """Create nested defaultdict for flows - pickle-friendly"""
    return defaultdict(dict)


class MDMSMCVRPSolution:
    def __init__(self, problem):
        self.problem = problem
        self.routes = {
            'E1': {l: [l, l] for l in problem.VD},
            'E2': {k: [k, k] for k in problem.VS}
        }
        # Fix: Replace lambda with named function
        self.flows = {
            'Y': defaultdict(dict),
            'Z': defaultdict(create_nested_defaultdict)  # ✅ Pickle-friendly
        }
        self.fitness = float('inf')
        self.is_feasible = False
        self.constraint_violations = {}

    def initialize_random(self):
        """Initialize a capacity-aware feasible solution"""
        try:
            # Reset routes and flows
            self.routes = {
                'E1': {l: [l, l] for l in self.problem.VD},
                'E2': {k: [k, k] for k in self.problem.VS}
            }
            self.flows = {
                'Y': defaultdict(dict),
                'Z': defaultdict(create_nested_defaultdict)
            }

            # IMPROVED: Assign satellites to primary vehicles more intelligently
            for l in self.problem.VD:
                route = self.routes['E1'][l]
                # Only assign 1-2 satellites per depot to avoid overloading
                available_satellites = self.problem.VS.copy()
                random.shuffle(available_satellites)

                # Assign 1-2 satellites (not all satellites to every depot)
                num_satellites = random.choice([1, 2]) if len(available_satellites) > 1 else 1
                selected_satellites = available_satellites[:num_satellites]

                for s in selected_satellites:
                    route.insert(-1, s)

            # IMPROVED: Capacity-aware customer assignment
            self._assign_customers_with_capacity_constraints()

            # Validate and repair if needed
            self.validate_and_repair_solution()

            self._calculate_flows()
            self.calculate_fitness()

        except Exception as e:
            print(f"Error in initialize_random: {e}")
            self.fitness = float('inf')

    def _assign_customers_with_capacity_constraints(self):
        """Assign customers to satellites respecting capacity constraints"""
        customers = self.problem.VC.copy()
        random.shuffle(customers)

        # Track satellite capacities
        satellite_loads = {k: 0 for k in self.problem.VS}

        # Calculate capacity for each satellite
        for k in self.problem.VS:
            satellite_loads[k] = 0

        unassigned_customers = []

        for customer in customers:
            # Calculate customer's total demand
            customer_demand = sum(self.problem.demands.get(customer, {}).get(l, 0)
                                  for l in self.problem.VD)

            # Find satellite with sufficient capacity
            best_satellite = None
            min_load = float('inf')

            for k in self.problem.VS:
                available_capacity = self.problem.W[k] - satellite_loads[k]
                if available_capacity >= customer_demand and satellite_loads[k] < min_load:
                    min_load = satellite_loads[k]
                    best_satellite = k

            if best_satellite:
                # Assign customer to satellite
                route = self.routes['E2'][best_satellite]
                insert_pos = random.randint(1, len(route) - 1)
                route.insert(insert_pos, customer)
                satellite_loads[best_satellite] += customer_demand
            else:
                # If no satellite has capacity, assign to least loaded
                least_loaded = min(self.problem.VS, key=lambda x: satellite_loads[x])
                route = self.routes['E2'][least_loaded]
                route.insert(-1, customer)
                satellite_loads[least_loaded] += customer_demand
                unassigned_customers.append(customer)

        if unassigned_customers:
            print(f"Warning: {len(unassigned_customers)} customers assigned despite capacity constraints")

    def _calculate_flows(self):
        """Enhanced flow calculation with capacity constraints"""
        try:
            # Reset flows
            self.flows = {
                'Y': defaultdict(dict),
                'Z': defaultdict(create_nested_defaultdict)
            }

            # Calculate satellite demands with capacity limits
            satellite_demands = {}
            for j in self.problem.VS:
                satellite_demands[j] = {}
                for l in self.problem.VD:
                    if j in self.routes['E2']:
                        customers_at_satellite = [c for c in self.routes['E2'][j]
                                                  if c in self.problem.VC]
                        total_demand = sum(self.problem.demands.get(c, {}).get(l, 0)
                                           for c in customers_at_satellite)

                        # Limit demand to satellite capacity
                        max_satellite_demand = self.problem.W[j] / len(self.problem.VD)
                        satellite_demands[j][l] = min(total_demand, max_satellite_demand)
                    else:
                        satellite_demands[j][l] = 0

            # Calculate primary flows with improved capacity constraints
            for l in self.problem.VD:
                if l not in self.routes['E1']:
                    continue

                route = self.routes['E1'][l]
                remaining_capacity = self.problem.P[l]

                for i in range(1, len(route)):
                    if i >= len(route):
                        break

                    prev_node = route[i - 1]
                    current_node = route[i]

                    if current_node in self.problem.VS:
                        required_flow = satellite_demands.get(current_node, {}).get(l, 0)
                        actual_flow = min(required_flow, remaining_capacity)

                        if actual_flow > 0:
                            self.flows['Y'][(prev_node, current_node)][l] = actual_flow
                            remaining_capacity = max(0, remaining_capacity - actual_flow)

            # Calculate secondary flows with compartment constraints
            for k in self.problem.VS:
                if k not in self.routes['E2']:
                    continue

                route = self.routes['E2'][k]
                compartment_loads = {l: 0 for l in self.problem.VD}

                for i in range(1, len(route)):
                    if i >= len(route):
                        break

                    prev_node = route[i - 1]
                    current_node = route[i]

                    if current_node in self.problem.VC:
                        for l in self.problem.VD:
                            demand = self.problem.demands.get(current_node, {}).get(l, 0)
                            available_capacity = self.problem.Q[k][l] - compartment_loads[l]

                            actual_flow = min(demand, available_capacity)
                            if actual_flow > 0:
                                self.flows['Z'][(prev_node, current_node)][k][l] = actual_flow
                                compartment_loads[l] += actual_flow

        except Exception as e:
            print(f"Error in _calculate_flows: {e}")

    def copy(self):
        """Safe copy method without deepcopy issues - FIXED"""
        try:
            # Create new solution instance
            new_solution = MDMSMCVRPSolution(self.problem)

            # Manually copy routes (avoiding deepcopy)
            new_solution.routes = {
                'E1': {k: list(v) for k, v in self.routes['E1'].items()},
                'E2': {k: list(v) for k, v in self.routes['E2'].items()}
            }

            # Manually copy flows (avoiding deepcopy)
            new_solution.flows = {
                'Y': {k: dict(v) for k, v in self.flows['Y'].items()},
                'Z': {k: {inner_k: dict(inner_v) for inner_k, inner_v in v.items()}
                      for k, v in self.flows['Z'].items()}
            }

            # Copy other attributes
            new_solution.fitness = self.fitness
            new_solution.is_feasible = self.is_feasible
            new_solution.constraint_violations = copy.copy(self.constraint_violations)

            return new_solution

        except Exception as e:
            print(f"Error in copy: {e}")
            # Fallback to simple initialization
            new_solution = MDMSMCVRPSolution(self.problem)
            return new_solution

    def validate_and_repair_solution(self):
        """Validate solution and attempt repairs"""
        print("=== SOLUTION VALIDATION ===")
        repairs_made = 0

        # Ensure all customers are assigned exactly once
        assigned_customers = set()
        duplicate_assignments = []

        for k in self.problem.VS:
            route = self.routes['E2'][k]
            unique_customers = []

            for node in route:
                if node in self.problem.VC:
                    if node in assigned_customers:
                        duplicate_assignments.append(node)
                    else:
                        assigned_customers.add(node)
                        unique_customers.append(node)
                elif node == k:  # Keep satellite nodes
                    unique_customers.append(node)

            # Rebuild route without duplicates
            if len(unique_customers) != len([n for n in route if n == k or n in self.problem.VC]):
                self.routes['E2'][k] = [k] + [c for c in unique_customers if c != k] + [k]
                repairs_made += 1

        # Assign unassigned customers
        unassigned = set(self.problem.VC) - assigned_customers
        if unassigned:
            print(f"WARNING: Repairing {len(unassigned)} unassigned customers")
            for customer in unassigned:
                # Assign to satellite with lowest load
                best_satellite = min(self.problem.VS,
                                     key=lambda s: len([c for c in self.routes['E2'][s] if c in self.problem.VC]))
                self.routes['E2'][best_satellite].insert(-1, customer)
                repairs_made += 1

        # Ensure routes start and end correctly
        for l in self.problem.VD:
            route = self.routes['E1'][l]
            if len(route) < 2 or route[0] != l or route[-1] != l:
                # Repair route structure
                satellites = [node for node in route if node in self.problem.VS]
                self.routes['E1'][l] = [l] + satellites + [l]
                repairs_made += 1

        for k in self.problem.VS:
            route = self.routes['E2'][k]
            if len(route) < 2 or route[0] != k or route[-1] != k:
                # Repair route structure
                customers = [node for node in route if node in self.problem.VC]
                self.routes['E2'][k] = [k] + customers + [k]
                repairs_made += 1

        if repairs_made > 0:
            print(f"Made {repairs_made} repairs to solution")
            # Recalculate flows after repairs
            self._calculate_flows()

        print(f"Validation complete. All {len(self.problem.VC)} customers assigned.")

    def get_route_statistics(self):
        """Get detailed statistics about the routes"""
        stats = {
            'primary_routes': {},
            'secondary_routes': {},
            'total_distance': 0,
            'total_customers': len(self.problem.VC),
            'customers_served': 0
        }

        # Primary route statistics
        for l, route in self.routes['E1'].items():
            route_distance = 0
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    route_distance += self.problem.distances[route[i]][route[i + 1]]

            stats['primary_routes'][l] = {
                'route': route,
                'satellites_visited': len([s for s in route if s in self.problem.VS]),
                'route_length': len(route),
                'distance': route_distance
            }
            stats['total_distance'] += route_distance

        # Secondary route statistics
        customers_served = set()
        for k, route in self.routes['E2'].items():
            customers_in_route = [c for c in route if c in self.problem.VC]
            customers_served.update(customers_in_route)

            route_distance = 0
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    route_distance += self.problem.distances[route[i]][route[i + 1]]

            stats['secondary_routes'][k] = {
                'route': route,
                'customers_served': len(customers_in_route),
                'route_length': len(route),
                'customers': customers_in_route,
                'distance': route_distance
            }
            stats['total_distance'] += route_distance

        stats['customers_served'] = len(customers_served)
        stats['service_rate'] = len(customers_served) / len(self.problem.VC) if self.problem.VC else 0

        return stats

    def validate_solution(self):
        """Validate the solution and return detailed constraint checking"""
        violations = []

        # Check if all customers are served
        served_customers = set()
        for k, route in self.routes['E2'].items():
            served_customers.update(c for c in route if c in self.problem.VC)

        unserved = set(self.problem.VC) - served_customers
        if unserved:
            violations.append(f"Unserved customers: {unserved}")

        # Check route structure
        for l, route in self.routes['E1'].items():
            if len(route) < 2 or route[0] != l or route[-1] != l:
                violations.append(f"Primary route {l} doesn't start/end at depot")

        for k, route in self.routes['E2'].items():
            if len(route) < 2 or route[0] != k or route[-1] != k:
                violations.append(f"Secondary route {k} doesn't start/end at satellite")

        # Check for forbidden routes
        forbidden_count = 0
        for l, route in self.routes['E1'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_count += 1

        for k, route in self.routes['E2'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_count += 1

        if forbidden_count > 0:
            violations.append(f"Solution uses {forbidden_count} forbidden routes")

        return {
            'is_valid': len(violations) == 0,
            'violations': violations,
            'statistics': self.get_route_statistics()
        }

    def debug_constraints(self):
        """Enhanced constraint debugging with updated penalty weight"""
        print("=== CONSTRAINT ANALYSIS ===")
        total_violations = 0
        penalty_weight = getattr(self.problem, 'penalty_weight', 100)  # Use problem's penalty weight

        # Check primary vehicle capacities
        print("\nPrimary Vehicle Capacity Analysis:")
        for l in self.problem.VD:
            if l not in self.routes['E1']:
                continue

            route = self.routes['E1'][l]
            total_load = 0
            for i in range(1, len(route)):
                if i >= len(route):
                    break
                prev, curr = route[i - 1], route[i]
                if curr in self.problem.VS:
                    load = self.flows['Y'].get((prev, curr), {}).get(l, 0)
                    total_load += load

            capacity = self.problem.P[l]
            violation = max(0, total_load - capacity)
            penalty = penalty_weight * violation if violation > 0 else 0
            total_violations += penalty
            status = "✅" if violation == 0 else "❌"
            print(
                f"  {status} Vehicle {l}: Load={total_load:.1f}, Capacity={capacity}, Violation={violation:.1f}, Penalty={penalty:.1f}")

        # Check satellite capacities
        print("\nSatellite Capacity Analysis:")
        for j in self.problem.VS:
            total_demand = 0
            if j in self.routes['E2']:
                customers = [c for c in self.routes['E2'][j] if c in self.problem.VC]
                for customer in customers:
                    customer_demand = sum(self.problem.demands.get(customer, {}).values())
                    total_demand += customer_demand

            capacity = self.problem.W[j]
            violation = max(0, total_demand - capacity)
            penalty = penalty_weight * violation if violation > 0 else 0
            total_violations += penalty
            status = "✅" if violation == 0 else "❌"
            customers_count = len([c for c in self.routes['E2'][j] if c in self.problem.VC]) if j in self.routes[
                'E2'] else 0
            print(
                f"  {status} Satellite {j}: Demand={total_demand:.1f}, Capacity={capacity}, Customers={customers_count}, Violation={violation:.1f}, Penalty={penalty:.1f}")

        # Check for forbidden routes
        print("\nForbidden Route Analysis:")
        forbidden_e1 = 0
        forbidden_e2 = 0

        for l, route in self.routes['E1'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_e1 += 1

        for k, route in self.routes['E2'].items():
            for i in range(len(route) - 1):
                if self.problem.is_route_forbidden(route[i], route[i + 1]):
                    forbidden_e2 += 1

        total_forbidden = forbidden_e1 + forbidden_e2
        forbidden_penalty = total_forbidden * penalty_weight * 100
        total_violations += forbidden_penalty

        status = "✅" if total_forbidden == 0 else "❌"
        print(
            f"  {status} Forbidden Routes: E1={forbidden_e1}, E2={forbidden_e2}, Total={total_forbidden}, Penalty={forbidden_penalty:.1f}")

        # Calculate actual route distances
        route_distance = 0
        for l, route in self.routes['E1'].items():
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    dist = self.problem.distances.get(route[i], {}).get(route[i + 1], 0)
                    route_distance += dist

        for k, route in self.routes['E2'].items():
            for i in range(len(route) - 1):
                if not self.problem.is_route_forbidden(route[i], route[i + 1]):
                    dist = self.problem.distances.get(route[i], {}).get(route[i + 1], 0)
                    route_distance += dist

        print(f"\n{'=' * 50}")
        print(f"SUMMARY:")
        print(f"{'=' * 50}")
        print(f"  Total Route Distance: {route_distance:.1f}")
        print(f"  Total Penalty: {total_violations:.1f}")
        print(f"  Solution Fitness: {self.fitness:.1f}")
        print(f"  Expected Fitness: {route_distance + total_violations:.1f}")
        print(f"  Penalty Weight: {penalty_weight}x")

        # Check demand satisfaction
        total_served_demand = 0
        total_required_demand = sum(sum(self.problem.demands[c].values()) for c in self.problem.VC)

        for j in self.problem.VS:
            if j in self.routes['E2']:
                for customer in self.routes['E2'][j]:
                    if customer in self.problem.VC:
                        total_served_demand += sum(self.problem.demands.get(customer, {}).values())

        demand_satisfaction = (total_served_demand / total_required_demand * 100) if total_required_demand > 0 else 0
        print(
            f"  Demand Satisfaction: {demand_satisfaction:.1f}% ({total_served_demand:.1f}/{total_required_demand:.1f})")

        return {
            'route_distance': route_distance,
            'total_penalty': total_violations,
            'expected_fitness': route_distance + total_violations,
            'forbidden_routes': total_forbidden,
            'demand_satisfaction': demand_satisfaction
        }

    def calculate_fitness(self) -> float:
        """Calculate and return the solution fitness (total distance + penalties)"""
        try:
            evaluation = self.problem.evaluate_solution(self.to_dict())
            self.fitness = evaluation['objective']
            self.constraint_violations = evaluation.get('violations', {})
            self.is_feasible = evaluation.get('penalty', 0) == 0
            return self.fitness
        except Exception as e:
            print(f"Error in calculate_fitness: {e}")
            self.fitness = float('inf')
            return self.fitness

    def to_dict(self) -> Dict:
        """Convert solution to dictionary format"""
        return {
            'E1': self.routes['E1'],
            'E2': self.routes['E2'],
            'Y': dict(self.flows['Y']),
            'Z': {k: dict(v) for k, v in self.flows['Z'].items()}
        }

    def to_dict_serializable(self):
        """Convert solution to JSON-serializable format"""
        return {
            'fitness': float(self.fitness),
            'is_feasible': self.is_feasible,
            'routes': {
                'E1': {str(k): list(v) for k, v in self.routes['E1'].items()},
                'E2': {str(k): list(v) for k, v in self.routes['E2'].items()}
            },
            'constraint_violations': self.constraint_violations
        }

    def save_to_file(self, filename):
        """Save solution to JSON file"""
        try:
            data = self.to_dict_serializable()
            data['validation'] = self.validate_solution()

            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Solution saved to {filename}")
        except Exception as e:
            print(f"Error saving solution: {e}")

    @classmethod
    def load_from_file(cls, filename, problem):
        """Load solution from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            solution = cls(problem)
            solution.fitness = data['fitness']
            solution.is_feasible = data.get('is_feasible', False)
            solution.constraint_violations = data.get('constraint_violations', {})

            # Reconstruct routes
            solution.routes['E1'] = {k: v for k, v in data['routes']['E1'].items()}
            solution.routes['E2'] = {k: v for k, v in data['routes']['E2'].items()}

            # Recalculate flows
            solution._calculate_flows()

            print(f"Solution loaded from {filename}")
            return solution

        except Exception as e:
            print(f"Error loading solution: {e}")
            return None

    def print_readable(self):
        """Print solution in human-readable format"""
        print(f"\n{'=' * 50}")
        print(f"SOLUTION SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total Fitness: {self.fitness:.2f}")
        print(f"Feasible: {'Yes' if self.is_feasible else 'No'}")

        if self.constraint_violations:
            print(f"Constraint Violations: {len(self.constraint_violations)} types")

        print(f"\nFirst Echelon Routes (Depots → Satellites):")
        for l, route in self.routes['E1'].items():
            route_str = ' → '.join(map(str, route))
            print(f"  Primary Vehicle {l}: {route_str}")

        print(f"\nSecond Echelon Routes (Satellites → Customers):")
        for k, route in self.routes['E2'].items():
            route_str = ' → '.join(map(str, route))
            customer_count = len([c for c in route if c in self.problem.VC])
            print(f"  Secondary Vehicle {k}: {route_str} ({customer_count} customers)")

        # Add constraint debugging
        self.debug_constraints()

    def __str__(self):
        """String representation of the solution"""
        return f"MDMSMCVRPSolution(fitness={self.fitness:.2f}, feasible={self.is_feasible})"

    def __repr__(self):
        """Detailed string representation"""
        return self.__str__()
