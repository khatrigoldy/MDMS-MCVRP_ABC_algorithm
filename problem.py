from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json
import numpy as np


class MDMSMCVRPProblem:
    def __init__(self, VD: List[str], VS: List[str], VC: List[str],
                 distances: Dict, demands: Dict,
                 P: Dict, Q: Dict, W: Dict, penalty_weight: float = 100.0): # Constraint violations occur because,
                                                                            # in complex optimization problems,
                                                                            # allowing “soft” violations with penalties
                                                                            # can help the solver find practical,
                                                                            # near-feasible solutions instead of failing
                                                                            # or delivering highly suboptimal results.
        """
        Initialize the MDMS-MCVRP problem instance.

        Args:
            VD: List of depots
            VS: List of satellites
            VC: List of customers
            distances: Distance matrix
            demands: Customer demands for each commodity
            P: Primary vehicle capacities
            Q: Secondary vehicle compartment capacities
            W: Satellite capacities
            penalty_weight: Weight for constraint violation penalties (reduced from 1000)
        """
        self.VD = VD
        self.VS = VS
        self.VC = VC
        self.distances = distances
        self.demands = demands
        self.P = P
        self.Q = Q
        self.W = W
        self.penalty_weight = penalty_weight  # More reasonable penalty weight

        # Define forbidden distance threshold
        self.FORBIDDEN_DISTANCE = 9999
        self.MAX_ALLOWED_DISTANCE = 5000  # Reasonable upper bound for valid routes

        # Validate problem instance
        self._validate_problem_instance()

        # Calculate problem statistics
        self.problem_stats = self._calculate_problem_statistics()

        # Print initialization summary
        self._print_initialization_summary()

    def _print_initialization_summary(self):
        """Print key problem characteristics"""
        print(f"\n{'=' * 50}")
        print("PROBLEM INITIALIZATION SUMMARY")
        print(f"{'=' * 50}")
        print(f"Nodes: {len(self.VD)} depots, {len(self.VS)} satellites, {len(self.VC)} customers")
        print(f"Total demand: {sum(sum(self.demands[c].values()) for c in self.VC):.1f}")
        print(f"Total satellite capacity: {sum(self.W.values()):.1f}")
        print(f"Penalty weight: {self.penalty_weight}x")

        # Check capacity feasibility
        total_demand = sum(sum(self.demands[c].values()) for c in self.VC)
        total_satellite_capacity = sum(self.W.values())
        capacity_ratio = total_demand / total_satellite_capacity

        if capacity_ratio > 1.0:
            print(
                f"⚠️  WARNING: Total demand ({total_demand:.1f}) exceeds satellite capacity ({total_satellite_capacity:.1f})")
            print(f"   Capacity utilization: {capacity_ratio:.1%}")
        else:
            print(f"✅ Capacity feasible: {capacity_ratio:.1%} utilization")

    def _validate_problem_instance(self):
        """Enhanced validation with forbidden route detection"""
        errors = []
        warnings = []

        # Check if all required sets are non-empty
        if not self.VD:
            errors.append("Depot set (VD) cannot be empty")
        if not self.VS:
            errors.append("Satellite set (VS) cannot be empty")
        if not self.VC:
            errors.append("Customer set (VC) cannot be empty")

        # Enhanced distance matrix validation
        all_nodes = set(self.VD + self.VS + self.VC)
        forbidden_routes = []

        for node1 in all_nodes:
            if node1 not in self.distances:
                errors.append(f"Distance matrix missing entries for node {node1}")
            else:
                for node2 in all_nodes:
                    if node2 not in self.distances[node1]:
                        errors.append(f"Distance matrix missing entry from {node1} to {node2}")
                    else:
                        distance = self.distances[node1][node2]
                        if distance >= self.FORBIDDEN_DISTANCE:
                            forbidden_routes.append((node1, node2, distance))

        # Report forbidden routes statistics
        if forbidden_routes:
            print(f"Found {len(forbidden_routes)} forbidden routes (distance >= {self.FORBIDDEN_DISTANCE})")

            # Check for critical forbidden routes that might cause issues
            critical_routes = []
            for node1, node2, dist in forbidden_routes:
                # Check if this creates connectivity issues
                if (node1 in self.VD and node2 in self.VS) or (node1 in self.VS and node2 in self.VC):
                    critical_routes.append((node1, node2, dist))

            if critical_routes:
                warnings.append(f"Found {len(critical_routes)} critical forbidden routes that may cause infeasibility")

        # Check demand structure
        for customer in self.VC:
            if customer not in self.demands:
                errors.append(f"Demands missing for customer {customer}")
            else:
                for depot in self.VD:
                    if depot not in self.demands[customer]:
                        errors.append(f"Demand missing for customer {customer}, commodity {depot}")

        # Check capacity structures
        for depot in self.VD:
            if depot not in self.P:
                errors.append(f"Primary vehicle capacity missing for depot {depot}")

        for satellite in self.VS:
            if satellite not in self.Q:
                errors.append(f"Secondary vehicle capacity missing for satellite {satellite}")
            else:
                for depot in self.VD:
                    if depot not in self.Q[satellite]:
                        errors.append(f"Compartment capacity missing for satellite {satellite}, commodity {depot}")

            if satellite not in self.W:
                errors.append(f"Satellite capacity missing for satellite {satellite}")

        if errors:
            raise ValueError("Problem instance validation failed:\n" + "\n".join(errors))

        if warnings:
            print("⚠️ Validation warnings:")
            for warning in warnings:
                print(f"  {warning}")

    def is_route_forbidden(self, from_node: str, to_node: str) -> bool:
        """Check if a route segment uses forbidden connections"""
        if from_node not in self.distances or to_node not in self.distances[from_node]:
            return True

        distance = self.distances[from_node][to_node]
        return distance >= self.FORBIDDEN_DISTANCE

    def get_route_distance(self, from_node: str, to_node: str) -> float:
        """Get distance with forbidden route detection"""
        if self.is_route_forbidden(from_node, to_node):
            return float('inf')  # Return infinity for forbidden routes

        return self.distances[from_node][to_node]

    def validate_route_feasibility(self, route: List[str]) -> Dict:
        """Validate that a route doesn't use forbidden connections"""
        forbidden_segments = []
        total_distance = 0

        for i in range(len(route) - 1):
            from_node, to_node = route[i], route[i + 1]

            if self.is_route_forbidden(from_node, to_node):
                forbidden_segments.append((from_node, to_node, self.distances[from_node][to_node]))
            else:
                total_distance += self.distances[from_node][to_node]

        return {
            'is_feasible': len(forbidden_segments) == 0,
            'forbidden_segments': forbidden_segments,
            'total_distance': total_distance if len(forbidden_segments) == 0 else float('inf')
        }

    def evaluate_solution(self, solution: Dict) -> Dict:
        """Enhanced solution evaluation with forbidden route detection"""
        try:
            total_distance = 0
            penalty = 0
            violations = {}
            detailed_costs = {
                'first_echelon_distance': 0,
                'second_echelon_distance': 0,
                'primary_capacity_penalty': 0,
                'secondary_capacity_penalty': 0,
                'satellite_capacity_penalty': 0,
                'forbidden_route_penalty': 0
            }

            # Calculate first echelon distance with forbidden route detection
            forbidden_routes_e1 = []
            for l, route in solution['E1'].items():
                route_validation = self.validate_route_feasibility(route)

                if not route_validation['is_feasible']:
                    forbidden_routes_e1.extend(route_validation['forbidden_segments'])
                    # Apply heavy penalty for forbidden routes
                    penalty_amount = len(route_validation['forbidden_segments']) * self.penalty_weight * 100
                    penalty += penalty_amount
                    detailed_costs['forbidden_route_penalty'] += penalty_amount
                else:
                    detailed_costs['first_echelon_distance'] += route_validation['total_distance']
                    total_distance += route_validation['total_distance']

            # Calculate second echelon distance with forbidden route detection
            forbidden_routes_e2 = []
            for k, route in solution['E2'].items():
                route_validation = self.validate_route_feasibility(route)

                if not route_validation['is_feasible']:
                    forbidden_routes_e2.extend(route_validation['forbidden_segments'])
                    # Apply heavy penalty for forbidden routes
                    penalty_amount = len(route_validation['forbidden_segments']) * self.penalty_weight * 100
                    penalty += penalty_amount
                    detailed_costs['forbidden_route_penalty'] += penalty_amount
                else:
                    detailed_costs['second_echelon_distance'] += route_validation['total_distance']
                    total_distance += route_validation['total_distance']

            # Record forbidden route violations
            if forbidden_routes_e1 or forbidden_routes_e2:
                violations['forbidden_routes'] = {
                    'first_echelon': forbidden_routes_e1,
                    'second_echelon': forbidden_routes_e2,
                    'total_count': len(forbidden_routes_e1) + len(forbidden_routes_e2)
                }

            # Check primary vehicle capacities (with more reasonable penalties)
            primary_violations = self._check_primary_capacity_constraints(solution, detailed_costs)
            if primary_violations:
                violations.update(primary_violations)

            # Check secondary vehicle compartment capacities
            secondary_violations = self._check_secondary_capacity_constraints(solution, detailed_costs)
            if secondary_violations:
                violations.update(secondary_violations)

            # Check satellite capacities
            satellite_violations = self._check_satellite_capacity_constraints(solution, detailed_costs)
            if satellite_violations:
                violations.update(satellite_violations)

            # Check demand satisfaction
            demand_violations = self._check_demand_satisfaction(solution)
            if demand_violations:
                violations.update(demand_violations)

            # Update penalty from detailed costs
            penalty = sum([
                detailed_costs['primary_capacity_penalty'],
                detailed_costs['secondary_capacity_penalty'],
                detailed_costs['satellite_capacity_penalty'],
                detailed_costs['forbidden_route_penalty']
            ])

            return {
                'total_distance': total_distance,
                'penalty': penalty,
                'violations': violations,
                'objective': total_distance + penalty,
                'detailed_costs': detailed_costs,
                'is_feasible': penalty == 0,
                'forbidden_routes_count': len(forbidden_routes_e1) + len(forbidden_routes_e2)
            }

        except Exception as e:
            print(f"Error in evaluate_solution: {e}")
            return {
                'total_distance': float('inf'),
                'penalty': float('inf'),
                'violations': {'evaluation_error': str(e)},
                'objective': float('inf'),
                'detailed_costs': {},
                'is_feasible': False,
                'forbidden_routes_count': 0
            }

    def _check_primary_capacity_constraints(self, solution: Dict, detailed_costs: Dict) -> Dict:
        """Check primary vehicle capacity constraints with reasonable penalties"""
        violations = {}

        for l in self.VD:
            if l not in solution['E1']:
                continue

            route = solution['E1'][l]
            max_load = 0

            # Calculate maximum load carried by primary vehicle
            current_load = 0
            for i in range(1, len(route)):
                prev_node = route[i - 1]
                current_node = route[i]

                # Get flow on this segment
                segment_flow = solution.get('Y', {}).get((prev_node, current_node), {}).get(l, 0)
                current_load += segment_flow
                max_load = max(max_load, current_load)

                # If returning to depot, reset load
                if current_node == l:
                    current_load = 0

            if max_load > self.P[l]:
                excess = max_load - self.P[l]
                penalty_amount = self.penalty_weight * excess
                detailed_costs['primary_capacity_penalty'] += penalty_amount

                violations[f'primary_capacity_{l}'] = {
                    'capacity': self.P[l],
                    'used': max_load,
                    'excess': excess,
                    'penalty': penalty_amount
                }

        return violations

    def _check_secondary_capacity_constraints(self, solution: Dict, detailed_costs: Dict) -> Dict:
        """Check secondary vehicle compartment capacity constraints"""
        violations = {}

        for k in self.VS:
            if k not in solution['E2']:
                continue

            route = solution['E2'][k]
            max_compartment_loads = {l: 0 for l in self.VD}

            # Track compartment loads throughout the route
            compartment_loads = {l: 0 for l in self.VD}

            for i in range(1, len(route)):
                prev_node = route[i - 1]
                current_node = route[i]

                for l in self.VD:
                    segment_flow = solution.get('Z', {}).get((prev_node, current_node), {}).get(k, {}).get(l, 0)
                    compartment_loads[l] += segment_flow
                    max_compartment_loads[l] = max(max_compartment_loads[l], compartment_loads[l])

                    # If delivering to customer, reduce load
                    if current_node in self.VC:
                        delivered = self.demands.get(current_node, {}).get(l, 0)
                        compartment_loads[l] = max(0, compartment_loads[l] - delivered)

            # Check capacity violations
            for l, max_load in max_compartment_loads.items():
                if max_load > self.Q[k][l]:
                    excess = max_load - self.Q[k][l]
                    penalty_amount = self.penalty_weight * excess
                    detailed_costs['secondary_capacity_penalty'] += penalty_amount

                    violations[f'secondary_capacity_{k}_{l}'] = {
                        'capacity': self.Q[k][l],
                        'used': max_load,
                        'excess': excess,
                        'penalty': penalty_amount
                    }

        return violations

    def _check_satellite_capacity_constraints(self, solution: Dict, detailed_costs: Dict) -> Dict:
        """Check satellite capacity constraints with improved calculation"""
        violations = {}

        for j in self.VS:
            if j not in solution['E2']:
                continue

            # Calculate total demand assigned to this satellite
            total_satellite_demand = 0
            customers_served = [c for c in solution['E2'][j] if c in self.VC]

            for customer in customers_served:
                customer_total_demand = sum(self.demands.get(customer, {}).get(l, 0) for l in self.VD)
                total_satellite_demand += customer_total_demand

            if total_satellite_demand > self.W[j]:
                excess = total_satellite_demand - self.W[j]
                penalty_amount = self.penalty_weight * excess
                detailed_costs['satellite_capacity_penalty'] += penalty_amount

                violations[f'satellite_capacity_{j}'] = {
                    'capacity': self.W[j],
                    'used': total_satellite_demand,
                    'excess': excess,
                    'penalty': penalty_amount,
                    'customers_served': len(customers_served)
                }

        return violations

    def _check_demand_satisfaction(self, solution: Dict) -> Dict:
        """Enhanced demand satisfaction checking"""
        violations = {}

        # Check if all customers are served exactly once
        served_customers = set()
        customer_assignments = {}

        for k, route in solution['E2'].items():
            for node in route:
                if node in self.VC:
                    if node in served_customers:
                        violations[f'duplicate_assignment_{node}'] = {
                            'previously_assigned_to': customer_assignments[node],
                            'also_assigned_to': k
                        }
                    else:
                        served_customers.add(node)
                        customer_assignments[node] = k

        unserved_customers = set(self.VC) - served_customers
        if unserved_customers:
            violations['unserved_customers'] = list(unserved_customers)

        return violations

    def debug_solution_constraints(self, solution: Dict) -> Dict:
        """Comprehensive constraint debugging for solution analysis"""
        debug_info = {
            'route_feasibility': {},
            'capacity_analysis': {},
            'flow_analysis': {},
            'demand_analysis': {}
        }

        # Analyze route feasibility
        for l, route in solution['E1'].items():
            debug_info['route_feasibility'][f'E1_{l}'] = self.validate_route_feasibility(route)

        for k, route in solution['E2'].items():
            debug_info['route_feasibility'][f'E2_{k}'] = self.validate_route_feasibility(route)

        # Capacity utilization analysis
        debug_info['capacity_analysis'] = {
            'primary_vehicles': {},
            'satellites': {},
            'secondary_compartments': {}
        }

        # Satellite utilization
        for j in self.VS:
            if j in solution['E2']:
                customers = [c for c in solution['E2'][j] if c in self.VC]
                total_demand = sum(sum(self.demands.get(c, {}).values()) for c in customers)
                utilization = total_demand / self.W[j] if self.W[j] > 0 else 0

                debug_info['capacity_analysis']['satellites'][j] = {
                    'capacity': self.W[j],
                    'used': total_demand,
                    'utilization': utilization,
                    'customers_count': len(customers),
                    'is_overloaded': total_demand > self.W[j]
                }

        return debug_info

    def get_solution_quality_metrics(self, solution: Dict) -> Dict:
        """Calculate comprehensive solution quality metrics"""
        evaluation = self.evaluate_solution(solution)

        metrics = {
            'objective_value': evaluation['objective'],
            'total_distance': evaluation['total_distance'],
            'total_penalty': evaluation['penalty'],
            'penalty_percentage': (evaluation['penalty'] / evaluation['objective'] * 100) if evaluation[
                                                                                                 'objective'] > 0 else 0,
            'is_feasible': evaluation['is_feasible'],
            'forbidden_routes_count': evaluation.get('forbidden_routes_count', 0)
        }

        # Calculate efficiency metrics
        total_demand = sum(sum(self.demands[c].values()) for c in self.VC)
        total_satellite_capacity = sum(self.W.values())

        metrics.update({
            'capacity_utilization': total_demand / total_satellite_capacity if total_satellite_capacity > 0 else 0,
            'distance_per_customer': evaluation['total_distance'] / len(self.VC) if len(self.VC) > 0 else 0,
            'solution_efficiency': total_demand / evaluation['objective'] if evaluation['objective'] > 0 else 0
        })

        return metrics

    def _calculate_problem_statistics(self):
        """Enhanced problem statistics calculation"""
        stats = {
            'num_depots': len(self.VD),
            'num_satellites': len(self.VS),
            'num_customers': len(self.VC),
            'num_commodities': len(self.VD),
            'total_nodes': len(self.VD) + len(self.VS) + len(self.VC)
        }

        # Calculate total demand
        total_demand = 0
        max_customer_demand = 0
        for customer in self.VC:
            customer_demand = sum(self.demands[customer][depot] for depot in self.VD)
            total_demand += customer_demand
            max_customer_demand = max(max_customer_demand, customer_demand)

        stats.update({
            'total_demand': total_demand,
            'average_customer_demand': total_demand / len(self.VC) if self.VC else 0,
            'max_customer_demand': max_customer_demand
        })

        # Calculate capacity statistics
        stats.update({
            'total_primary_capacity': sum(self.P.values()),
            'total_satellite_capacity': sum(self.W.values()),
            'average_primary_capacity': np.mean(list(self.P.values())),
            'average_satellite_capacity': np.mean(list(self.W.values())),
            'capacity_feasibility_ratio': total_demand / sum(self.W.values()) if sum(self.W.values()) > 0 else float(
                'inf')
        })

        # Distance matrix statistics
        valid_distances = []
        forbidden_count = 0

        for from_node, destinations in self.distances.items():
            for to_node, distance in destinations.items():
                if distance >= self.FORBIDDEN_DISTANCE:
                    forbidden_count += 1
                else:
                    valid_distances.append(distance)

        if valid_distances:
            stats.update({
                'min_distance': min(valid_distances),
                'max_distance': max(valid_distances),
                'avg_distance': np.mean(valid_distances),
                'forbidden_routes_count': forbidden_count,
                'valid_routes_count': len(valid_distances)
            })

        return stats

    def calculate_lower_bound(self) -> float:
        """Enhanced lower bound calculation avoiding forbidden routes"""
        try:
            min_distance = 0

            # For each customer, find minimum valid distance from any satellite
            for customer in self.VC:
                min_dist_to_customer = float('inf')
                for satellite in self.VS:
                    if not self.is_route_forbidden(satellite, customer):
                        dist = self.distances[satellite][customer]
                        min_dist_to_customer = min(min_dist_to_customer, dist)

                if min_dist_to_customer != float('inf'):
                    min_distance += min_dist_to_customer * 2  # Round trip

            # Add minimum valid distances from depots to satellites
            for depot in self.VD:
                min_dist_to_satellite = float('inf')
                for satellite in self.VS:
                    if not self.is_route_forbidden(depot, satellite):
                        dist = self.distances[depot][satellite]
                        min_dist_to_satellite = min(min_dist_to_satellite, dist)

                if min_dist_to_satellite != float('inf'):
                    min_distance += min_dist_to_satellite * 2  # Round trip

            return min_distance

        except Exception as e:
            print(f"Error calculating lower bound: {e}")
            return 0

    # ... (keeping the remaining methods unchanged: save/load, summary printing, etc.)

    def save_problem_instance(self, filename: str):
        """Save the problem instance to a JSON file"""
        try:
            problem_data = {
                'VD': self.VD,
                'VS': self.VS,
                'VC': self.VC,
                'distances': self.distances,
                'demands': self.demands,
                'P': self.P,
                'Q': self.Q,
                'W': self.W,
                'penalty_weight': self.penalty_weight,
                'statistics': self.problem_stats
            }

            with open(filename, 'w') as f:
                json.dump(problem_data, f, indent=2)

            print(f"Problem instance saved to {filename}")

        except Exception as e:
            print(f"Error saving problem instance: {e}")

    @classmethod
    def load_problem_instance(cls, filename: str):
        """Load a problem instance from a JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            # Remove statistics if present (will be recalculated)
            data.pop('statistics', None)

            return cls(**data)

        except Exception as e:
            print(f"Error loading problem instance: {e}")
            return None

    def print_problem_summary(self):
        """Enhanced problem summary with constraint feasibility analysis"""
        print("=" * 60)
        print("MDMS-MCVRP PROBLEM INSTANCE SUMMARY")
        print("=" * 60)

        print(f"Problem Size:")
        print(f"  Depots: {len(self.VD)}")
        print(f"  Satellites: {len(self.VS)}")
        print(f"  Customers: {len(self.VC)}")
        print(f"  Total Nodes: {len(self.VD) + len(self.VS) + len(self.VC)}")

        print(f"\nCapacity Information:")
        print(f"  Total Primary Vehicle Capacity: {sum(self.P.values())}")
        print(f"  Total Satellite Capacity: {sum(self.W.values())}")
        print(f"  Average Primary Capacity: {np.mean(list(self.P.values())):.2f}")
        print(f"  Average Satellite Capacity: {np.mean(list(self.W.values())):.2f}")

        print(f"\nDemand Information:")
        total_demand = sum(sum(self.demands[c].values()) for c in self.VC)
        print(f"  Total Demand: {total_demand}")
        print(f"  Average Customer Demand: {total_demand / len(self.VC):.2f}")

        # Capacity feasibility analysis
        capacity_ratio = total_demand / sum(self.W.values()) if sum(self.W.values()) > 0 else float('inf')
        print(f"\nFeasibility Analysis:")
        print(f"  Capacity Utilization: {capacity_ratio:.1%}")
        if capacity_ratio > 1.0:
            print(f"  ⚠️ WARNING: Demand exceeds satellite capacity!")
        else:
            print(f"  ✅ Capacity constraints are satisfiable")

        print(f"\nDistance Matrix:")
        if 'forbidden_routes_count' in self.problem_stats:
            print(f"  Valid routes: {self.problem_stats['valid_routes_count']}")
            print(f"  Forbidden routes: {self.problem_stats['forbidden_routes_count']}")
            print(f"  Average valid distance: {self.problem_stats.get('avg_distance', 0):.2f}")

        print(f"\nLower Bound Estimate: {self.calculate_lower_bound():.2f}")
        print(f"Penalty Weight: {self.penalty_weight}x")

        print("=" * 60)

    def __str__(self):
        """String representation of the problem"""
        return f"MDMS-MCVRP({len(self.VD)}D-{len(self.VS)}S-{len(self.VC)}C)"

    def __repr__(self):
        """Detailed string representation"""
        return self.__str__()
