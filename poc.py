import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
from typing import Optional, Tuple, Dict, Any, List

class EnvironmentCMDPStability(gym.Env):
    """
    SECTION: Environment Initialization and Configuration
    Initializes the quayside operations environment under a Constrained Markov Decision Process (CMDP) framework.
    Defines discrete and continuous action spaces, observation space dimensions, Ornstein-Uhlenbeck stochastic drift 
    parameters for environmental factors (wind, waves, visibility, precipitation, water velocity), harmonic tide 
    constants, and physical quay limits.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, dictionary_configuration: Optional[Dict[str, Any]] = None):
        super().__init__()
        # Mathematical / Algorithmic Calculation: Configuration dictionary parsing and default parameter assignment.
        self.dictionary_configuration = dictionary_configuration or {}
        self.quay_length: float = self.dictionary_configuration.get("quay_length", 1200.0)
        self.crane_count: int = self.dictionary_configuration.get("crane_count", 10)
        self.crane_distance: float = self.dictionary_configuration.get("crane_distance", 30.0)
        self.simulation_duration: float = self.dictionary_configuration.get("simulation_duration", 168.0)
        self.dt: float = self.dictionary_configuration.get("dt", 0.25)
        self.queue_capacity: float = self.dictionary_configuration.get("queue_capacity", 15)
        self.vessel_count_start: int = self.dictionary_configuration.get("initial_vessel_count", 5)
        self.allocation_limit: float = self.dictionary_configuration.get("allocation_limit", 6)
        self.gamma: float = self.dictionary_configuration.get("gamma", 0.99)
        
        # Action space definition: Dict space encompassing vessel index selection, berth fraction continuous placement,
        # crane allocation count, and continuous ballast adjustment rate.
        self.action_space = spaces.Dict({
            "vessel_idx": spaces.Discrete(int(self.queue_capacity)),
            "berth_fraction": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "crane_count": spaces.Discrete(int(self.allocation_limit)),
            "ballast_rate": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        # Observation dimension calculation: (queue_capacity * 11 features per vessel) + (crane_count * 6 features per crane) + 12 environmental telemetry features.
        observation_dimension = (int(self.queue_capacity) * 11) + (int(self.crane_count) * 6) + 12
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(observation_dimension,), dtype=np.float32
        )
        
        # Ornstein-Uhlenbeck parameters: Mean reversion rate (theta), long-term mean (mu), and volatility (sigma) for 
        # [Wind Speed, Wave Height, Visibility, Precipitation, Water Velocity].
        self.ou_theta = np.array([0.15, 0.10, 0.08, 0.12, 0.10], dtype=np.float64)
        self.ou_mu = np.array([6.5, 0.9, 14.0, 0.15, 0.5], dtype=np.float64)
        self.ou_sigma = np.array([1.4, 0.18, 0.5, 0.04, 0.05], dtype=np.float64)
        
        # Environmental correlation matrix governing cross-dependencies between meteorological and oceanographic factors.
        self.correlation_matrix = np.array([
            [1.00,  0.75, -0.40,  0.30,  0.20],
            [0.75,  1.00, -0.50,  0.25,  0.35],
            [-0.40, -0.50, 1.00, -0.30, -0.10],
            [0.30,  0.25, -0.30,  1.00,  0.05],
            [0.20,  0.35, -0.10,  0.05,  1.00]
        ], dtype=np.float64)
        
        # Covariance matrix calculation: Outer product of sigmas scaled by correlation matrix.
        covariance_matrix = np.outer(self.ou_sigma, self.ou_sigma) * self.correlation_matrix
        try:
            # Cholesky decomposition: Factorizes covariance matrix $\boldsymbol{\Sigma} = \mathbf{L}\mathbf{L}^T$ for correlated Wiener process generation.
            self.cholesky_matrix = np.linalg.cholesky(covariance_matrix)
        except np.linalg.LinAlgError:
            """Empirically constructed correlation matrices or rounding errors in floating-point operations can occasionally render 
            $\boldsymbol{\Sigma}$ non-positive definite (i.e., possessing zero or negative eigenvalues, or severe ill-conditioning).
            This fallback replaces the full Cholesky factor with a diagonal matrix containing the individual standard deviations"""
            
            self.cholesky_matrix = np.diag(self.ou_sigma)

        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        SECTION: Environment State Reset
        Resets temporal simulation clocks, vessel queue/berth/finish tracking lists, baseline environmental parameters 
        (wind, wave, visibility, precipitation, water velocity), and initializes crane positions along the quay.
        """
        super().reset(seed=seed)
        self.clock_time: float = 0.0
        self.vessels_queue: List[Dict[str, Any]] = []
        self.vessels_berth: List[Dict[str, Any]] = []
        self.vessels_finish: List[Dict[str, Any]] = []
        
        # Baseline environmental initializations
        self.speed_wind: float = 5.0
        self.direction_wind: float = 90.0
        self.height_wave: float = 0.5
        self.period_wave: float = 6.0
        self.visibility: float = 15.0
        self.precipitation: float = 0.0
        self.temperature: float = 24.0
        self.velocity_water: float = 0.4
        
        # Crane initial state tensor allocation [position, status, assigned_vessel_id, target_pos, unused, operational_hours, velocity]
        self.cranes = np.zeros((int(self.crane_count), 7), dtype=np.float32)
        for crane_index in range(int(self.crane_count)):
            self.cranes[crane_index, 0] = crane_index * (self.quay_length / self.crane_count)
            self.cranes[crane_index, 1] = 0.0
            self.cranes[crane_index, 2] = -1.0
            self.cranes[crane_index, 3] = 100.0
            self.cranes[crane_index, 4] = 0.0
            self.cranes[crane_index, 5] = 0.0
            self.cranes[crane_index, 6] = 0.0

        self._generate_arrivals(origin_indicator=True)
        return self._get_observation(), self._get_action_masks()

    def _calculate_tide(self, time_parameter: float) -> float:
        """
        SECTION: Harmonic Tide Modeling
        Computes sea water elevation at any simulation clock time using constituent harmonic components (M2, S2, M4).
        """
        # Angular frequencies for M2 (principal lunar semidiurnal, ~12.42h), S2 (principal solar semidiurnal, 12.00h), M4 (quarter-diurnal).
        omega_m2 = 2.0 * np.pi / 12.4206
        omega_s2 = 2.0 * np.pi / 12.0000
        omega_m4 = 4.0 * np.pi / 12.4206
        # Harmonic superposition equation calculating tidal water level elevation $h(t)$.
        return 2.8 * np.cos(omega_m2 * time_parameter) + 1.2 * np.cos(omega_s2 * time_parameter) + 0.3 * np.cos(omega_m4 * time_parameter)

    def _update_physics(self):
        """
        SECTION: Stochastic Environmental Physics Update
        Advances environmental state variables through discretized Ornstein-Uhlenbeck stochastic differential equations,
        incorporating Cholesky-correlated Wiener diffusion processes and Milstein numerical corrections.
        """
        dt = self.dt
        # Generate uncorrelated standard normal white noise increments.
        dW_raw = np.random.normal(0, np.sqrt(dt), size=5)
        # Apply Cholesky matrix to induce correct spatial/cross-variable correlation structure: $dW = \mathbf{L} dW_{\text{raw}}$.
        dW = self.cholesky_matrix @ dW_raw
        environment_state = np.array([
            self.speed_wind, self.height_wave, self.visibility, self.precipitation, self.velocity_water
        ], dtype=np.float64)
        # OU Drift term: $\theta (\mu - X_t) dt$.
        drift = self.ou_theta * (self.ou_mu - environment_state) * dt
        diffusion = dW
        # Milstein numerical correction term for non-linear SDE integration accuracy: $0.1 (dW^2 - dt)$.
        milstein_correction = 0.1 * (dW**2 - dt)
        # State update equation: $X_{t+\Delta t} = X_t + \text{drift} + \text{diffusion} + \text{milstein}$.
        updated_environment = environment_state + drift + diffusion + milstein_correction
        # Apply non-negativity and physical bounding clips.
        self.speed_wind = max(0.0, updated_environment[0])
        self.height_wave = max(0.1, updated_environment[1])
        self.visibility = np.clip(updated_environment[2], 0.1, 25.0)
        self.precipitation = max(0.0, updated_environment[3])
        self.velocity_water = max(0.0, updated_environment[4])

    def _generate_arrivals(self, origin_indicator: bool = False):
        """
        SECTION: Vessel Arrival Generation
        Generates incoming vessels into the arrival queue using a Poisson distribution intensity scaled by time step dt,
        assigning physical attributes (length, draft, mass, initial GM, ballast, priority, and due dates).
        """
        arrival_intensity = 0.45 * self.dt
        # Poisson arrival count sampling, or fixed initial count on reset.
        origin_count = self.vessel_count_start if origin_indicator else np.random.poisson(arrival_intensity)
        for _ in range(origin_count):
            if len(self.vessels_queue) >= self.queue_capacity:
                break
            vessel_identifier = len(self.vessels_queue) + len(self.vessels_berth) + len(self.vessels_finish)
            vessel_length = np.random.uniform(150.0, 340.0)
            vessel_mass = vessel_length * 120.0
            vessel_kg = 8.5
            vessel_gm = 1.8
            vessel_heel = 0.0
            vessel_ballast = 2000.0
            move_val = np.random.randint(80, 250)
            self.vessels_queue.append({
                "id": vessel_identifier,
                "length": vessel_length,
                "draft": np.random.uniform(9.0, 15.5),
                "move_balance": move_val,
                "move_sum": move_val,
                "time_arrival": self.clock_time,
                "date_due": self.clock_time + np.random.uniform(30.0, 90.0),
                "berth_preference": np.random.uniform(0.0, self.quay_length - vessel_length),
                "weight_priority": np.random.uniform(1.0, 3.0),
                "mass": vessel_mass,
                "kg": vessel_kg,
                "gm": vessel_gm,
                "heel": vessel_heel,
                "ballast": vessel_ballast
            })

    def _enforce_kinematics(self):
        """
        SECTION: Crane Kinematics and Collision Avoidance
        Enforces physical velocity and acceleration limits, frictional damping, and strict minimum distance separation constraints
        between adjacent quay cranes using iterative positional projection.
        """
        velocity_limit = 45.0
        acceleration_limit = 15.0
        friction_coefficient = 0.02
        for crane_index in range(int(self.crane_count)):
            position_target = self.cranes[crane_index, 0]
            if self.cranes[crane_index, 2] != -1:
                vessel_identifier = int(self.cranes[crane_index, 2])
                for vessel_item in self.vessels_berth:
                    if vessel_item["id"] == vessel_identifier:
                        position_target = vessel_item["coordinate"] + (vessel_item["length"] / 2.0)
                        break
            # Proportional velocity control toward target position: $v_{\text{target}} = \text{clip}(2(p_{\text{target}} - p_{\text{current}}))$.
            position_error = position_target - self.cranes[crane_index, 0]
            velocity_target = np.clip(position_error * 2.0, -velocity_limit, velocity_limit)
            velocity_difference = velocity_target - self.cranes[crane_index, 6]
            # Acceleration calculation with frictional damping: $a = \text{clip}(v_{\text{diff}}) - c_f v$.
            acceleration = np.clip(velocity_difference, -acceleration_limit, acceleration_limit) - (friction_coefficient * self.cranes[crane_index, 6])
            self.cranes[crane_index, 6] += acceleration * self.dt
            self.cranes[crane_index, 0] += self.cranes[crane_index, 6] * self.dt
        # Clip positions within quay boundaries.
        self.cranes[:, 0] = np.clip(self.cranes[:, 0], 0.0, self.quay_length)
        
        # Iterative Gauss-Seidel projection to enforce minimum inter-crane safety distance separation ($d_{\min} = 30.0$).
        iteration_limit = 50
        tolerance = 1e-4
        for _ in range(iteration_limit):
            positions_start = self.cranes[:, 0].copy()
            indices_order = np.argsort(self.cranes[:, 0])
            for index_value in range(1, len(indices_order)):
                index_history = indices_order[index_value - 1]
                index_current = indices_order[index_value]
                position_minimum = self.cranes[index_history, 0] + self.crane_distance
                if self.cranes[index_current, 0] < position_minimum:
                    self.cranes[index_current, 0] = position_minimum
                    self.cranes[index_current, 6] = 0.0
            for index_value in range(len(indices_order) - 2, -1, -1):
                index_future = indices_order[index_value + 1]
                index_current = indices_order[index_value]
                position_maximum = self.cranes[index_future, 0] - self.crane_distance
                if self.cranes[index_current, 0] > position_maximum:
                    self.cranes[index_current, 0] = position_maximum
                    self.cranes[index_current, 6] = 0.0
            delta_limit = np.max(np.abs(self.cranes[:, 0] - positions_start))
            if delta_limit < tolerance:
                break

    def _get_action_masks(self) -> Dict[str, np.ndarray]:
        """
        SECTION: Action Masking
        Generates validity masks for discrete vessel queue selections to prevent indexing errors when queues are empty.
        """
        mask_vessel = np.zeros(int(self.queue_capacity), dtype=np.float32)
        if len(self.vessels_queue) > 0:
            mask_vessel[:len(self.vessels_queue)] = 1.0
        else:
            mask_vessel[0] = 1.0
        return {"vessel_mask": mask_vessel}

    def step(self, action: Dict[str, Any]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        SECTION: Environment Step Execution
        Advances the simulation step by processing agent actions (berthing placement, crane allocation, ballast adjustment),
        evaluating weather safety constraints, updating vessel stability (GM and heel angle), computing potential-based shaping rewards,
        and accumulating constraint violation costs.
        
        $$R_t = \left( \alpha_1 R_{\text{queue}} + \alpha_2 R_{\text{berth}} + R_{\text{shaping}} \right) \Delta t + \alpha_3 R_{\text{completion}}$$
        $$R_{\text{queue}} = -\left(\frac{N_{\text{queue}}}{Q_{\text{capacity}}}\right)^2 - 0.05 \max\left(0.0, \frac{N_{\text{queue}}}{Q_{\text{capacity}}} - 0.5\right)$$where $N_{\text{queue}}$ is the current number of vessels waiting in the queue, and $Q_{\text{capacity}}$ is the maximum queue capacity (set to $15$).
        $$R_{\text{berth}} = \left( 2.0 \cdot \mathbb{I}(\text{berth\_success}) - 0.5 \cdot \mathbb{I}(\text{berth\_failure}) \right) \cdot \left( \frac{w_{\text{priority}}}{3.0} \right)$$where $\mathbb{I}(\cdot)$ is an indicator function, and $w_{\text{priority}} \in [1.0, 3.0]$ represents the vessel's priority weight.
        $$R_{\text{shaping}} = 0.1 \left( \gamma \Phi(s_{t+1}) - \Phi(s_t) \right)$$where the potential function $\Phi(s)$ is defined as the negative sum of normalized remaining move balances:$$\Phi(s) = -\sum_{v \in \text{active}} \frac{\text{move\_balance}_v}{\max\left(1.0, \text{move\_sum}_v\right)}$$
        Rewards successful completion of vessel operations, granting a high bonus for on-time delivery ($\tau_v = 0.0$) and a linear penalty for tardiness ($\tau_v > 0.0$):$$R_{\text{completion}} = \sum_{v \in \text{finished}} \left( \frac{w_{\text{priority}, v}}{3.0} \right) \cdot \begin{cases} 100.0, & \text{if } \tau_v = 0.0 \\ -0.05 \cdot \tau_v, & \text{if } \tau_v > 0.0 \end{cases}$$where $\tau_v = \max(0.0, t_{\text{current}} - t_{\text{due}})$ is the tardiness duration of vessel $v$.
        """
        self._update_physics()
        # Weather safety constraint flags: storm lock and crane lockdown thresholds.
        storm_lock_flag = (self.height_wave > 2.8) or (self.visibility < 0.4) or (self.speed_wind > 18.5)
        crane_lockdown_flag = (self.speed_wind > 20.5)

        vessel_index = int(action["vessel_idx"])
        berth_fraction = float(action["berth_fraction"][0])
        cranes_count = int(action["crane_count"]) + 1
        ballast_rate = float(action["ballast_rate"][0])
        
        # Potential function calculation before state transition: $\Phi(s) = -\sum \frac{\text{move\_balance}}{\text{move\_sum}}$.
        all_active_vessels = self.vessels_queue + self.vessels_berth
        potential_before = -sum([(v["move_balance"] / max(1.0, v["move_sum"])) for v in all_active_vessels])
        
        constraint_cost = 0.0
        berth_success_flag = False
        berth_failure_flag = False
        vessel_berthed_weight = 0.0
        queue_count = float(len(self.vessels_queue))

        # Process berthing action if queue is non-empty.
        if len(self.vessels_queue) > 0:
            if storm_lock_flag:
                berth_failure_flag = True
            else:
                if vessel_index >= len(self.vessels_queue):
                    vessel_index = len(self.vessels_queue) - 1
                vessel_item = self.vessels_queue[vessel_index]
                berth_maximum = self.quay_length - vessel_item["length"]
                coordinate = berth_fraction * max(0.0, berth_maximum)
                tide = self._calculate_tide(self.clock_time)
                chart_datum = 10.0
                keel_clearance = 1.5
                draft_limit = tide + chart_datum
                # Spatial boundary permission, draft clearance check, and spatial overlap collision checks.
                coordinate_permission_flag = (coordinate + vessel_item["length"] <= self.quay_length) and (coordinate >= 0.0)
                draft_permission_flag = (vessel_item["draft"] <= draft_limit + keel_clearance)
                overlap_permission_flag = all((coordinate + vessel_item["length"] <= vessel_berth_item["coordinate"]) or (coordinate >= vessel_berth_item["coordinate"] + vessel_berth_item["length"]) for vessel_berth_item in self.vessels_berth)
                
                if coordinate_permission_flag and draft_permission_flag and overlap_permission_flag:
                    vessel_item["coordinate"] = coordinate
                    vessel_item["time_berth"] = self.clock_time
                    vessel_item["allocated_cranes"] = cranes_count
                    self.vessels_queue.pop(vessel_index)
                    self.vessels_berth.append(vessel_item)
                    berth_success_flag = True
                    vessel_berthed_weight = vessel_item["weight_priority"] / 3.0
                else:
                    berth_failure_flag = True
                    constraint_cost += 1.0

        self.clock_time += self.dt
        
        # Environmental productivity multipliers affecting crane operation rates.
        wind_multiplier = 0.0 if crane_lockdown_flag else max(0.0, 1.0 - 0.06 * max(0.0, self.speed_wind - 10.0))
        rain_multiplier = max(0.1, 1.0 - 0.15 * self.precipitation)
        visibility_multiplier = min(1.0, self.visibility / 5.0)
        berth_congestion_factor = max(0.5, 1.0 - 0.05 * float(len(self.vessels_berth)))
        environment_multiplier = wind_multiplier * rain_multiplier * visibility_multiplier * berth_congestion_factor

        for crane_index in range(int(self.crane_count)):
            self.cranes[crane_index, 1] = 3.0 if crane_lockdown_flag else (1.0 if self.cranes[crane_index, 2] != -1 else 0.0)
            if self.cranes[crane_index, 1] == 1.0:
                self.cranes[crane_index, 5] += 12.5 * self.dt

        self._enforce_kinematics()

        vessels_finish = []
        tardiness_sum = 0.0
        finished_tardiness_list = []
        for vessel_item in self.vessels_berth:
            crane_allocation_count = vessel_item.get("allocated_cranes", 2)
            # Crane operation rate calculation with diminishing returns for excessive crane allocation.
            rate_operation = crane_allocation_count * environment_multiplier * (1.0 - 0.07 * max(0, crane_allocation_count - 2))
            vessel_item["move_balance"] -= rate_operation * self.dt
            
            cargo_mass_delta = rate_operation * 15.0 * self.dt
            vessel_item["mass"] -= cargo_mass_delta
            vessel_item["kg"] += 0.001 * rate_operation * self.dt
            
            ballast_delta = ballast_rate * 50.0 * self.dt
            vessel_item["ballast"] = np.clip(vessel_item["ballast"] + ballast_delta, 500.0, 5000.0)
            
            # Naval architecture stability calculation: Displacement volume, metacentric radius, keel-to-metacenter height, free surface correction, GM, heel angle.
            displacement_volume = vessel_item["mass"] / 1.025
            metacentric_radius = 4.5 * (vessel_item["length"] / 300.0)
            keel_to_metacenter = 5.0 + metacentric_radius
            free_surface_correction = 0.2 * (vessel_item["ballast"] / 3000.0)
            vessel_item["gm"] = keel_to_metacenter - vessel_item["kg"] - free_surface_correction
            
            heeling_moment = (ballast_delta * 12.0) + (cargo_mass_delta * 4.0)
            vessel_item["heel"] = np.clip(heeling_moment / max(0.5, vessel_item["gm"]), -15.0, 15.0)
            
            # Constraint violation checks for GM (< 0.5m) and heel angle (> 5.0 degrees).
            gm_violation = max(0.0, 0.5 - vessel_item["gm"])
            heel_violation = max(0.0, abs(vessel_item["heel"]) - 5.0)
            
            if gm_violation > 0.0:
                constraint_cost += 0.2
            if heel_violation > 0.0:
                constraint_cost += 0.2

            if vessel_item["move_balance"] <= 0:
                tardiness = max(0.0, self.clock_time - vessel_item["date_due"])
                finished_tardiness_list.append((tardiness, vessel_item["weight_priority"]))
                vessels_finish.append(vessel_item)
                
        for vessel_finish_item in vessels_finish:
            self.vessels_berth.remove(vessel_finish_item)
            self.vessels_finish.append(vessel_finish_item)

        # Potential function calculation after state transition.
        all_active_vessels_after = self.vessels_queue + self.vessels_berth
        potential_after = -sum([(v["move_balance"] / max(1.0, v["move_sum"])) for v in all_active_vessels_after])
        
        alpha_1, alpha_2, alpha_3 = 0.1, 0.3, 0.6
        
        queue_ratio = queue_count / float(self.queue_capacity)
        reward_queue = - (queue_ratio ** 2) - 0.05 * max(0.0, queue_ratio - 0.5)
        
        reward_berth = (2.0 if berth_success_flag else 0.0) * vessel_berthed_weight - (0.5 if berth_failure_flag else 0.0)
        
        reward_completion = 0.0
        for tardiness, priority in finished_tardiness_list:
            p_weight = priority / 3.0
            if tardiness == 0.0:
                reward_completion += 100.0 * p_weight
            else:
                reward_completion -= 0.05 * tardiness * p_weight

        # Potential-based reward shaping component: $R_{\text{shaping}} = 0.1(\gamma \Phi_{t+1} - \Phi_t)$.
        reward_shaping = 0.1 * (self.gamma * potential_after - potential_before)
        
        reward = float((alpha_1 * reward_queue + alpha_2 * reward_berth + reward_shaping) * self.dt + alpha_3 * reward_completion)

        self._generate_arrivals(origin_indicator=False)

        termination_indicator = self.clock_time >= self.simulation_duration
        truncation_indicator = False
        information = self._get_action_masks()
        information["constraint_cost"] = constraint_cost
        return self._get_observation(), reward, termination_indicator, truncation_indicator, information

    def _get_observation(self) -> np.ndarray:
        """
        SECTION: Observation Space Vector Construction
        Flattens and normalizes entity feature representations (vessels in queue, cranes, and environmental telemetry)
        into a unified vector for neural network consumption.
        """
        observation_list = []
        for index_value in range(int(self.queue_capacity)):
            if index_value < len(self.vessels_queue):
                vessel_item = self.vessels_queue[index_value]
                observation_list.extend([
                    vessel_item["length"] / 350.0, vessel_item["draft"] / 18.0,
                    vessel_item["move_balance"] / 300.0, (vessel_item["date_due"] - self.clock_time) / 90.0,
                    vessel_item["berth_preference"] / self.quay_length, vessel_item["weight_priority"] / 3.0,
                    vessel_item["mass"] / 40000.0, vessel_item["kg"] / 15.0,
                    vessel_item["gm"] / 5.0, vessel_item["heel"] / 15.0,
                    1.0
                ])
            else:
                observation_list.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                
        for crane_index in range(int(self.crane_count)):
            observation_list.extend([
                self.cranes[crane_index, 0] / self.quay_length,
                self.cranes[crane_index, 1] / 3.0,
                self.cranes[crane_index, 2] / 100.0,
                self.cranes[crane_index, 3] / 100.0,
                self.cranes[crane_index, 4] / 60.0,
                self.cranes[crane_index, 5] / 2000.0
            ])
            
        observation_list.extend([
            self.clock_time / self.simulation_duration,
            self._calculate_tide(self.clock_time) / 6.0,
            self.speed_wind / 40.0,
            self.direction_wind / 360.0,
            self.height_wave / 6.0,
            self.period_wave / 15.0,
            self.visibility / 25.0,
            self.precipitation / 12.0,
            (self.temperature - 5.0) / 35.0,
            self.velocity_water / 3.0,
            float(len(self.vessels_queue)) / 15.0,
            float(len(self.vessels_berth)) / 10.0
        ])
        return np.array(observation_list, dtype=np.float32)

class NetworkActorCriticCMDP(nn.Module):
    """
    SECTION: Transformer-Based Actor-Critic Neural Network Architecture
    Implements multi-head self-attention and cross-attention mechanisms over heterogeneous entity embeddings (vessels, cranes, environment)
    to facilitate relational reasoning and support policy and value function estimations under CMDP constraints.
    """
    def __init__(self, state_dimension: int, dictionary_configuration: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.dictionary_configuration = dictionary_configuration or {}
        self.max_vessels = int(self.dictionary_configuration.get("queue_capacity", 15))
        self.vessel_feat_dim = 11
        self.num_cranes = int(self.dictionary_configuration.get("crane_count", 10))
        self.crane_feat_dim = 6
        self.env_feat_dim = 12
        model_dimension = int(self.dictionary_configuration.get("d_model", 200))
        heads_count = int(self.dictionary_configuration.get("num_heads", 4))

        # Linear projection layers mapping raw entity features to latent model dimension $d_{\text{model}}$.
        self.vessel_projection = nn.Linear(self.vessel_feat_dim, model_dimension)
        self.crane_projection = nn.Linear(self.crane_feat_dim, model_dimension)
        self.environment_projection = nn.Linear(self.env_feat_dim, model_dimension)
        
        # Self-attention and cross-attention multi-head transformer blocks.
        self.graph_attention = nn.MultiheadAttention(embed_dim=model_dimension, num_heads=heads_count, batch_first=True)
        self.cross_attention_layer = nn.MultiheadAttention(embed_dim=model_dimension, num_heads=heads_count, batch_first=True)
        self.layer_normalization_one = nn.LayerNorm(model_dimension)
        
        # Sequential action conditional embeddings.
        self.discrete_action_embedding = nn.Embedding(self.max_vessels, model_dimension)
        self.crane_action_embedding = nn.Embedding(int(self.dictionary_configuration.get("allocation_limit", 6)), model_dimension)
        
        # Fusion MLP layers with residual connections and layer normalization.
        self.fusion_fully_connected_one = nn.Linear(model_dimension * 3 + model_dimension + model_dimension * 2, 512)
        self.fusion_layer_normalization_one = nn.LayerNorm(512)
        self.fusion_fully_connected_two = nn.Linear(512, 512)
        self.fusion_layer_normalization_two = nn.LayerNorm(512)
        
        # Policy and value heads.
        self.actor_vessel_layer = nn.Linear(512, self.max_vessels)
        self.actor_berth_mean_layer = nn.Linear(512 + model_dimension, 1)
        self.actor_berth_scale_layer = nn.Linear(512 + model_dimension, 1)
        self.actor_crane_layer = nn.Linear(512, int(self.dictionary_configuration.get("allocation_limit", 6)))
        self.actor_ballast_mean_layer = nn.Linear(512 + model_dimension, 1)
        self.actor_ballast_scale_layer = nn.Linear(512 + model_dimension, 1)
        
        self.critic_layer = nn.Linear(512, 1)
        self.cost_critic_layer = nn.Linear(512, 1)
        
        self._initialize_weights()

    def _initialize_weights(self):
        """
        SECTION: Orthogonal Weight Initialization
        Initializes neural network linear layers using orthogonal matrix decomposition to stabilize gradient flow.
        """
        for module_item in self.modules():
            if isinstance(module_item, nn.Linear):
                nn.init.orthogonal_(module_item.weight, gain=np.sqrt(2.0))
                if module_item.bias is not None:
                    nn.init.constant_(module_item.bias, 0.0)

    def forward(self, state_tensor: torch.Tensor, mask_vessel: Optional[torch.Tensor] = None, sampled_vessel: Optional[torch.Tensor] = None, sampled_crane: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SECTION: Network Forward Propagation
        Projects raw state vectors into entity embeddings, applies attention layers, conditions actions sequentially,
        and outputs policy logits, distribution parameters, value estimates, and cost value estimates.
        """
        batch_size = state_tensor.size(0)
        v_end = self.max_vessels * self.vessel_feat_dim
        c_end = v_end + (self.num_cranes * self.crane_feat_dim)
        
        vessels_raw = state_tensor[:, :v_end].view(batch_size, self.max_vessels, self.vessel_feat_dim)
        cranes_raw = state_tensor[:, v_end:c_end].view(batch_size, self.num_cranes, self.crane_feat_dim)
        env_raw = state_tensor[:, c_end:]
        
        vessel_presence_mask = (vessels_raw[:, :, -1] > 0.0)
        crane_padding_mask = torch.ones((batch_size, self.num_cranes), dtype=torch.bool, device=state_tensor.device)
        key_padding_mask = ~torch.cat([vessel_presence_mask, crane_padding_mask], dim=1)

        v_emb = self.vessel_projection(vessels_raw)
        c_emb = self.crane_projection(cranes_raw)
        e_emb = self.environment_projection(env_raw)
        
        entities = torch.cat([v_emb, c_emb], dim=1)
        entity_valid_mask = torch.cat([vessel_presence_mask, crane_padding_mask], dim=1).float().unsqueeze(-1)
        entities = entities * entity_valid_mask
        
        # Multi-head self-attention over heterogeneous entities (vessels and cranes).
        attention_output, _ = self.graph_attention(entities, entities, entities, key_padding_mask=key_padding_mask)
        entities = self.layer_normalization_one(entities + attention_output * entity_valid_mask)
        
        entity_sum = (entities * entity_valid_mask).sum(dim=1)
        entity_count = entity_valid_mask.sum(dim=1).clamp(min=1.0)
        entity_pooled = entity_sum / entity_count
        
        entity_max = (entities * entity_valid_mask + (1.0 - entity_valid_mask) * -1e9).max(dim=1).values
        
        # Cross-attention conditioning environment queries against entity keys and values.
        cross_attended_query = e_emb.unsqueeze(1)
        cross_attended_output, _ = self.cross_attention_layer(
            query=cross_attended_query,
            key=entities,
            value=entities,
            key_padding_mask=key_padding_mask
        )
        cross_attended_environment = cross_attended_output.squeeze(1)
        
        if sampled_vessel is None:
            sampled_vessel = torch.zeros(batch_size, dtype=torch.long, device=state_tensor.device)
        else:
            sampled_vessel = torch.clamp(sampled_vessel, min=0, max=self.max_vessels - 1)

        if sampled_crane is None:
            sampled_crane = torch.zeros(batch_size, dtype=torch.long, device=state_tensor.device)
        else:
            max_crane_idx = int(self.dictionary_configuration.get("allocation_limit", 6)) - 1
            sampled_crane = torch.clamp(sampled_crane, min=0, max=max_crane_idx)
            
        vessel_action_embedded = self.discrete_action_embedding(sampled_vessel)
        crane_action_embedded = self.crane_action_embedding(sampled_crane)
        
        # Concatenate pooled entity representations, max entity features, environment embeddings, cross-attention output, and conditional action embeddings.
        combined = torch.cat([entity_pooled, entity_max, e_emb, cross_attended_environment, vessel_action_embedded, crane_action_embedded], dim=1)
        hidden_layer_one = F.tanh(self.fusion_layer_normalization_one(self.fusion_fully_connected_one(combined)))
        hidden_layer_two = F.tanh(self.fusion_layer_normalization_two(self.fusion_fully_connected_two(hidden_layer_one) + hidden_layer_one))
        
        vessel_logits = self.actor_vessel_layer(hidden_layer_two)
        if mask_vessel is not None:
            vessel_logits = vessel_logits + (1.0 - mask_vessel) * -1e9
            
        conditional_berth_input = torch.cat([hidden_layer_two, vessel_action_embedded], dim=1)
        berth_mean = self.actor_berth_mean_layer(conditional_berth_input)
        berth_scale = F.softplus(self.actor_berth_scale_layer(conditional_berth_input)) + 1e-4
        
        crane_logits = self.actor_crane_layer(hidden_layer_two)
        
        conditional_ballast_input = torch.cat([hidden_layer_two, crane_action_embedded], dim=1)
        ballast_mean = self.actor_ballast_mean_layer(conditional_ballast_input)
        ballast_scale = F.softplus(self.actor_ballast_scale_layer(conditional_ballast_input)) + 1e-4
        
        value_tensor = self.critic_layer(hidden_layer_two)
        cost_value_tensor = self.cost_critic_layer(hidden_layer_two)
        
        return vessel_logits, berth_mean, berth_scale, crane_logits, ballast_mean, ballast_scale, value_tensor, cost_value_tensor

class AgentPPOCMDP:
    """
    SECTION: PPO Agent with Lagrangian Constraint Optimization
    Implements Proximal Policy Optimization (PPO) with Generalized Advantage Estimation (GAE) for both reward returns and cost returns,
    maintaining dual value and cost critics, and adapting the Lagrangian multiplier to enforce safety constraints.
    """
    def __init__(self, state_dimension: int, dictionary_configuration: Optional[Dict[str, Any]] = None):
        self.dictionary_configuration = dictionary_configuration or {}
        self.model = NetworkActorCriticCMDP(state_dimension, self.dictionary_configuration)
        learning_rate = float(self.dictionary_configuration.get("lr", 3.0e-4))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.gamma = float(self.dictionary_configuration.get("gamma", 0.99))
        self.gae_lambda = float(self.dictionary_configuration.get("gae_lambda", 0.95))
        self.clip_eps = float(self.dictionary_configuration.get("clip_eps", 0.2))
        
        self.lagrangian_multiplier = 0.1
        self.cost_limit = 0.05
        
        self.states = []
        self.actions = []
        self.masks = []
        self.log_probs = []
        self.rewards = []
        self.costs = []
        self.values = []
        self.cost_values = []
        self.dones = []

    def select_action(self, observation_array: np.ndarray, action_mask: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        SECTION: Action Selection
        Samples discrete and continuous actions from actor distributions given current observation and action validity masks.
        """
        state_tensor = torch.FloatTensor(observation_array).unsqueeze(0)
        mask_tensor = torch.FloatTensor(action_mask["vessel_mask"]).unsqueeze(0)
        
        with torch.no_grad():
            v_log, _, _, _, _, _, _, _ = self.model(state_tensor, mask_vessel=mask_tensor)
            v_dist = torch.distributions.Categorical(logits=v_log)
            v_act = v_dist.sample()
            
            c_log, _, _, _, _, _, _, _ = self.model(state_tensor, mask_vessel=mask_tensor, sampled_vessel=v_act)
            c_dist = torch.distributions.Categorical(logits=c_log)
            c_act = c_dist.sample()
            
            _, b_mean, b_scale, _, ball_mean, ball_scale, value_tensor, cost_value_tensor = self.model(state_tensor, mask_vessel=mask_tensor, sampled_vessel=v_act, sampled_crane=c_act)
            
            v_prob = v_dist.log_prob(v_act)
            c_prob = c_dist.log_prob(c_act)
            
            # Normal distribution sampling with log probability adjustments for sigmoid squashing transformation.
            b_dist = torch.distributions.Normal(b_mean, b_scale)
            b_raw = b_dist.sample()
            b_act = torch.sigmoid(b_raw)
            b_prob = b_dist.log_prob(b_raw).sum(dim=-1) - torch.log(b_act * (1.0 - b_act) + 1e-6).sum(dim=-1)
            
            ball_dist = torch.distributions.Normal(ball_mean, ball_scale)
            ball_raw = ball_dist.sample()
            ball_act = torch.sigmoid(ball_raw)
            ball_prob = ball_dist.log_prob(ball_raw).sum(dim=-1) - torch.log(ball_act * (1.0 - ball_act) + 1e-6).sum(dim=-1)
            
            log_probability_total = v_prob + b_prob + c_prob + ball_prob
            
        action_dictionary = {
            "vessel_idx": v_act.item(),
            "berth_fraction": b_act.cpu().numpy().flatten(),
            "crane_count": c_act.item(),
            "ballast_rate": (ball_act.cpu().numpy().flatten() * 2.0) - 1.0
        }
        return action_dictionary, log_probability_total, value_tensor, cost_value_tensor

    def store_transition(self, observation_array, action_dictionary, action_mask, log_probability_tensor, reward, cost, value_tensor, cost_value_tensor, done):
        """
        SECTION: Transition Storage
        Appends state, action, mask, log probability, reward, cost, value, cost value, and termination flag to rollout memory buffers.
        """
        self.states.append(observation_array)
        self.actions.append(action_dictionary)
        self.masks.append(action_mask["vessel_mask"])
        self.log_probs.append(log_probability_tensor)
        self.rewards.append(reward)
        self.costs.append(cost)
        self.values.append(value_tensor)
        self.cost_values.append(cost_value_tensor)
        self.dones.append(done)

    def update_policy(self, state_next: np.ndarray, mask_next: Dict[str, np.ndarray], epochs: int = 5, batch_size: int = 128, current_epoch: int = 0, max_epochs: int = 10):
        """
        SECTION: Policy Update and Lagrangian Optimization
        Computes GAE advantages for rewards and constraint costs, performs PPO clipped policy updates incorporating constraint penalties,
        optimizes value and cost critic losses, and updates the Lagrangian multiplier based on empirical constraint violations.
        """
        states = torch.FloatTensor(np.array(self.states))
        masks = torch.FloatTensor(np.array(self.masks))
        b_vessels = torch.LongTensor([a["vessel_idx"] for a in self.actions])
        
        b_berths = torch.FloatTensor(np.array([a["berth_fraction"] for a in self.actions]))
        b_cranes = torch.LongTensor([a["crane_count"] for a in self.actions])
        b_ballasts = torch.FloatTensor(np.array([(a["ballast_rate"] + 1.0) / 2.0 for a in self.actions]))
        
        old_log_probs = torch.stack(self.log_probs).detach()
        rewards = self.rewards
        costs = self.costs
        values = torch.cat(self.values).detach().view(-1)
        cost_values = torch.cat(self.cost_values).detach().view(-1)
        dones = self.dones

        with torch.no_grad():
            state_next_tensor = torch.FloatTensor(state_next).unsqueeze(0)
            mask_next_tensor = torch.FloatTensor(mask_next["vessel_mask"]).unsqueeze(0)
            _, _, _, _, _, _, next_value, next_cost_value = self.model(state_next_tensor, mask_vessel=mask_next_tensor)
            
        advantages = torch.zeros_like(values)
        cost_advantages = torch.zeros_like(cost_values)
        last_gae_lam = 0.0
        last_cost_gae = 0.0
        values_extended = torch.cat([values, next_value.detach().view(-1)])
        cost_values_extended = torch.cat([cost_values, next_cost_value.detach().view(-1)])
        
        # Generalized Advantage Estimation (GAE) calculation for both reward returns and constraint cost returns.
        for t in reversed(range(len(rewards))):
            non_terminal_flag = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * values_extended[t + 1] * non_terminal_flag - values_extended[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * non_terminal_flag * last_gae_lam
            advantages[t] = last_gae_lam
            
            cost_delta = costs[t] + self.gamma * cost_values_extended[t + 1] * non_terminal_flag - cost_values_extended[t]
            last_cost_gae = cost_delta + self.gamma * self.gae_lambda * non_terminal_flag * last_cost_gae
            cost_advantages[t] = last_cost_gae

        returns = advantages + values
        cost_returns = cost_advantages + cost_values
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        cost_advantages = (cost_advantages - cost_advantages.mean()) / (cost_advantages.std() + 1e-8)
        
        total_advantage = advantages

        entropy_coefficient = max(0.001, 0.05 * (1.0 - float(current_epoch) / float(max(1, max_epochs))))

        dataset_size = states.size(0)
        for _ in range(epochs):
            indices = np.random.permutation(dataset_size)
            for start_idx in range(0, dataset_size, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                b_states = states[batch_indices]
                b_batch_masks = masks[batch_indices]
                batch_vessels = b_vessels[batch_indices]
                batch_berths = b_berths[batch_indices]
                batch_cranes = b_cranes[batch_indices]
                batch_ballasts = b_ballasts[batch_indices]
                b_old_log_probs = old_log_probs[batch_indices]
                b_advantages = total_advantage[batch_indices]
                b_returns = returns[batch_indices]
                b_cost_returns = cost_returns[batch_indices]
                b_cost_advantages = cost_advantages[batch_indices]

                v_log, _, _, _, _, _, _, _ = self.model(b_states, mask_vessel=b_batch_masks)
                v_dist = torch.distributions.Categorical(logits=v_log)
                new_v_lp = v_dist.log_prob(batch_vessels)
                v_ent = v_dist.entropy().mean()
                
                c_log, _, _, _, _, _, _, _ = self.model(b_states, mask_vessel=b_batch_masks, sampled_vessel=batch_vessels)
                c_dist = torch.distributions.Categorical(logits=c_log)
                new_c_lp = c_dist.log_prob(batch_cranes)
                c_ent = c_dist.entropy().mean()
                
                _, b_mean, b_scale, _, ball_mean, ball_scale, b_values, b_cost_values = self.model(b_states, mask_vessel=b_batch_masks, sampled_vessel=batch_vessels, sampled_crane=batch_cranes)
                b_values = b_values.view(-1)
                b_cost_values = b_cost_values.view(-1)
                
                b_dist = torch.distributions.Normal(b_mean, b_scale)
                clamped_b = torch.logit(torch.clamp(batch_berths, min=1e-5, max=1.0 - 1e-5))
                new_b_lp = b_dist.log_prob(clamped_b).sum(dim=-1) - torch.log(batch_berths * (1.0 - batch_berths) + 1e-6).sum(dim=-1)
                b_ent = b_dist.entropy().mean()
                
                ball_dist = torch.distributions.Normal(ball_mean, ball_scale)
                clamped_ball = torch.logit(torch.clamp(batch_ballasts, min=1e-5, max=1.0 - 1e-5))
                new_ball_lp = ball_dist.log_prob(clamped_ball).sum(dim=-1) - torch.log(batch_ballasts * (1.0 - batch_ballasts) + 1e-6).sum(dim=-1)
                ball_ent = ball_dist.entropy().mean()
                
                new_log_probs = new_v_lp + new_b_lp + new_c_lp + new_ball_lp
                entropy = v_ent + b_ent + c_ent + ball_ent

                # PPO clipped surrogate objective and Lagrangian constraint penalty term.
                ratios = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                constraint_penalty = (self.lagrangian_multiplier * (ratios * b_cost_advantages)).mean()
                
                value_loss = F.mse_loss(b_values, b_returns)
                cost_value_loss = F.mse_loss(b_cost_values, b_cost_returns)
                loss = policy_loss + constraint_penalty + 0.5 * value_loss + 0.5 * cost_value_loss - entropy_coefficient * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

        mean_episode_cost = sum(self.costs) / max(1, len(self.costs))
        error = mean_episode_cost - self.cost_limit
        
        # Lagrangian multiplier gradient ascent update rule: $\lambda \leftarrow \text{clip}(\lambda + \alpha_{\lambda} (\bar{C} - C_{\text{limit}}), 0, 5)$.
        self.lagrangian_multiplier = float(np.clip(self.lagrangian_multiplier + 0.01 * error, 0.0, 5.0))

        self.states.clear()
        self.actions.clear()
        self.masks.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.costs.clear()
        self.values.clear()
        self.cost_values.clear()
        self.dones.clear()

def execute_simulation():
    """
    SECTION: Simulation Training and Animation Rendering
    Initializes environment and PPO agent configurations, runs training epochs across simulation episodes, collects evaluation trajectories,
    and sets up Matplotlib figures and animation loops to render quayside operational states and vessel stability telemetry dynamically.
    """
    env_config = {
        "quay_length": 1200.0,
        "crane_count": 10,
        "crane_distance": 30.0,
        "simulation_duration": 168,
        "dt": 0.25,
        "queue_capacity": 15,
        "initial_vessel_count": 6,
        "allocation_limit": 10,
        "gamma": 0.99,
        "lr": 3.0e-4
    }
    env = EnvironmentCMDPStability(env_config)
    agent = AgentPPOCMDP(env.observation_space.shape[0], env_config)
    episode_count = 15
    
    for episode in range(episode_count):
        state, information = env.reset()
        done = False
        epoch_reward_sum = 0.0
        while not done:
            action, log_probability_tensor, value_tensor, cost_value_tensor = agent.select_action(state, information)
            state_next, reward, termination_indicator, truncation_indicator, information_next = env.step(action)
            done = termination_indicator or truncation_indicator
            epoch_reward_sum += reward
            cost = information_next.get("constraint_cost", 0.0)
            agent.store_transition(state, action, information, log_probability_tensor, reward, cost, value_tensor, cost_value_tensor, done)
            state = state_next
            information = information_next
        agent.update_policy(state, information, current_epoch=episode, max_epochs=episode_count)
        print(f"Training Epoch {episode + 1} Reward: {epoch_reward_sum:.2f} Lambda: {agent.lagrangian_multiplier:.3f}")

    state, information = env.reset()
    done = False
    trajectory_collection = []
    
    while not done:
        action, _, _, _ = agent.select_action(state, information)
        state_next, reward, termination_indicator, truncation_indicator, information_next = env.step(action)
        done = termination_indicator or truncation_indicator
        trajectory_collection.append({
            "time": env.clock_time,
            "tide": env._calculate_tide(env.clock_time),
            "wind": env.speed_wind,
            "wave": env.height_wave,
            "vis": env.visibility,
            "precip": env.precipitation,
            "queue_len": len(env.vessels_queue),
            "berthed": [dict(v) for v in env.vessels_berth],
            "completed": len(env.vessels_finish),
            "cranes": env.cranes.copy(),
            "lambda": agent.lagrangian_multiplier,
            "observation": state.copy()
        })
        state = state_next
        information = information_next

    # Matplotlib animation setup and rendering.
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 11), facecolor='#030712')
    
    axis_quay = plt.subplot2grid((3, 1), (0, 0), rowspan=1, colspan=1, fig=fig)
    axis_stability = plt.subplot2grid((3, 1), (1, 0), rowspan=1, colspan=1, fig=fig)
    axis_status = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1, fig=fig)

    for ax in [axis_quay, axis_stability, axis_status]:
        ax.set_facecolor('#0f172a')
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.xaxis.label.set_color('#cbd5e1')
        ax.yaxis.label.set_color('#cbd5e1')
        ax.title.set_color('#f8fafc')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1e293b')
            spine.set_linewidth(1.5)

    axis_quay.set_xlim(0, env.quay_length)
    axis_quay.set_ylim(-5, 50)
    axis_quay.set_title("Quayside Operations Twin with Robust Non-Hacking Reward", fontsize=12, fontweight='bold', pad=8)

    axis_stability.set_xlim(0, env.simulation_duration)
    axis_stability.set_ylim(-10, 10)
    axis_stability.set_title("Vessel GM and Heel Telemetry", fontsize=12, fontweight='bold', pad=8)

    axis_status.axis('off')
    status_box = FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.03", ec='#38bdf8', fc='#090d16', lw=1.5, transform=axis_status.transAxes)
    axis_status.add_patch(status_box)
    status_text = axis_status.text(0.05, 0.5, "", color='#38bdf8', fontsize=10, family='monospace', va='center')

    def init():
        return status_text,

    def animate(frame_index):
        axis_quay.clear()
        axis_quay.set_facecolor('#0f172a')
        axis_quay.set_xlim(0, env.quay_length)
        axis_quay.set_ylim(-5, 50)
        axis_quay.set_title("Quayside Operations Twin and Reward", fontsize=12, fontweight='bold', pad=8)
        
        data = trajectory_collection[frame_index]
        
        wave_x = np.linspace(0, env.quay_length, 200)
        wave_y = 2.0 * np.sin(wave_x * 0.02 + frame_index * 0.2) * (data["wave"] / 2.0)
        axis_quay.fill_between(wave_x, -5, wave_y, color='#0284c7', alpha=0.25)
        axis_quay.plot(wave_x, wave_y, color='#38bdf8', lw=1.0, alpha=0.5)

        axis_quay.plot([0, env.quay_length], [0, 0], color='#cbd5e1', lw=6, solid_capstyle='round')
        
        for vessel_item in data["berthed"]:
            rect_glow = Rectangle((vessel_item["coordinate"], 3), vessel_item["length"], 18, facecolor='#1d4ed8', edgecolor='#60a5fa', alpha=0.4, lw=3, zorder=3)
            rect_core = Rectangle((vessel_item["coordinate"], 5), vessel_item["length"], 14, facecolor='#2563eb', edgecolor='#93c5fd', alpha=0.9, lw=1.5, zorder=4)
            axis_quay.add_patch(rect_glow)
            axis_quay.add_patch(rect_core)
            axis_quay.text(vessel_item["coordinate"] + 8, 12.0, f"ID #{vessel_item['id']} [GM:{vessel_item['gm']:.1f}m Heel:{vessel_item['heel']:.1f}°]", color='#f8fafc', fontsize=9, fontweight='bold', zorder=5)

        for crane_index, crane_data in enumerate(data["cranes"]):
            crane_x = crane_data[0]
            crane_color = '#ef4444' if crane_data[1] == 3.0 else '#f59e0b'
            axis_quay.plot([crane_x, crane_x], [0, 32], color=crane_color, lw=3, zorder=6)
            axis_quay.plot([crane_x - 6, crane_x + 6], [32, 32], color=crane_color, lw=3, zorder=6)
            axis_quay.scatter([crane_x], [32], color='#ffffff', s=35, zorder=7)

        axis_stability.clear()
        axis_stability.set_facecolor('#0f172a')
        axis_stability.set_xlim(0, env.simulation_duration)
        axis_stability.set_ylim(-10, 10)
        axis_stability.set_title("Vessel GM and Heel Telemetry", fontsize=12, fontweight='bold', pad=8)
        
        times = [t["time"] for t in trajectory_collection[:frame_index + 1]]
        gm_vals = [t["berthed"][0]["gm"] if len(t["berthed"]) > 0 else 0.0 for t in trajectory_collection[:frame_index + 1]]
        heel_vals = [t["berthed"][0]["heel"] if len(t["berthed"]) > 0 else 0.0 for t in trajectory_collection[:frame_index + 1]]
        
        if len(times) > 0:
            axis_stability.plot(times, gm_vals, color='#38bdf8', lw=2, label='GM (m)')
            axis_stability.plot(times, heel_vals, color='#f43f5e', lw=2, label='Heel (°)')
            axis_stability.legend(loc='upper right', facecolor='#0f172a', edgecolor='#1e293b', fontsize=8)

        status_text.set_text(
            f"TIME: {data['time']:5.1f}h  |  TIDE: {data['tide']:5.2f}m  |  WIND: {data['wind']:4.1f}m/s  |  WAVE: {data['wave']:5.1f}m\n"
            f"VIS: {data['vis']:4.1f}km  |  PRECIP: {data['precip']:4.2f}mm/h  |  QUEUE: {data['queue_len']} vessels\n"
            f"COMPLETED: {data['completed']} vessels  |  LAGRANGIAN MULTIPLIER (LAMBDA): {data['lambda']:.4f}"
        )
        return status_text,

    animation_instance = animation.FuncAnimation(fig, animate, init_func=init, frames=len(trajectory_collection), interval=60, blit=False, repeat=True)
    plt.tight_layout(pad=3.0)
    plt.show()

if __name__ == "__main__":
    execute_simulation()