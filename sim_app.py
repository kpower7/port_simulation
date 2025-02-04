import streamlit as st
import simpy
import random
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# -----------------------------------------------------------
# SHIPS_DATA and default config
# -----------------------------------------------------------
DEFAULT_SHIPS_DATA = [
    {"name": "CEZANNE", "containers": 5642},
    {"name": "CCNI ANDES", "containers": 4338},
    {"name": "CMA CGM LEO", "containers": 4187},
    {"name": "ONE REINFORCEMENT", "containers": 3752},
    {"name": "POLAR ECUADOR", "containers": 3431},
    {"name": "W KITHIRA", "containers": 2311},
    {"name": "SUAPE EXPRESS", "containers": 2062},
    {"name": "MAERSK ATHABASCA", "containers": 1998},
    {"name": "MAERSK SENTOSA", "containers": 1696},
    {"name": "MAERSK MONTE LINZOR", "containers": 1392},
    {"name": "METHONI", "containers": 942},
    {"name": "MAERSK CAPTAIN", "containers": 742},
    {"name": "MAERSK ONE REINFORCEMENT", "containers": 752},
    {"name": "MAERSK SUI", "containers": 642},
    {"name": "MAERSK KATHIRA", "containers": 542},
    {"name": "MAERSK NIOVI", "containers": 442},
]

MAX_BERTHS_DEFAULT = 3
CRANES_PER_BERTH_DEFAULT = 4
CONTAINER_UNLOAD_TIME_DEFAULT = {"mean": 2.5, "std": 0.5}  # minutes
BERTH_TRANSITION_TIME_DEFAULT = 60  # minutes

CONTAINER_TYPES_DEFAULT = {
    "dry": {
        "probability": 0.95,
        "dwell_time": {
            "mean": 2.47,  # days
            "std": 1.5     # days
        }
    },
    "reefer": {
        "probability": 0.05,
        "dwell_time": {
            "mean": 2.1,   # days
            "std": 0.8     # days
        }
    }
}

MODAL_SPLIT_DEFAULT = {"truck": 0.85, "train": 0.15}

GATE_HOURS_DEFAULT = {
    "open": 7,    # 7 AM
    "close": 17,  # 5 PM
    "trucks_per_hour": 500
}

TRAIN_SCHEDULE_DEFAULT = {"trains_per_day": 6, "capacity": 250}


# -----------------------------------------------------------
# Container & Ship classes
# -----------------------------------------------------------
class Container:
    def __init__(self, id, arrival_time, container_types, modal_split):
        self.id = id
        self.arrival_time = arrival_time

        # Flag to mark new containers (arriving on ships) so we only track
        # dwell times for these containers
        self.is_new = True  

        # Reefer probability (with minor random variation)
        reefer_prob = max(
            0,
            min(1, random.normalvariate(
                container_types["reefer"]["probability"],
                container_types["reefer"]["probability"] * 0.1
            ))
        )
        self.type = 'reefer' if random.random() < reefer_prob else 'dry'

        # Modal probability (with minor random variation)
        train_prob = max(
            0,
            min(1, random.normalvariate(
                modal_split["train"],
                modal_split["train"] * 0.067
            ))
        )
        self.modal = 'train' if random.random() < train_prob else 'truck'

        self.yard_entry_time = None
        self.ready_time = None
        self.departure_time = None

        # Dwell time (days -> minutes)
        if self.type == 'reefer':
            dwell_time = random.normalvariate(
                container_types["reefer"]["dwell_time"]["mean"],
                container_types["reefer"]["dwell_time"]["std"]
            ) * 24 * 60
        else:
            dwell_time = random.normalvariate(
                container_types["dry"]["dwell_time"]["mean"],
                container_types["dry"]["dwell_time"]["std"]
            ) * 24 * 60

        self.dwell_time = max(60, dwell_time)  # minimum 1 hour dwell


class Ship:
    def __init__(self, env, id, container_count, container_types, modal_split):
        self.env = env
        self.id = id
        self.container_count = container_count
        self.container_types = container_types
        self.modal_split = modal_split
        self.containers = self._generate_containers()

    def _generate_containers(self):
        containers = []
        for i in range(self.container_count):
            containers.append(Container(
                f"{self.id}_container_{i}",
                self.env.now,
                self.container_types,
                self.modal_split
            ))
        return containers


# -----------------------------------------------------------
# Statistics class (with berth/crane wait times and train logs)
# -----------------------------------------------------------
class Statistics:
    def __init__(self, env, sim_start_time=None):
        """
        sim_start_time: optional datetime from which we track the simulation time. 
                       If provided, we can convert env.now (in minutes) to an actual datetime.
        """
        self.env = env
        self.sim_start_time = sim_start_time

        # Container processing
        self.containers = {
            'total': 0,
            'dry': {'truck': 0, 'train': 0},
            'reefer': {'truck': 0, 'train': 0}
        }
        self.arrivals = {
            'ships': 0,
            'containers': {
                'total': 0,
                'dry': 0,
                'reefer': 0
            }
        }
        self.departures = {
            'truck': {
                'total': 0,
                'by_hour': [0] * 24,
                'dry': 0,
                'reefer': 0
            },
            'train': {
                'total': 0,
                'by_hour': [0] * 24,
                'dry': 0,
                'reefer': 0
            }
        }

        # Yard stats
        self.yard_state = {
            'current': {'dry': 0, 'reefer': 0},
            'max': {'dry': 0, 'reefer': 0}
        }
        self.yard_full_events = 0

        # Dwell times
        self.dwell_times = {
            'dry': [],
            'reefer': []
        }

        # Wait times (in minutes)
        self.wait_times = {
            'ship': [],    # total ship wait
            'gate': [],
            'train': [],
            'berth': [],
            'crane': []
        }

        self.missed_train_connections = 0

        # Hourly time series for arrivals & departures
        self.hourly_stats = {
            'arrivals': defaultdict(lambda: defaultdict(int)),
            'departures': {
                'truck': defaultdict(lambda: defaultdict(int)),
                'train': defaultdict(lambda: defaultdict(int))
            }
        }
        
        # Yard utilization by hour
        self.yard_utilization = {
            'dry': {'truck': defaultdict(int), 'train': defaultdict(int)},
            'reefer': {'truck': defaultdict(int), 'train': defaultdict(int)}
        }

        # Resource usage by hour (0–1 fraction)
        self.resource_usage_by_hour = {
            'berths': defaultdict(float),
            'cranes': defaultdict(float),
            'gate': defaultdict(float),
            'yard': defaultdict(float)
        }

        # Train departure records
        self.train_departure_records = []

    # ------------------------
    # Logging methods
    # ------------------------
    def log_ship_arrival(self):
        self.arrivals['ships'] += 1

    def log_container_arrival(self, container_type):
        self.arrivals['containers']['total'] += 1
        self.arrivals['containers'][container_type] += 1
        current_hour = int(self.env.now / 60)
        self.hourly_stats['arrivals'][current_hour][container_type] += 1

    def log_container_departure(self, container_type, modal, hour):
        self.departures[modal]['total'] += 1
        self.departures[modal]['by_hour'][hour] += 1
        self.departures[modal][container_type] += 1
        
        current_hour = int(self.env.now / 60)
        self.hourly_stats['departures'][modal][current_hour][container_type] += 1

    def update_yard_state(self, dry_count, reefer_count):
        self.yard_state['current']['dry'] = dry_count
        self.yard_state['current']['reefer'] = reefer_count
        self.yard_state['max']['dry'] = max(self.yard_state['max']['dry'], dry_count)
        self.yard_state['max']['reefer'] = max(self.yard_state['max']['reefer'], reefer_count)
        
        current_hour = int(self.env.now / 60)
        for ctype in ['dry', 'reefer']:
            for modal in ['truck', 'train']:
                self.yard_utilization[ctype][modal][current_hour] = self.yard_state['current'][ctype]

    def log_container(self, container_type, modal):
        self.containers['total'] += 1
        self.containers[container_type][modal] += 1

    def log_dwell_time(self, container_type, dwell_time):
        # Called only if container is truly new
        self.dwell_times[container_type].append(dwell_time)

    def log_wait_time(self, wait_type, wait_time):
        self.wait_times[wait_type].append(wait_time)

    def log_yard_full(self):
        self.yard_full_events += 1

    def log_missed_train(self):
        self.missed_train_connections += 1

    def log_train_departure(self, dry_loaded, reefer_loaded, capacity):
        """Record a single train departure event."""
        total_loaded = dry_loaded + reefer_loaded
        # Convert simulation minutes into an actual datetime if sim_start_time is given
        if self.sim_start_time:
            departure_dt = self.sim_start_time + timedelta(minutes=self.env.now)
        else:
            # Otherwise just store simulation time (in minutes)
            departure_dt = self.env.now

        self.train_departure_records.append({
            "Departure Time": departure_dt,
            "Dry Loaded": dry_loaded,
            "Reefer Loaded": reefer_loaded,
            "Total Loaded": total_loaded,
            "Capacity": capacity
        })

    # ------------------------
    # Summary method
    # ------------------------
    def get_summary(self):
        total = self.containers['total']
        dry_total = self.containers['dry']['truck'] + self.containers['dry']['train']
        reefer_total = self.containers['reefer']['truck'] + self.containers['reefer']['train']
        
        return {
            'Arrivals': {
                'Ships': self.arrivals['ships'],
                'Containers': {
                    'Total': self.arrivals['containers']['total'],
                    'Dry': self.arrivals['containers']['dry'],
                    'Reefer': self.arrivals['containers']['reefer']
                }
            },
            'Departures': {
                'Truck': {
                    'Total': self.departures['truck']['total'],
                    'Dry': self.departures['truck']['dry'],
                    'Reefer': self.departures['truck']['reefer'],
                    'Peak Hour': (max(enumerate(self.departures['truck']['by_hour']), 
                                      key=lambda x: x[1])[0]
                                  if any(self.departures['truck']['by_hour']) else 0)
                },
                'Train': {
                    'Total': self.departures['train']['total'],
                    'Dry': self.departures['train']['dry'],
                    'Reefer': self.departures['train']['reefer'],
                    'Peak Hour': (max(enumerate(self.departures['train']['by_hour']), 
                                      key=lambda x: x[1])[0]
                                  if any(self.departures['train']['by_hour']) else 0)
                }
            },
            'Yard State': {
                'Current': self.yard_state['current'],
                'Maximum': self.yard_state['max']
            },
            'Container Counts': {
                'Total Processed': total,
                'By Type': {
                    'Dry': {
                        'Total': dry_total,
                        'Truck': self.containers['dry']['truck'],
                        'Train': self.containers['dry']['train']
                    },
                    'Reefer': {
                        'Total': reefer_total,
                        'Truck': self.containers['reefer']['truck'],
                        'Train': self.containers['reefer']['train']
                    }
                },
                'Modal Split': {
                    'Truck': ((self.containers['dry']['truck'] + self.containers['reefer']['truck'])
                              / total if total > 0 else 0),
                    'Train': ((self.containers['dry']['train'] + self.containers['reefer']['train'])
                              / total if total > 0 else 0)
                }
            },
            'Dwell Times (days)': {
                'Dry': {
                    'Mean': np.mean(self.dwell_times['dry']) / (24 * 60) if self.dwell_times['dry'] else 0,
                    'Median': np.median(self.dwell_times['dry']) / (24 * 60) if self.dwell_times['dry'] else 0,
                    'Std Dev': np.std(self.dwell_times['dry']) / (24 * 60) if self.dwell_times['dry'] else 0,
                    '95th Percentile': (np.percentile(self.dwell_times['dry'], 95) / (24 * 60)
                                        if self.dwell_times['dry'] else 0)
                },
                'Reefer': {
                    'Mean': np.mean(self.dwell_times['reefer']) / (24 * 60) if self.dwell_times['reefer'] else 0,
                    'Median': np.median(self.dwell_times['reefer']) / (24 * 60) if self.dwell_times['reefer'] else 0,
                    'Std Dev': np.std(self.dwell_times['reefer']) / (24 * 60) if self.dwell_times['reefer'] else 0,
                    '95th Percentile': (np.percentile(self.dwell_times['reefer'], 95) / (24 * 60)
                                        if self.dwell_times['reefer'] else 0)
                }
            },
            'Wait Times (minutes)': {
                'Ship': self.wait_times['ship'],
                'Berth': self.wait_times['berth'],
                'Crane': self.wait_times['crane'],
                'Gate': self.wait_times['gate'],
                'Train': self.wait_times['train']
            },
            'Operational Issues': {
                'Yard Full Events': self.yard_full_events,
                'Missed Train Connections': self.missed_train_connections
            }
        }


# -----------------------------------------------------------
# PortSimulation
# -----------------------------------------------------------
class PortSimulation:
    def __init__(
        self,
        max_berths,
        cranes_per_berth,
        unload_time_mean,
        unload_time_std,
        berth_transition_time,
        container_types,
        modal_split,
        gate_hours,
        ships_data,
        trains_per_day,
        train_capacity,
        starting_yard_util_percent=50,  # NEW: Default 50%
        simulation_start=None  # optional datetime
    ):
        self.env = simpy.Environment()
        self.max_berths = max_berths
        self.cranes_per_berth = cranes_per_berth
        self.unload_time_mean = unload_time_mean
        self.unload_time_std = unload_time_std
        self.berth_transition_time = berth_transition_time
        self.container_types = container_types
        self.modal_split = modal_split
        self.gate_hours = gate_hours
        self.ships_data = ships_data

        # Train parameters
        self.trains_per_day = trains_per_day
        self.train_capacity = train_capacity

        # Resource capacities
        self.berths = simpy.Resource(self.env, capacity=self.max_berths)
        self.cranes = simpy.Resource(self.env, capacity=self.max_berths * self.cranes_per_berth)
        self.gate = simpy.Resource(self.env, capacity=50)

        # Yards (regular vs reefer, split by truck vs train)
        self.yard_containers = {
            'dry': {'truck': [], 'train': []},
            'reefer': {'truck': [], 'train': []}
        }
        self.regular_yard = {
            'truck': simpy.Container(self.env, capacity=int(25000 * modal_split["truck"])),
            'train': simpy.Container(self.env, capacity=int(25000 * modal_split["train"]))
        }
        self.reefer_yard = {
            'truck': simpy.Container(self.env, capacity=int(2000 * modal_split["truck"])),
            'train': simpy.Container(self.env, capacity=int(2000 * modal_split["train"]))
        }

        # Fill yard at start (proportionally) with "dummy" containers
        # that are not tracked in dwell times
        fraction = starting_yard_util_percent / 100.0
        for yard_type in (self.regular_yard, self.reefer_yard):
            for modal_key in yard_type.keys():
                initial_fill = int(yard_type[modal_key].capacity * fraction)
                if initial_fill > 0:
                    yard_type[modal_key].put(initial_fill)

        # Build train schedule (times within 24-hr period)
        interval = 24 / self.trains_per_day
        self.train_times = [interval * i for i in range(self.trains_per_day)]

        # Statistics
        self.stats = Statistics(self.env, sim_start_time=simulation_start)

        # Start processes
        self.env.process(self.schedule_trains())
        self.env.process(self.monitor_utilization())

    def schedule_trains(self):
        """Dispatch trains at the scheduled times each day."""
        while True:
            current_hour = (self.env.now / 60) % 24
            # Find next train time
            next_train = next((t for t in self.train_times if t > current_hour), None)

            if next_train is None:
                # No more trains today; wait until the first train tomorrow
                wait_time = (24 - current_hour + self.train_times[0]) * 60
            else:
                wait_time = (next_train - current_hour) * 60

            yield self.env.timeout(wait_time)
            self.env.process(self.handle_train_departure())

    def monitor_utilization(self):
        """
        Measure resource usage (berths, cranes, gate, yard) every hour.
        Usage is (# in use) / (capacity).
        """
        while True:
            current_hour = int(self.env.now // 60)

            # Berths usage: (# in use) / total capacity
            berths_in_use = self.berths.count
            self.stats.resource_usage_by_hour['berths'][current_hour] = (
                berths_in_use / self.berths.capacity if self.berths.capacity > 0 else 0
            )

            # Cranes usage
            cranes_in_use = self.cranes.count
            self.stats.resource_usage_by_hour['cranes'][current_hour] = (
                cranes_in_use / self.cranes.capacity if self.cranes.capacity > 0 else 0
            )

            # Gate usage
            gate_in_use = self.gate.count
            self.stats.resource_usage_by_hour['gate'][current_hour] = (
                gate_in_use / self.gate.capacity if self.gate.capacity > 0 else 0
            )

            # Yard usage
            reg_truck = self.regular_yard['truck'].level
            reg_train = self.regular_yard['train'].level
            ref_truck = self.reefer_yard['truck'].level
            ref_train = self.reefer_yard['train'].level
            total_used = reg_truck + reg_train + ref_truck + ref_train
            total_capacity = (self.regular_yard['truck'].capacity + self.regular_yard['train'].capacity +
                              self.reefer_yard['truck'].capacity + self.reefer_yard['train'].capacity)
            yard_frac = total_used / total_capacity if total_capacity > 0 else 0
            self.stats.resource_usage_by_hour['yard'][current_hour] = yard_frac

            yield self.env.timeout(60)

    def generate_ship_arrivals(self):
        """Generate ships continuously based on SHIPS_DATA distribution."""
        ship_id = 0
        while True:
            container_count = np.random.choice([s["containers"] for s in self.ships_data])
            ship_obj = Ship(self.env, f"ship_{ship_id}", container_count, self.container_types, self.modal_split)
            self.env.process(self.handle_ship(ship_obj))
            
            # 12–36 hours between arrivals
            yield self.env.timeout(random.uniform(12*60, 36*60))
            ship_id += 1

    def handle_ship(self, ship):
        """Process ship arrival, berth wait, unloading, berth transition."""
        arrival_time = self.env.now
        self.stats.log_ship_arrival()
        
        # Request berth
        berth_req_start = self.env.now
        with self.berths.request() as berth_req:
            yield berth_req
            berth_wait = self.env.now - berth_req_start
            self.stats.log_wait_time('berth', berth_wait)

            # Unload containers concurrently
            unload_procs = []
            for c in ship.containers:
                self.stats.log_container_arrival(c.type)
                unload_procs.append(self.env.process(self.unload_container(c)))
            yield simpy.events.AllOf(self.env, unload_procs)

            # Berth transition
            yield self.env.timeout(self.berth_transition_time)
        
        # Total ship wait
        self.stats.log_wait_time('ship', self.env.now - arrival_time)

    def unload_container(self, container):
        """Unload container using crane, then go to yard."""
        crane_req_start = self.env.now
        with self.cranes.request() as crane_req:
            yield crane_req
            crane_wait = self.env.now - crane_req_start
            self.stats.log_wait_time('crane', crane_wait)

            unload_time = random.normalvariate(self.unload_time_mean, self.unload_time_std)
            yield self.env.timeout(max(1, unload_time))

            self.env.process(self.process_container(container))

    def process_container(self, container):
        """Container dwell in yard, then depart by truck or train."""
        yard = self.reefer_yard if container.type == 'reefer' else self.regular_yard
        container_list = self.yard_containers[container.type][container.modal]
        
        if yard[container.modal].level < yard[container.modal].capacity:
            container.yard_entry_time = self.env.now
            container_list.append(container)
            yield yard[container.modal].put(1)

            self.stats.update_yard_state(
                self.regular_yard['truck'].level + self.regular_yard['train'].level,
                self.reefer_yard['truck'].level + self.reefer_yard['train'].level
            )

            # dwell
            yield self.env.timeout(container.dwell_time)

            # truck departure
            if container.modal == "truck":
                if container in container_list:
                    yield self.env.process(self.handle_truck_departure(container))

            self.stats.log_container(container.type, container.modal)
        else:
            self.stats.log_yard_full()

    def handle_truck_departure(self, container):
        """Gate wait + final departure for truck containers."""
        while True:
            current_hour = int((self.env.now / 60) % 24)
            if current_hour < self.gate_hours["open"] or current_hour >= self.gate_hours["close"]:
                # Wait for gate opening
                if current_hour >= self.gate_hours["close"]:
                    next_opening = ((24 - current_hour) + self.gate_hours["open"]) * 60
                else:
                    next_opening = (self.gate_hours["open"] - current_hour) * 60
                yield self.env.timeout(next_opening)
                continue
            
            start_wait = self.env.now
            with self.gate.request() as gate_req:
                yield gate_req
                gate_wait = self.env.now - start_wait

                # re-check gate hours
                current_hour = int((self.env.now / 60) % 24)
                if current_hour < self.gate_hours["open"] or current_hour >= self.gate_hours["close"]:
                    # Gate closed while waiting
                    continue

                container_list = self.yard_containers[container.type]['truck']
                if container in container_list and not hasattr(container, 'departure_processed'):
                    yield self.env.timeout(10)  # gate processing time
                    container.departure_time = self.env.now
                    container.departure_processed = True

                    container_list.remove(container)
                    if container.type == 'reefer':
                        yield self.reefer_yard['truck'].get(1)
                    else:
                        yield self.regular_yard['truck'].get(1)

                    actual_dwell_time = container.departure_time - container.yard_entry_time
                    departure_hour = int((self.env.now / 60) % 24)
                    if (self.gate_hours["open"] <= departure_hour < self.gate_hours["close"]
                            and container.is_new):
                        # Only log dwell time for newly arrived containers
                        self.stats.log_wait_time('gate', gate_wait)
                        self.stats.log_container_departure(container.type, 'truck', departure_hour)
                        self.stats.log_dwell_time(container.type, actual_dwell_time)
                    break
                else:
                    break

    def handle_train_departure(self):
        """Train departure logic with capacity checks."""
        current_hour = int((self.env.now / 60) % 24)
        reg_count = self.regular_yard['train'].level
        ref_count = self.reefer_yard['train'].level
        
        total_train_containers = reg_count + ref_count
        # If fewer than 100 containers, treat as missed train
        if total_train_containers < 100:
            self.stats.log_missed_train()
            return
        
        target = min(self.train_capacity, total_train_containers)
        if total_train_containers > 0:
            # Distribute between dry/reefer in proportion
            regular_ratio = reg_count / (reg_count + ref_count) if (reg_count + ref_count) > 0 else 0
            regular_take = min(reg_count, int(target * regular_ratio))
            reefer_take = min(ref_count, target - regular_take)

            # Remove from yard
            if regular_take > 0:
                yield self.regular_yard['train'].get(regular_take)
                # We also need to remove actual Container objects from yard_containers
                removed_count = 0
                container_list = self.yard_containers['dry']['train']
                # Remove only up to `regular_take` from container_list
                for _ in range(regular_take):
                    for c in container_list:
                        if not hasattr(c, 'departure_time'):
                            # Mark departure time
                            c.departure_time = self.env.now
                            departure_hour = int((self.env.now / 60) % 24)
                            if c.is_new:  # Only log dwell time for newly arrived
                                self.stats.log_container_departure('dry', 'train', departure_hour)
                                dwell = c.departure_time - c.yard_entry_time
                                self.stats.log_dwell_time('dry', dwell)
                            container_list.remove(c)
                            removed_count += 1
                            break
                    if removed_count >= regular_take:
                        break

            if reefer_take > 0:
                yield self.reefer_yard['train'].get(reefer_take)
                # Remove reefer containers from yard_containers
                removed_count = 0
                container_list = self.yard_containers['reefer']['train']
                for _ in range(reefer_take):
                    for c in container_list:
                        if not hasattr(c, 'departure_time'):
                            # Mark departure time
                            c.departure_time = self.env.now
                            departure_hour = int((self.env.now / 60) % 24)
                            if c.is_new:  # Only log dwell time for newly arrived
                                self.stats.log_container_departure('reefer', 'train', departure_hour)
                                dwell = c.departure_time - c.yard_entry_time
                                self.stats.log_dwell_time('reefer', dwell)
                            container_list.remove(c)
                            removed_count += 1
                            break
                    if removed_count >= reefer_take:
                        break

            # Train loading/wait time
            train_wait = 30
            self.stats.log_wait_time('train', train_wait)
            yield self.env.timeout(train_wait)

            self.stats.update_yard_state(
                self.regular_yard['truck'].level + self.regular_yard['train'].level,
                self.reefer_yard['truck'].level + self.reefer_yard['train'].level
            )

            # Log train departure
            self.stats.log_train_departure(dry_loaded=regular_take,
                                           reefer_loaded=reefer_take,
                                           capacity=self.train_capacity)
        else:
            self.stats.log_missed_train()

    def run(self, minutes):
        """Run simulation for 'minutes'."""
        # Start generating ships
        self.env.process(self.generate_ship_arrivals())
        # Run simulation
        self.env.run(until=minutes)


# -----------------------------------------------------------
# Plotting & Streamlit
# -----------------------------------------------------------
def plot_statistics(stats):
    """
    Final reorganized layout:
      1. Dwell Times
      2. Equipment Utilization (Overall)
      3. Deep Dives (Berths, Cranes, Yard, Gate, Train)
      4. Container Flows Over Time
      5. Additional Stats
    """

    # -----------------------
    # 1. Dwell Times
    # -----------------------
    st.markdown("## 1. Dwell Times")
    dry_dwell_days = [x / (24 * 60) for x in stats.dwell_times['dry']]
    reefer_dwell_days = [x / (24 * 60) for x in stats.dwell_times['reefer']]

    c1, c2 = st.columns(2)
    with c1:
        fig_dwell_hist = go.Figure()
        if dry_dwell_days:
            fig_dwell_hist.add_trace(go.Histogram(x=dry_dwell_days, name='Dry', opacity=0.5))
        if reefer_dwell_days:
            fig_dwell_hist.add_trace(go.Histogram(x=reefer_dwell_days, name='Reefer', opacity=0.5))
        fig_dwell_hist.update_layout(
            barmode='overlay',
            title='Dwell Time Distribution (Days)',
            xaxis_title='Dwell Time (days)',
            yaxis_title='Count'
        )
        fig_dwell_hist.update_traces(opacity=0.6)
        st.plotly_chart(fig_dwell_hist, use_container_width=True)

    with c2:
        fig_dwell_box = go.Figure()
        if dry_dwell_days:
            fig_dwell_box.add_trace(go.Box(y=dry_dwell_days, name='Dry'))
        if reefer_dwell_days:
            fig_dwell_box.add_trace(go.Box(y=reefer_dwell_days, name='Reefer'))
        fig_dwell_box.update_layout(
            title='Dwell Time Box Plot (Days)',
            yaxis_title='Days'
        )
        st.plotly_chart(fig_dwell_box, use_container_width=True)

    # -----------------------
    # 2. Equipment Utilization (Overall)
    # -----------------------
    st.markdown("## 2. Equipment Utilization (Overall)")
    usage_data = []
    all_usage_hours = set(stats.resource_usage_by_hour['berths'].keys()) | \
                      set(stats.resource_usage_by_hour['cranes'].keys()) | \
                      set(stats.resource_usage_by_hour['gate'].keys()) | \
                      set(stats.resource_usage_by_hour['yard'].keys())
    sorted_usage_hours = sorted(all_usage_hours)
    for hour in sorted_usage_hours:
        usage_data.append({
            'hour': hour,
            'berths': stats.resource_usage_by_hour['berths'][hour],
            'cranes': stats.resource_usage_by_hour['cranes'][hour],
            'yard': stats.resource_usage_by_hour['yard'][hour],
            'gate': stats.resource_usage_by_hour['gate'][hour]
        })
    df_usage = pd.DataFrame(usage_data).sort_values('hour') if usage_data else pd.DataFrame()

    if df_usage.empty:
        st.write("No resource usage data.")
    else:
        fig_usage_line = px.line(
            df_usage,
            x='hour',
            y=['berths', 'cranes', 'yard', 'gate'],
            labels={'value': 'Usage Fraction', 'hour': 'Hour'},
            title='Resource Usage Over Time (0–1)'
        )
        fig_usage_line.update_layout(legend_title_text='Resource')
        st.plotly_chart(fig_usage_line, use_container_width=True)

        # Heatmap
        resource_order = ["berths", "cranes", "yard", "gate"]
        data_for_heatmap = []
        for row in df_usage.itertuples(index=False):
            data_for_heatmap.append({"hour": row.hour, "resource": "berths", "usage": row.berths})
            data_for_heatmap.append({"hour": row.hour, "resource": "cranes", "usage": row.cranes})
            data_for_heatmap.append({"hour": row.hour, "resource": "yard",   "usage": row.yard})
            data_for_heatmap.append({"hour": row.hour, "resource": "gate",   "usage": row.gate})
        df_heatmap = pd.DataFrame(data_for_heatmap)
        if not df_heatmap.empty:
            df_heatmap["resource"] = pd.Categorical(df_heatmap["resource"], 
                                                    categories=resource_order, 
                                                    ordered=True)
            df_pivot = df_heatmap.pivot(index="resource", columns="hour", values="usage")
            fig_usage_heat = px.imshow(
                df_pivot,
                color_continuous_scale=["blue", "red"],
                zmin=0,
                zmax=1,
                labels=dict(color="Usage Fraction"),
                aspect="auto"
            )
            fig_usage_heat.update_layout(
                title="Resource Usage Heatmap (0=Blue, 1=Red)",
                xaxis_title="Hour",
                yaxis_title="Resource"
            )
            st.plotly_chart(fig_usage_heat, use_container_width=True)

    # -----------------------
    # 3. Deep Dives (Berths, Cranes, Yard, Gate, Train)
    # -----------------------
    st.markdown("## 3. Deep Dives")

    # A) Berths
    st.markdown("### Berths")
    berth_wait_times = stats.wait_times['berth']
    if berth_wait_times:
        fig_berth_wait = go.Figure()
        fig_berth_wait.add_trace(go.Histogram(x=berth_wait_times, name='Berth Wait (min)'))
        fig_berth_wait.update_layout(
            title="Distribution of Berth Wait Times (min)",
            xaxis_title='Wait Time (min)',
            yaxis_title='Count'
        )
        st.plotly_chart(fig_berth_wait, use_container_width=True)
    else:
        st.write("No berth wait time data recorded.")

    # B) Cranes
    st.markdown("### Cranes")
    crane_wait_times = stats.wait_times['crane']
    if crane_wait_times:
        fig_crane_wait = go.Figure()
        fig_crane_wait.add_trace(go.Histogram(x=crane_wait_times, name='Crane Wait (min)'))
        fig_crane_wait.update_layout(
            title="Distribution of Crane Wait Times (min)",
            xaxis_title='Wait Time (min)',
            yaxis_title='Count'
        )
        st.plotly_chart(fig_crane_wait, use_container_width=True)
    else:
        st.write("No crane wait time data recorded.")

    # C) Yard
    st.markdown("### Yard")
    st.write(f"**Yard Full Events**: {stats.yard_full_events}")
    if not df_usage.empty:
        fig_yard_hist = go.Figure()
        fig_yard_hist.add_trace(go.Histogram(x=df_usage["yard"], name='Yard Usage Fraction'))
        fig_yard_hist.update_layout(
            title="Distribution of Yard Usage Fraction",
            xaxis_title='Usage Fraction',
            yaxis_title='Count'
        )
        st.plotly_chart(fig_yard_hist, use_container_width=True)

    # D) Gate
    st.markdown("### Gate")
    gate_wait_times = stats.wait_times['gate']
    if gate_wait_times:
        fig_gate_wait = go.Figure()
        fig_gate_wait.add_trace(go.Histogram(x=gate_wait_times, name='Gate Wait (min)'))
        fig_gate_wait.update_layout(
            title="Distribution of Gate Wait Times (min)",
            xaxis_title='Wait Time (min)',
            yaxis_title='Count'
        )
        st.plotly_chart(fig_gate_wait, use_container_width=True)
    else:
        st.write("No gate wait time data recorded.")

    # E) Train
    st.markdown("### Train Departures")
    # Show a table of each train departure: time, dry loaded, reefer loaded, total loaded, capacity
    if stats.train_departure_records:
        df_trains = pd.DataFrame(stats.train_departure_records)
        st.dataframe(df_trains)
    else:
        st.write("No train departure data recorded.")

    # -----------------------
    # 4. Container Flows Over Time
    # -----------------------
    st.markdown("## 4. Container Flows Over Time")
    arrivals_list = []
    arrivals_hours = sorted(stats.hourly_stats['arrivals'].keys())
    for hour in arrivals_hours:
        arrivals_list.append({
            'hour': hour,
            'dry': stats.hourly_stats['arrivals'][hour]['dry'],
            'reefer': stats.hourly_stats['arrivals'][hour]['reefer']
        })
    df_arrivals = pd.DataFrame(arrivals_list)

    departures_list = []
    departure_hours = set(stats.hourly_stats['departures']['truck'].keys()) | \
                      set(stats.hourly_stats['departures']['train'].keys())
    sorted_dep_hours = sorted(departure_hours)
    for hour in sorted_dep_hours:
        truck_dry = stats.hourly_stats['departures']['truck'][hour]['dry']
        truck_reefer = stats.hourly_stats['departures']['truck'][hour]['reefer']
        train_dry = stats.hourly_stats['departures']['train'][hour]['dry']
        train_reefer = stats.hourly_stats['departures']['train'][hour]['reefer']
        departures_list.append({
            'hour': hour,
            'Truck-Dry': truck_dry,
            'Truck-Reefer': truck_reefer,
            'Train-Dry': train_dry,
            'Train-Reefer': train_reefer
        })
    df_departures = pd.DataFrame(departures_list).sort_values('hour') if departures_list else pd.DataFrame()

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Arrivals")
        if df_arrivals.empty:
            st.write("No container arrivals data.")
        else:
            fig_arrivals = px.line(
                df_arrivals,
                x='hour',
                y=['dry', 'reefer'],
                labels={'value': 'Containers', 'hour': 'Hour'},
                title='Arrivals Over Time'
            )
            fig_arrivals.update_layout(legend_title_text='Container Type')
            st.plotly_chart(fig_arrivals, use_container_width=True)

    with c4:
        st.markdown("#### Departures")
        if df_departures.empty:
            st.write("No container departures data.")
        else:
            fig_departures = px.line(
                df_departures,
                x='hour',
                y=['Truck-Dry', 'Truck-Reefer', 'Train-Dry', 'Train-Reefer'],
                labels={'value': 'Containers', 'hour': 'Hour'},
                title='Departures Over Time'
            )
            fig_departures.update_layout(legend_title_text='Departure Type')
            st.plotly_chart(fig_departures, use_container_width=True)
    
# -----------------------------------------------------------
# Streamlit UI (main entry)
# -----------------------------------------------------------
def main():
    st.set_page_config(page_title="SCMb Capstone 2025", layout="wide")
    st.title("Scenario 1 - Port of New York/New Jersey 🗽")
    
    st.sidebar.header("Settings ⚙️")

    max_berths = st.sidebar.number_input("Max Berths", min_value=1, value=MAX_BERTHS_DEFAULT, step=1)
    cranes_per_berth = st.sidebar.number_input("Cranes per Berth", min_value=1, value=CRANES_PER_BERTH_DEFAULT, step=1)
    unload_mean = st.sidebar.number_input("Container Unload Time Mean (min)",
                                          value=CONTAINER_UNLOAD_TIME_DEFAULT["mean"], step=0.1)
    unload_std = st.sidebar.number_input("Container Unload Time Std (min)",
                                         value=CONTAINER_UNLOAD_TIME_DEFAULT["std"], step=0.1)
    berth_transition = st.sidebar.number_input("Berth Transition Time (min)",
                                               value=BERTH_TRANSITION_TIME_DEFAULT, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Container Type Probabilities")
    dry_prob = st.sidebar.slider("Dry Probability", 0.0, 1.0,
                                 CONTAINER_TYPES_DEFAULT["dry"]["probability"], 0.01)
    reefer_prob = 1 - dry_prob

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Modal Split")
    truck_prob = st.sidebar.slider("Truck Probability", 0.0, 1.0,
                                   MODAL_SPLIT_DEFAULT["truck"], 0.01)
    train_prob = 1.0 - truck_prob

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Gate Hours")
    gate_open = st.sidebar.number_input("Gate Open (hour)", min_value=0, max_value=23,
                                        value=GATE_HOURS_DEFAULT["open"], step=1)
    gate_close = st.sidebar.number_input("Gate Close (hour)", min_value=0, max_value=23,
                                         value=GATE_HOURS_DEFAULT["close"], step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Train Schedule")
    trains_per_day = st.sidebar.number_input("Trains per Day", min_value=1,
                                             value=TRAIN_SCHEDULE_DEFAULT["trains_per_day"], step=1)
    train_capacity = st.sidebar.number_input("Train Capacity", min_value=1,
                                             value=TRAIN_SCHEDULE_DEFAULT["capacity"], step=1)

    st.sidebar.markdown("---")
    # NEW: Starting yard utilization
    starting_yard_util = st.sidebar.slider("Starting Yard Utilization (%)", 0, 100, 50, 1)

    simulation_days = st.sidebar.number_input("Simulation Duration (days)", min_value=1, value=30, step=1)

    # Optional: let user pick a "start date/time" for the simulation
    # so that train departures are shown as actual datetimes
    use_start_date = st.sidebar.checkbox("Use Start Date", value=False)
    if use_start_date:
        sim_start_date = st.sidebar.date_input("Simulation Start Date", datetime(2025, 1, 1).date())
        sim_start_time = st.sidebar.time_input("Simulation Start Time", datetime(2025, 1, 1, 0, 0).time())
        combined_datetime = datetime.combine(sim_start_date, sim_start_time)
        simulation_start = combined_datetime
    else:
        simulation_start = None

    container_types = {
        "dry": {
            "probability": dry_prob,
            "dwell_time": CONTAINER_TYPES_DEFAULT["dry"]["dwell_time"]
        },
        "reefer": {
            "probability": reefer_prob,
            "dwell_time": CONTAINER_TYPES_DEFAULT["reefer"]["dwell_time"]
        }
    }

    modal_split = {"truck": truck_prob, "train": train_prob}

    gate_hours = {
        "open": gate_open,
        "close": gate_close,
        "trucks_per_hour": GATE_HOURS_DEFAULT["trucks_per_hour"]
    }

    if st.button("Run Simulation"):
        sim = PortSimulation(
            max_berths=max_berths,
            cranes_per_berth=cranes_per_berth,
            unload_time_mean=unload_mean,
            unload_time_std=unload_std,
            berth_transition_time=berth_transition,
            container_types=container_types,
            modal_split=modal_split,
            gate_hours=gate_hours,
            ships_data=DEFAULT_SHIPS_DATA,
            trains_per_day=trains_per_day,
            train_capacity=train_capacity,
            starting_yard_util_percent=starting_yard_util,
            simulation_start=simulation_start
        )
        sim.run(simulation_days * 24 * 60)

        st.subheader("Analysis & Plots")
        plot_statistics(sim.stats)


if __name__ == "__main__":
    main()