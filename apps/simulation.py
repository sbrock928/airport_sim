import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from app import app


class Airport_Simulation:
    def __init__(self, arrival_dist_scale, taxi_delay_parameters,gate_delay_parameters):
        self.clock=0.0                      #simulation clock
        self.num_arrivals=0                 #total number of arrivals
        self.num_in_system=0
        self.arrival_dist_scale = arrival_dist_scale
        self.taxi_delay_parameters = taxi_delay_parameters
        self.gate_delay_parameters = gate_delay_parameters
        self.t_arrival=self.gen_int_arr()   #time of next arrival
        self.t_departure=float('inf')      #departure time from runway
        self.gate_sum=0                     #Sum of gate_departure times
        self.taxi_sum=0                     #Sum of gate_departure times
        self.state_T=0                     #current state of runway (binary)
        self.total_wait_time=0.0            #total wait time
        self.num_in_q=0                     #current number in queue
        self.number_in_queue=0              #planes in line(counter)
        self.num_of_departures=0           #number of planes departed


    def time_adv(self):
        t_next_event = min(self.t_arrival, self.t_departure)
        self.total_wait_time += (self.num_in_q * (t_next_event - self.clock))


        self.clock = t_next_event

        if self.t_arrival < self.t_departure:
            self.arrival()
        else:
            self.departure()

    def arrival(self):
        self.num_arrivals += 1
        self.num_in_system += 1

        if self.state_T == 1:
            self.num_in_q += 1
            self.number_in_queue += 1
            self.t_arrival = self.clock + self.gen_int_arr()
        else:
            self.state_T = 1
            self.gate = self.gen_gate_delay_time()
            self.taxi = self.gen_taxi_delay_time()
            self.gate_sum+= self.gate
            self.taxi_sum+= self.taxi
            self.t_departure = self.clock + self.gate + self.taxi
            self.t_arrival = self.clock + self.gen_int_arr()


    def departure(self):
        self.num_of_departures += 1
        if self.num_in_q > 0:
            self.gate = self.gen_gate_delay_time()
            self.taxi = self.gen_taxi_delay_time()
            self.gate_sum+= self.gate
            self.taxi_sum+= self.taxi
            self.t_departure = self.clock + self.gate + self.taxi
            self.num_in_q -= 1
        else:
            self.t_departure = float('inf')
            self.state_T = 0


    def gen_int_arr(self):  # function to generate arrival times using exponential distribution
        return .1*(np.random.exponential(self.arrival_dist_scale['scale'], size = 1)[0])

    def gen_gate_delay_time(self):  # function to generate service time for teller 1 using inverse trnasform
        return (-np.log(1 - (np.random.uniform(low=0.0, high=1.0))) * (np.random.exponential(self.gate_delay_parameters['scale'], size = 1)[0]))
    def gen_taxi_delay_time(self):  # function to generate service time for teller 1 using inverse trnasform
        return (np.random.lognormal(mean= self.taxi_delay_parameters['mu'], sigma=self.taxi_delay_parameters['sigma'], size = 1)[0])





