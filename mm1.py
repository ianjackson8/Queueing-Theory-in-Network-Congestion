'''
M/M/1 Queue Simulation Implementation

Author: Ian Jackson
Date: 3/17/2025
'''

#== Imports ==#
import os
import math
import heapq
import argparse
import simpy

import matplotlib.pyplot as plt
import numpy as np

from enum import Enum
from typing import Union
from numpy import random as rnd

rnd.seed(42)  

#== Congestion Control Classes ==#
class CongestionControl:
    '''
    base class for congestion control algorithms.
    this class can be extended to implement specific congestion control strategies.
    '''
    def __init__(self):
        self.cwnd = 1
        self.ssthresh = 10
        self.acked = 0

        self.cwnd_log = [self.cwnd]
        self.time_log = [0]
        self.ssthresh_log = [self.ssthresh]
        self.state_log = []

    def on_ack(self, time: float):
        '''
        called when an ACK is received
        this method should be overridden in subclasses to implement specific behavior
        '''
        raise NotImplementedError("Subclasses should implement this method.")
    
    def on_loss(self, time: float):
        '''
        called when a packet loss is detected
        this method should be overridden in subclasses to implement specific behavior
        '''
        raise NotImplementedError("Subclasses should implement this method.")
    
    def get_cwnd(self) -> int:
        '''
        returns the current congestion window size (cwnd)
        
        Returns:
            int: the current congestion window size
        '''
        return self.cwnd
    
    def log_cwnd(self, cur_time: float, state: str) -> None:
        '''
        logs the current congestion window size and time for analysis

        Args:
            cur_time (float): The current time in the simulation
            state (str): The state of the congestion control (e.g., "ACK", "LOSS")
        '''
        self.cwnd_log.append(self.cwnd)
        self.time_log.append(cur_time)
        self.state_log.append(state)

class NoCongestionControl(CongestionControl):
    def __init__(self):
        '''
        Initialize No Congestion Control parameters.
        This class represents a scenario where no congestion control is applied.
        '''
        super().__init__()
        self.state = "No Control"

    def on_ack(self, time: float):
        pass

    def on_loss(self, time: float):
        pass

class TCPReno(CongestionControl):
    def __init__(self):
        '''
        Initialize TCP Reno congestion control parameters.
        '''
        super().__init__()
        self.state = "Slow Start"  # initial state
        self.state_log.append(self.state)

    def on_ack(self, time: float):
        '''
        handle an AKC event

        Args:
            time (float): the time at which the ACK was received
        '''
        if self.state == "Slow Start":
            self.cwnd += 1
            if self.cwnd >= self.ssthresh:
                self.state = "Congestion Avoidance"
        elif self.state == "Congestion Avoidance":
            self.cwnd += 1 / self.cwnd

        self.cwnd_log.append(self.cwnd)
        self.time_log.append(time)
        self.state_log.append(self.state)
        self.ssthresh_log.append(self.ssthresh)
        print(f"[{time:.4f}] [TCP Reno], cwnd={self.cwnd:.2f}, ssthresh={self.ssthresh:.2f}, state={self.state}")

    def on_loss(self, time: float):
        '''
        handle a packet loss event

        Args:
            time (float): the time at which the loss was detected
        '''
        self.ssthresh = max(self.cwnd // 2, 1)  
        self.cwnd = 1

        self.cwnd_log.append(self.cwnd)
        self.time_log.append(time)
        self.state_log.append("Loss")
        self.ssthresh_log.append(self.ssthresh)

        self.state = "Slow Start"

        print(f"[{time:.4f}] [TCP Reno LOSS], cwnd reset to {self.cwnd}, ssthresh={self.ssthresh}")

    def on_dup_ack(self, time: float):
        '''
        Handle 3 duplicate ACKs (Fast Retransmit + Fast Recovery).
        '''
        self.ssthresh = max(self.cwnd // 2, 1)
        self.cwnd = self.ssthresh + 3
        self.state = "Fast Recovery"

        self.cwnd_log.append(self.cwnd)
        self.time_log.append(time)
        self.state_log.append(self.state)
        self.ssthresh_log.append(self.ssthresh)

        print(f"[{time:.4f}] [TCP Reno DUPACK], Fast Recovery, cwnd={self.cwnd}, ssthresh={self.ssthresh}")


#==  Classes ==#
class Packet:
    '''
    represent a packet in queue
    '''
    def __init__(self, id: int, arrival_time: float, service_time: float, deadline: float):
        '''
        initialize a Packet instance.

        Args:
            id (int): unique ID for the packet
            arrival_time (float): arrival time of the packet
            service_time (float): time required to service the packet
            deadline (float): deadline by which the packet must be processed
        '''
        self.id = id
        self.arrival_time = arrival_time
        self.service_time = service_time
        self.deadline = deadline

        self.start_service_time = None
        self.finish_time = None

class NetworkRouter:
    def __init__(self, env: simpy.Environment, mu: float, theta: float, queue_size: int, 
                 cc: CongestionControl = None, sender = None):
        '''
        Initialize a NetworkRouter instance.

        Args:
            env (simpy.Environment): simulation environment
            mu (float): service rate of the router
            theta (float): arrival rate of packets
            queue_size (int): maximum size of the queue
            cc (CongestionControl, optional): congestion control mechanism
        '''
        # simpy environment
        self.env = env
        self.server = simpy.Resource(env, capacity=1)

        # parameters
        self.mu = mu
        self.theta = theta
        self.queue_size = queue_size
        self.queue = []

        # congestion control
        self.cc = cc if cc is not None else NoCongestionControl()
        self.sender = sender

        # statistics
        self.dropped = 0
        self.serviced = 0
        self.total_delay = 0

        # logs 
        self.time_log = [0]
        self.serviced_log = [0]
        self.dropped_log = [0]
        self.queue_log = [0]

    def process_packet(self, packet: Packet):
        '''
        process a packet in the network router.

        Args:
            packet (Packet): the packet to be processed
        '''
        now = self.env.now

        # drop if deadline has passed
        if now - packet.arrival_time > packet.deadline:
            self.dropped += 1
            print(f"[{now:.4f}] Packet {packet.id} DROPPED (expired)")
            self.log_state()

            # congestion control
            if self.cc: self.cc.on_loss(now)

            # notify sender of loss
            if self.sender: self.sender.notify_loss()

            # remove packet from queue if it exists
            if packet in self.queue:
                self.queue.remove(packet)

            return
        
        # try to rquest the server
        with self.server.request() as req:
            result = yield req | self.env.timeout(packet.deadline)

            if req not in result:
                # timeout occurred, drop the packet
                self.dropped += 1
                print(f"[{now:.4f}] Packet {packet.id} DROPPED (timeout)")
                self.log_state()

                # congestion control
                if self.cc: self.cc.on_loss(now)

                # notify sender of loss
                if self.sender: self.sender.notify_loss()

                # remove packet from queue if it exists
                if packet in self.queue:
                    self.queue.remove(packet)

                return
            
            packet.start_service_time = self.env.now
            yield self.env.timeout(packet.service_time)
            packet.finish_time = self.env.now

            delay = packet.finish_time - packet.arrival_time
            self.total_delay += delay
            self.serviced += 1
            print(f"[{self.env.now:.4f}] Packet {packet.id} SERVICED, delay={delay:.4f}")
            self.log_state()

            # congestion control
            if self.cc: self.cc.on_ack(self.env.now)

            # notify sender of successful service
            if self.sender: self.sender.notify_ack(pkt_id=packet.id)

            # remove packet from queue
            if packet in self.queue:
                self.queue.remove(packet)

    def receive(self, packet: Packet):
        '''
        receive a packet and process it.

        Args:
            packet (Packet): the packet to be received
        '''
        if self.queue_size and len(self.queue) >= self.queue_size:
            # queue is full, drop the packet
            self.dropped += 1
            print(f"[{self.env.now:.4f}] Packet {packet.id} DROPPED (queue full)")
            self.log_state()

            # congestion control
            if self.cc: self.cc.on_loss(self.env.now)

            # notify sender of loss
            if self.sender: self.sender.notify_loss()

        else:
            self.queue.append(packet)
            self.env.process(self.process_packet(packet))

    def log_state(self):
        '''
        Log the current state of the router.
        '''
        self.time_log.append(self.env.now)
        self.serviced_log.append(self.serviced)
        self.dropped_log.append(self.dropped)
        self.queue_log.append(len(self.queue))

class Sender:
    def __init__(self, env: simpy.Environment, router: NetworkRouter, lam: float, mu: float, theta: float,
                 num_packets: int, cc: CongestionControl = None):
        '''
        initialize a Sender instance.

        Args:
            env (simpy.Environment): simulation environment
            router (NetworkRouter): the network router to send packets to
            lam (float): arrival rate of packets
            mu (float): service rate of the router
            theta (float): deadline rate for packets
            num_packets (int): number of packets to generate
            cc (CongestionControl, optional): congestion control mechanism
        '''
        self.env = env
        self.router = router
        self.lam = lam
        self.mu = mu
        self.theta = theta
        self.num_packets = num_packets
        self.cc = cc

        self.in_flight = 0
        self.sent_packets = 0

        self.last_acked_id = -1
        self.dup_ack_count = 0

    def start(self):
        self.env.process(self.send_packets())

    def send_packets(self):
        while self.sent_packets < self.num_packets:
            # if self.in_flight < self.cc.cwnd:
            #     service_time = rnd.exponential(1 / mu)
            #     pkt = Packet(self.sent_packets, self.env.now, service_time, self.theta)
            #     self.in_flight += 1
            #     self.router.receive(pkt)
            #     self.sent_packets += 1
            # yield self.env.timeout(rnd.exponential(1 / lam))

            window_size = int(self.cc.cwnd) - self.in_flight
            for _ in range(window_size):
                if self.sent_packets >= self.num_packets:
                    break
                service_time = rnd.exponential(1 / mu)
                pkt = Packet(self.sent_packets, self.env.now, service_time, self.theta)
                self.in_flight += 1
                self.router.receive(pkt)
                self.sent_packets += 1
            yield self.env.timeout(0.5)

    def notify_ack(self, pkt_id = None):
        self.in_flight = max(0, self.in_flight - 1)

        # Detect duplicate ACKs
        if pkt_id == self.last_acked_id:
            self.dup_ack_count += 1
            if self.dup_ack_count == 3:
                if hasattr(self.cc, 'on_dup_ack'):
                    self.cc.on_dup_ack(self.env.now)
        else:
            self.last_acked_id = pkt_id
            self.dup_ack_count = 1  # reset

    def notify_loss(self):
        self.in_flight = max(0, self.in_flight - 1)

#== Methods ==#
def packet_generator(env: simpy.Environment, router: NetworkRouter, lam: float, mu: float, theta: float, num_packets: int):
    '''
    Generate packets and send them to the router.

    Args:
        env (simpy.Environment): simulation environment
        router (NetworkRouter): the network router to send packets to
        lam (float): arrival rate of packets
        mu (float): service rate of the router
        theta (float): deadline rate for packets
        num_packets (int): number of packets to generate
    '''
    for i in range(num_packets):
        arrival_time = env.now
        service_time = rnd.exponential(1 / mu)
        # deadline = rnd.exponential(1 / theta)
        deadline = theta if theta > 0 else float('inf')

        packet = Packet(id=i, arrival_time=arrival_time, service_time=service_time, deadline=deadline)
        print(f"[{env.now:.4f}] Packet {i} ARRIVAL")
        
        router.receive(packet)
        yield env.timeout(rnd.exponential(1 / lam))

def run_simulation(lam: float, mu: float, theta: float, queue_size: int, num_packets: int, cc: CongestionControl = None):
    '''
    Run the M/M/1 queue simulation.

    Args:
        lam (float): arrival rate of packets
        mu (float): service rate of the router
        theta (float): deadline rate for packets
        queue_size (int): maximum size of the queue
        num_packets (int): number of packets to generate
        cc (CongestionControl, optional): congestion control mechanism
    '''
    print("== Simulation Start ==")
    env = simpy.Environment()

    if cc:
        sender = Sender(env, None, lam, mu, theta, num_packets, cc)
        router = NetworkRouter(env, mu=mu, theta=theta, queue_size=queue_size, cc=cc, sender=sender)
        sender.router = router
        sender.start()
    else:
        router = NetworkRouter(env, mu=mu, theta=theta, queue_size=queue_size, cc=cc)
        env.process(packet_generator(env, router, lam, mu, theta, num_packets))

    env.run()

    # report statistics
    total = num_packets
    dropped = router.dropped
    serviced = router.serviced
    avg_delay = router.total_delay / serviced if serviced > 0 else 0
    throughput = serviced / env.now if env.now > 0 else 0

    print("\n== Simulation Results ==")
    print(f"Simulation Time: {env.now:.2f} seconds")
    print(f"Total Packets: {total}")
    print(f"Serviced: {serviced}")
    print(f"Dropped: {dropped}")
    print(f"Loss Rate: {100 * dropped / total:.2f}%")
    print(f"Average Delay: {avg_delay:.4f} seconds")
    print(f"Throughput: {throughput:.4f} packets/sec")
    print("=========================")

    plot_results(router)
    if cc: plot_cwnd(cc)

def plot_results(router: NetworkRouter):
    '''
    Plot the results of the simulation.

    Args:
        router (NetworkRouter): the network router containing the logs
    '''

    # create a results folder if does not exist
    results_folder = "results"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # cumulative Serviced and Dropped Packets
    plt.figure(figsize=(10, 5))
    plt.plot(router.time_log, router.serviced_log, label="Serviced Packets", linewidth=2, color='green')
    plt.plot(router.time_log, router.dropped_log, label="Dropped Packets", linewidth=2, color='red')
    plt.xlabel("Time")
    plt.ylabel("Cumulative Packets")
    plt.title("Cumulative Packet Events Over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "cumulative_packet_events_over_time.png"))

    # queue Length Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(router.time_log, router.queue_log, label="Queue Length", color='purple')
    plt.xlabel("Time")
    plt.ylabel("Queue Length")
    plt.title("Queue Length Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, "queue_length_over_time.png"))

def plot_cwnd(cc):
    if not hasattr(cc, 'cwnd_log') or not hasattr(cc, 'state_log'):
        print("ERR: Missing cwnd/state logs in congestion control object.")
        return
    
    # get name of congestion control method
    if isinstance(cc, NoCongestionControl):
        cc_meth = "no_cc"
    elif isinstance(cc, TCPReno):
        cc_meth = "reno"
    else:
        cc_meth = "unk"

    # extract logs
    time = cc.time_log
    cwnd = cc.cwnd_log
    ssthresh = cc.ssthresh_log
    states = cc.state_log

    # init plot
    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # plot cwnd and ssthresh
    ax.plot(time, cwnd, label="cwnd", color='blue')
    # ax.plot(time, ssthresh, label="ssthresh", color='orange', linestyle='--')

    # highlight regions by TCP state
    color_map = {
        "Slow Start": "white",
        "Congestion Avoidance": "white",
        "Fast Recovery": "white",
        "Loss": "white"
    }

    start_time = time[0]
    current_state = states[0]

    for i in range(1, len(time)):
        if states[i] != current_state:
            end_time = time[i]
            # ax.axvspan(start_time, end_time, facecolor=color_map.get(current_state, "white"), alpha=0.3, label=current_state if start_time == time[0] else "")
            ax.axvspan(start_time, end_time, facecolor=color_map.get(current_state, "white"), alpha=0.3)
            start_time = end_time
            current_state = states[i]

    for i in range(len(states)):
        if states[i] == "Loss":
            ax.axvline(time[i], color='black', linestyle=':', alpha=1, label='Loss' if i == 0 else "")

    # Capture final region
    ax.axvspan(start_time, time[-1], facecolor=color_map.get(current_state, "white"), alpha=0.3, label=current_state if states.count(current_state) == 1 else "")

    # Final plot setup
    ax.set_title(f"TCP {cc_meth} cwnd Over Time with State Regions")
    ax.set_xlabel("Time")
    ax.set_ylabel("Congestion Window (cwnd)")
    ax.legend()
    # plt.legend()
    ax.grid(True)
    plt.tight_layout()

    plt.savefig(os.path.join("results", f"{cc_meth}_cwnd_over_time.png"))

#== Main Execution ==#
if __name__ == "__main__":
    # define params
    lam = 10
    mu = 10
    theta = 1
    queue_size = 4
    num_packets = 100
    tcp_cc = TCPReno()

    print("== Simulation Parameters ==")
    print(f"  - Arrival Rate (λ): {lam}")
    print(f"  - Service Rate (μ): {mu}")
    print(f"  - Deadline Rate (θ): {theta}")
    print(f"  - Queue Size: {'Infinite' if queue_size is None else queue_size}")
    print(f"  - Total Packets: {num_packets}")
    print(f"  - Congestion Control: {tcp_cc if tcp_cc else 'None'}")
    print("===========================\n")

    run_simulation(lam, mu, theta, queue_size, num_packets, cc=tcp_cc)