'''
M/M/c Queue Simulation Implementation
OLD!

Author: Ian Jackson
Date: 4/16/2025
'''

#== Imports ==#
import os
import math
import heapq
import argparse

import matplotlib.pyplot as plt

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

        self.cwnd_log = []
        self.time_log = []
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

class TCPReno(CongestionControl):
    def on_ack(self, time: float):
        '''
        handle ACKs in TCP Reno

        Args:
            time (float): the time at which the ACK was received
        '''
        # if cwnd is less than ssthresh, increase cwnd linearly
        if self.cwnd < self.ssthresh:
            self.cwnd *= 2
            phase = "Slow Start"

        # after reaching ssthresh, increase cwnd by 1/cwnd for each ACK
        else:
            self.acked += 1
            phase = "Additive Increase"
            if self.acked >= self.cwnd:
                self.cwnd += 1
                self.acked = 0

        # log the current cwnd and time
        self.log_cwnd(time, phase)
        print(f"[{time:.4f}] [TCP Reno], cwnd={self.cwnd}, ssthresh={self.ssthresh}, state={phase}")

    def on_loss(self, time: float):
        '''
        handle packet loss in TCP Reno

        Args:
            time (float): the time at which the packet loss was detected
        '''
        # set ssthresh to half of cwnd, but at least 1
        self.ssthresh = max(self.cwnd // 2, 1)  
        self.cwnd = 1
        self.acked = 0

        # log the current cwnd and time
        self.log_cwnd(time, "Multiplicative Decrease")
        print(f"[{time:.4f}] [TCP Reno], cwnd={self.cwnd}, ssthresh={self.ssthresh}, state=Multiplicative Decrease")

class TCPCubic(CongestionControl):
    def __init__(self):
        super().__init__()
        self.beta = 0.7             # multiplicative decrease factor
        self.c = 0.4                # CUBIC scaling constant
        self.w_max = self.cwnd      # last maximum cwnd before loss
        self.epoch_start = None     # time when the current epoch started

    def on_ack(self, time: float):
        '''
        handle ACKs in TCP CUBIC

        Args:
            time (float): the time at which the ACK was received
        '''
        if self.epoch_start is None:
            self.epoch_start = time
            self.w_max = self.cwnd

        t = time - self.epoch_start
        K = ((self.w_max * (1 - self.beta)) / self.c) ** (1/3)
        cubic_cwnd = self.c * ((t - K) ** 3) + self.w_max

        # Ensure cwnd increases smoothly
        self.cwnd = max(1, round(cubic_cwnd))

        self.log_cwnd(time, "Cubic Increase")
        print(f"[{time:.4f}] [TCP CUBIC], cwnd={self.cwnd}, w_max={self.w_max}, state=Cubic Increase")

    def on_loss(self, time: float):
        '''
        handle packet loss in TCP CUBIC

        Args:
            time (float): the time at which the packet loss was detected
        '''
        self.w_max = self.cwnd
        self.cwnd = max(1, round(self.cwnd * self.beta))
        self.epoch_start = None  # reset epoch on loss

        self.log_cwnd(time, "Cubic Loss")
        print(f"[{time:.4f}] [TCP CUBIC], cwnd={self.cwnd}, w_max={self.w_max}, state=Cubic Loss")

#== MM1 Queue Classes ==#
class PacketStatus(Enum):
    ARRIVAL = 0
    DROP = 1
    SERVICED = 2
    BLOCK = 3

class Packet():
    '''
    Class to represent a packet
    '''
    def __init__(self, id: int, arrival_time: float, mu: float, theta: float, is_exp_drop: bool):
        '''
        initialize instance of a packet

        Args:
            id (int): packet id
            arrival_time (float): time of arrival
            mu (float): service rate
            theta (float): deadline rate or deadline time (dept on is_exp_drop)
            is_exp_drop (bool): does deadline follow exp dist? otherwise, fixed
        '''
        self.id = id
        self.arrival_time = arrival_time
        self.mu = mu
        self.theta = theta
        self.is_exp_drop = is_exp_drop
        self.server_id = None

        # determine calculated vars
        self.service_time = self.__calc_service_time(self.mu)
        self.drop_time = self.__calc_drop_time(self.theta, self.is_exp_drop)

        self.service_end_time = self.arrival_time + self.service_time
        self.limit_time = self.arrival_time + self.drop_time

    def __calc_service_time(self, mu: float) -> float:
        '''
        calculate service time based on mu, 
        T_s ~ Exp(mu)

        Args:
            mu (float): rate parameter

        Returns:
            float: how long it takes to service the job
        '''
        return rnd.exponential(1 / mu)
    
    def __calc_drop_time(self, theta: float, is_exp_drop: bool) -> float:
        '''
        calculate drop time based on param theta, could be rate param for exp dist or true value
        T_d = {
            ~ Exp(theta) if is_exp_drop,
            = theta if not is_exp_drop
        }

        Args:
            theta (float): rate param or drop time 
            is_exp_drop (bool): use theta as rate param for exp dist

        Returns:
            float: time until job expires
        '''
        if is_exp_drop:
            return rnd.exponential(theta)
        else:
            return theta

class Event():
    '''
    class to represent an event of a packet
    '''
    def __init__(self, packet_id: int, event_time: float, status: PacketStatus = PacketStatus.ARRIVAL, server_id: int | None = None):
        '''
        init instance of an event

        Args:
            packet_id (int): packet id
            event_time (float): when event occurs
            status (PacketStatus, optional): packet status. Defaults to PacketStatus.ARRIVAL.
            server_id (int | None, optional): server id if applicable. Defaults to None.
        '''
        self.packet_id = packet_id
        self.event_time = event_time
        self.status = status
        self.server_id = server_id

class Queue():
    '''
    class to represent a queue (M/M/c)
    '''
    def __init__(self, size: int | None, num_servers: int = 1):
        '''
        init instance of queue

        Args:
            size (int | None): size of queue (None is infinite)
            num_servers (int, optional): number of servers. Defaults to 1.
        '''
        self.size = size

        # init stats
        self.n_service = 0
        self.n_drop = 0
        self.n_block = 0

        # init queues
        self.priority_queue = []
        self.event_queue = []
        heapq.heapify(self.event_queue)  
        self.all_packets = []

        self.queue_len = 0

        # init servers
        self.num_servers = num_servers
        self.servers = [False] * num_servers  # false means server is free, True means busy

        # tracking/logging
        self.time_log = [0]
        self.queue_len_log = [0]
        self.service_log = [0]
        self.drop_log = [0]
        self.block_log = [0]
        self.latency_log = []

    def insert_event(self, event: Event) -> None:
        '''
        insert event in sorted order

        Args:
            event (Event): event to be added
        '''
        heapq.heappush(self.event_queue, (event.event_time, event))

    def insert_packet(self, packet: Packet) -> None:
        '''
        insert packet and create new event

        Args:
            packet (Packet): packet to be added
        '''
        self.all_packets.append(packet)

        event = self.handle_events(packet, packet.arrival_time)

        if event is not None:
            self.insert_event(event)
        
    def handle_events(self, packet: Packet, event_time: float, status: PacketStatus = PacketStatus.ARRIVAL) -> Event | None:
        '''
        Handles events for a given packet, updates the queue and statistics.

        Args:
            packet (Packet): The packet for which the event is being handled.
            event_time (float): The time at which the event occurs.
            status (PacketStatus, optional): The status of the event. Defaults to PacketStatus.ARRIVAL.

        Returns:
            Event | None: A new event if applicable, otherwise None.
        '''
        queue_len = len(self.priority_queue)
        print(f"[{event_time:.4f}] Server Status: {' '.join(['B' if s else 'I' for s in self.servers])}")

        if status == PacketStatus.ARRIVAL:
            self.queue_len = len(self.priority_queue)

            # check if the queue is full
            if self.size is not None and self.queue_len >= self.size:
                # if so, then packet blocked
                self.n_block += 1  
                print(f"[{event_time:.4f}] [{queue_len}] Packet {packet.id} BLOCKED")
                return Event(packet.id, event_time, PacketStatus.BLOCK)

            # check for an idle server
            for i in range(self.num_servers):
                if not self.servers[i]: # server found
                    # set to busy
                    self.servers[i] = True

                    # set the packet to the corrcet server
                    packet.server_id = i
                    return Event(packet.id, event_time + packet.service_time, PacketStatus.SERVICED, server_id=i)

            # no server available, enqueue
            self.priority_queue.append(packet)
            return Event(packet.id, packet.limit_time, PacketStatus.DROP)

        elif status == PacketStatus.DROP:
            # packet expires before being serviced
            self.n_drop += 1  
            print(f"[{event_time:.4f}] [S{packet.server_id}] [{queue_len}] Packet {packet.id} DROPPED")

            # remove packet from priority and event queues
            self.priority_queue = [p for p in self.priority_queue if p.id != packet.id]
            self.event_queue = [(et, e) for (et, e) in self.event_queue if e.packet_id != packet.id]

            # no further events for this packet
            return None  

        elif status == PacketStatus.SERVICED:
            # packet successfully gets serviced and exits
            self.n_service += 1 
            print(f"[{event_time:.4f}] [S{packet.server_id}] [{queue_len}] Packet {packet.id} SERVICED")
            self.latency_log.append(event_time - packet.arrival_time)

            # mark server idle
            self.servers[packet.server_id] = False

            # remove the serviced packet from the queue
            self.priority_queue = [p for p in self.priority_queue if p.id != packet.id]
            self.event_queue = [(et, e) for (et, e) in self.event_queue if e.packet_id != packet.id]

            # check if theres another packet waiting
            if len(self.priority_queue) > 0:
                # grab packet and service it, tag the server
                next_packet = self.priority_queue.pop(0)
                next_packet.server_id = packet.server_id
                self.servers[packet.server_id] = True

                self.event_queue = [(et, e) for (et, e) in self.event_queue if e.packet_id != next_packet.id]
                return Event(next_packet.id, event_time + next_packet.service_time, PacketStatus.SERVICED, server_id=packet.server_id)
                
            # if empty, nothing to do
            return None
        
        elif status == PacketStatus.BLOCK:
            # self.n_block += 1 
            return None
        
    def handle_jobs(self, cc: CongestionControl = None) -> None:
        '''
        Processes the next event in the event queue.
        This method updates the system state by handling the next scheduled event
        and scheduling any follow-up events accordingly.

        Args:
            cc (CongestionControl, optional): An instance of a congestion control algorithm, if applicable. Defaults to None.
        '''
        # retrieve the first event from queue and update queue
        _, event = heapq.heappop(self.event_queue)

        # process event, determines what happens to packet
        new_event = self.handle_events(self.all_packets[event.packet_id], event.event_time, event.status)
        
        # if congestion control is enabled, handle ACKs or losses
        if cc:
            if event.status == PacketStatus.SERVICED:
                cc.on_ack(event.event_time)
            elif event.status in [PacketStatus.DROP, PacketStatus.BLOCK]:
                cc.on_loss(event.event_time)

        self.log_stats(event.event_time, event.status)  # log stats after handling the event

        if new_event is not None:
            # insert new event into queue 
            self.insert_event(new_event)

    def log_stats(self, cur_time: float, status: PacketStatus) -> None:
        '''
        Logs the current statistics of the queue.

        Args:
            cur_time (float): The current time in the simulation.
            status (PacketStatus): The type of event just handled.
        '''
        if status in [PacketStatus.SERVICED, PacketStatus.DROP, PacketStatus.BLOCK]:
            self.time_log.append(cur_time)
            self.queue_len_log.append(len(self.priority_queue))
            self.service_log.append(self.n_service)
            self.drop_log.append(self.n_drop)
            self.block_log.append(self.n_block)

class Simulator():
    def __init__(self, lmbda: float, mu: float, theta: float, size: int | None, c: int, is_exp_drop: bool = False, tcp_cc: str = None):
        '''
        init instance of simulator

        Args:
            lmbda (float): arrival rate of packets
            mu (float): service rate
            theta (float): deadline rate or deadline time
            size (int | None): size of queue (None if infinite)
            c (int): number of servers
            is_exp_drop (bool, optional): does deadline time follow exp dist. Defaults to False.
            tpc_cc (str, optional): congestion control algorithm to use. Defaults to None.
        '''
        self.lmbda = lmbda
        self.mu = mu
        self.theta = theta
        self.size = size
        self.c = c
        self.is_exp_drop = is_exp_drop

        if tcp_cc == "reno":
            self.cc = TCPReno()
        elif tcp_cc == "cubic":
            self.cc = TCPCubic()
        else:
            self.cc = None

        # create instance of queue
        self.queue = Queue(size=self.size, num_servers=self.c)

    def run(self, n_packet: int) -> Union[float, float]:
        '''
        run simulation

        Args:
            n_packet (int): number of packets to be serviced

        Returns:
            Union[float, float]: tuple of blocked rate, dropped rate
        '''
        # init sim time and packet counter
        cur_time = 0
        packet_id = 0
        inflight = set()

        # process packets until specified num of packet is reached
        while packet_id < n_packet or self.queue.event_queue:
            cwnd = self.cc.get_cwnd() if self.cc else 1

            # Push cwnd packets and advance time
            while packet_id < n_packet and len(inflight) < cwnd:
                # generate new packet and insert into queue
                packet = Packet(packet_id, cur_time, self.mu, self.theta, self.is_exp_drop)
                print(f"[{cur_time:.4f}] [{len(self.queue.priority_queue)}] Packet {packet_id} ARRIVAL")

                self.queue.insert_packet(packet)
                inflight.add(packet_id)

                # advance sim time and increment packet id
                packet_id += 1
                cur_time += rnd.exponential(1 / self.lmbda)

            if self.queue.event_queue:
                next_event_time, next_event = heapq.heappop(self.queue.event_queue)
                packet = self.queue.all_packets[next_event.packet_id]

                # Handle the packet event
                new_event = self.queue.handle_events(packet, next_event_time, next_event.status)

                # Congestion control logic
                if self.cc:
                    if next_event.status == PacketStatus.SERVICED:
                        self.cc.on_ack(next_event_time)
                    elif next_event.status in [PacketStatus.DROP, PacketStatus.BLOCK]:
                        self.cc.on_loss(next_event_time)

                # Remove from in-flight tracking
                inflight.discard(next_event.packet_id)

                # Add follow-up event if any
                if new_event:
                    self.queue.insert_event(new_event)

                # Log queue stats
                self.queue.log_stats(next_event_time, next_event.status)

                # Advance time to next event
                cur_time = max(cur_time, next_event_time)

            # Advance time if more packets are to arrive (simulate interarrival spacing)
            if self.cc is None and packet_id < n_packet:
                cur_time += rnd.exponential(1 / self.lmbda)

        # create a simualtion stats dict
        sim_stats = {
            'sim_time': cur_time,
            'n_service': int(self.queue.service_log[-1]),
            'n_drop': int(self.queue.drop_log[-1]),
            'n_block': int(self.queue.block_log[-1]),
            'sim_pb': self.queue.n_block / n_packet,
            'sim_pd': self.queue.n_drop / n_packet,
            'sim_loss': (self.queue.n_drop + self.queue.n_block) / n_packet,
            'avg_latency': sum(self.queue.latency_log) / len(self.queue.latency_log) if self.queue.latency_log else 0
        }

        # compute and return blocking and dropping probabilities
        return sim_stats

    def plot_results(self):
        '''
        Plots the results of the simulation, including queue length over time
        and cumulative counts of serviced, dropped, and blocked packets.
        This method generates two plots:
            1. Queue Length Over Time: Shows how the length of the queue changes over time.
            2. Cumulative Packet Events Over Time: Displays the cumulative number of serviced and dropped packets over time.
        '''
        times = self.queue.time_log
        zipped = sorted(zip(self.queue.time_log, self.queue.service_log, self.queue.drop_log, self.queue.block_log))
        sorted_times, sorted_service, sorted_drop, sorted_block = zip(*zipped)

        # create a results folder if does not exist
        results_folder = "results"
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)

        # create the queue length plot
        plt.figure(figsize=(10, 4))
        plt.plot(sorted_times, self.queue.queue_len_log[:len(sorted_times)], label='Queue Length')
        plt.xlabel('Time')
        plt.ylabel('Queue Length')
        plt.title('Queue Length Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(results_folder, "queue_length_over_time.png"))

        # create the cumulative packet events plot
        plt.figure(figsize=(10, 4))
        plt.plot(sorted_times, sorted_service, label="Serviced Packets")
        # plt.plot(sorted_times, sorted_drop, label="Dropped Packets")
        plt.plot(sorted_times, sorted_block, label="Dropped Packets", color='red')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Count')
        plt.title('Cumulative Packet Events Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(results_folder, "cumulative_packet_events_over_time.png"))

        # TODO: fix coloring of the plot
        if self.cc:
            zipped = sorted(zip(self.cc.time_log, self.cc.cwnd_log, self.cc.state_log))
            times, cwnds, states = zip(*zipped)
            plt.figure(figsize=(12, 4))
            plt.plot(times, cwnds, color='black', alpha=0.6, label='TCP cwnd')

            # Fill background based on TCP state
            ax = plt.gca()
            for i in range(len(times) - 1):
                state = states[i]
                start_time = times[i]
                end_time = times[i + 1]

                ax.axvspan(start_time, end_time, alpha=0.2,
                        color={
                            'Slow Start': 'blue', 
                            'Additive Increase': 'yellow', 
                            'Multiplicative Decrease': 'red',
                            'Cubic Increase': 'yellow',
                            'Cubic Loss': 'red'
                        }.get(state, 'gray'),
                        label=state if state not in ax.get_legend_handles_labels()[1] else "")

            plt.xlabel('Time')
            plt.ylabel('Congestion Window Size')
            plt.title('TCP Reno cwnd Over Time with State Transitions')
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(results_folder, "tcp_reno_cwnd_over_time.png"))

#== Methods ==#


#== Main Execution ==#
def main():
    # set parameters
    mu = 10  # service rate (μ)
    lmbda = 20 # arrival rate (λ)
    theta = 1 # deadline rate or fixed deadline time (θ)
    n_packet = 300
    c = 2  # number of servers 

    queue_size = 4 
    is_exp_drop = False

    tcp_cc = 'reno'
    
    # init simulator
    sim = Simulator(lmbda=lmbda, mu=mu, theta=theta, size=queue_size, c=c, is_exp_drop=is_exp_drop, tcp_cc=tcp_cc)

    # run sim
    sim_stats = sim.run(n_packet=n_packet)
    sim.plot_results()

    # extract stats
    sim_time = sim_stats['sim_time']
    sim_service = sim_stats['n_service']
    sim_drop = sim_stats['n_drop']
    sim_block = sim_stats['n_block']
    sim_pb = sim_stats['sim_pb']  # blocked rate
    sim_pd = sim_stats['sim_pd']  # dropped rate
    sim_drop = sim_stats['sim_loss']  # total loss rate
    avg_latency = sim_stats['avg_latency']

    if int(sim_drop) > 0:
        print("\033[33mWARN: Drop count more than 0\033[0m")
    
    # print results
    print(f"\n=== Simulation Results for M/M/c Queue ===")
    print("Simulation Parameters:")
    print(f"  - Arrival Rate (λ): {lmbda}")
    print(f"  - Service Rate (μ): {mu}")
    print(f"  - Deadline Rate (θ): {theta} (Exp: {is_exp_drop})")
    print(f"  - Number of Servers (c): {c}")
    print(f"  - Queue Size: {'Infinite' if queue_size is None else queue_size}")
    print(f"  - Total Packets: {n_packet}")
    print(f"  - Congestion Control: {tcp_cc if tcp_cc else 'None'}")

    print("\nSimulation Statistics:")
    print(f"  - Simulation Time: {sim_time:.4f} seconds")
    print(f"  - Blocked Rate: {sim_pb:.4f} ({sim_pb * 100:.2f}%)")
    print(f"  - Dropped Rate: {sim_pd:.4f} ({sim_pd * 100:.2f}%)")
    print(f"  - Total Loss Rate: {sim_drop:.4f} ({sim_drop * 100:.2f}%)")
    print(f"  - Total Packets Processed: {int(sim_service + sim_drop + sim_block)}")
    print(f"    - Service Count: {sim_service}")
    print(f"    - Drop Count: {int(sim_drop)}")
    print(f"    - Block Count: {sim_block}")
    print(f"  - Throughput: {sim_service / sim_time:.4f} packets/second")
    print(f"  - Average Latency: {avg_latency:.4f} seconds")

if __name__ == "__main__":
    main()