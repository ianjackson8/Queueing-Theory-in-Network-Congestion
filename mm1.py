'''
M/M/1 Queue Simulation Implementation

Author: Ian Jackson
Date: 3/17/2025
'''

#== Imports ==#
import os
import heapq

import matplotlib.pyplot as plt

from enum import Enum
from typing import Union
from numpy import random as rnd

#== Congestion Control Classes ==#
class CongestionControl:
    '''
    base class for congestion control algorithms.
    this class can be extended to implement specific congestion control strategies.
    '''
    def __init__(self):
        self.cwnd = 1
        self.ssthresh = 2
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
            self.cwnd += 1
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

#== MM1 Queue Classes ==#
class PacketStatus(Enum):
    ARRIVAL = 0
    DROP = 1
    SERVICED = 2

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
    def __init__(self, packet_id: int, event_time: float, status: PacketStatus = PacketStatus.ARRIVAL):
        '''
        init instance of an event

        Args:
            packet_id (int): packet id
            event_time (float): when event occurs
            status (PacketStatus, optional): packet status. Defaults to PacketStatus.ARRIVAL.
        '''
        self.packet_id = packet_id
        self.event_time = event_time
        self.status = status

class Queue():
    '''
    class to represent a queue (M/M/1)
    '''
    def __init__(self, size: int | None):
        '''
        init instance of queue

        Args:
            size (int | None): size of queue (None is infinite)
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

        # tracking/logging
        self.time_log = [0]
        self.queue_len_log = [0]
        self.service_log = [0]
        self.drop_log = [0]
        self.block_log = [0]

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

        if status == PacketStatus.ARRIVAL:
            self.queue_len = len(self.priority_queue)

            # check if the queue is full
            if self.size is not None and self.queue_len >= self.size:
                # if so, then packet blocked
                self.n_block += 1  
                print(f"[{event_time:.4f}] [{queue_len}] Packet {packet.id} BLOCKED")
                return None  

            # add packet to priority queue
            self.priority_queue.append(packet)

            # if queue was empty, process immediately; else, it waits for its limit time (expiration)
            if self.queue_len == 0:
                return Event(packet.id, packet.service_end_time, PacketStatus.SERVICED)
            else:
                return Event(packet.id, packet.limit_time, PacketStatus.DROP)

        elif status == PacketStatus.DROP:
            # packet expires before being serviced
            self.n_drop += 1  
            print(f"[{event_time:.4f}] [{queue_len}] Packet {packet.id} DROPPED")

            # remove packet from priority and event queues
            self.priority_queue = [p for p in self.priority_queue if p.id != packet.id]
            self.event_queue = [(et, e) for (et, e) in self.event_queue if e.packet_id != packet.id]

            # no further events for this packet
            return None  

        elif status == PacketStatus.SERVICED:
            # packet successfully gets serviced and exits
            self.n_service += 1 
            print(f"[{event_time:.4f}] [{queue_len}] Packet {packet.id} SERVICED")

            # remove the serviced packet from the queue
            self.priority_queue = [p for p in self.priority_queue if p.id != packet.id]
            self.event_queue = [(et, e) for (et, e) in self.event_queue if e.packet_id != packet.id]

            # if the queue is now empty, nothing to do
            if len(self.priority_queue) == 0:
                return None

            # otherwise, schedule service for the next packet in the queue
            next_packet = self.priority_queue[0]
            return Event(next_packet.id, event_time + next_packet.service_time, PacketStatus.SERVICED)
        
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
            elif event.status == PacketStatus.DROP:
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
        if status in [PacketStatus.SERVICED, PacketStatus.DROP]:
            self.time_log.append(cur_time)
            self.queue_len_log.append(len(self.priority_queue))
            self.service_log.append(self.n_service)
            self.drop_log.append(self.n_drop)
            self.block_log.append(self.n_block)

class Simulator():
    def __init__(self, lmbda: float, mu: float, theta: float, size: int | None, is_exp_drop: bool = False, use_tcp_reno: bool = False):
        '''
        init instance of simulator

        Args:
            lmbda (float): arrival rate of packets
            mu (float): service rate
            theta (float): deadline rate or deadline time
            size (int | None): size of queue (None if infinite)
            is_exp_drop (bool, optional): does deadline time follow exp dist. Defaults to False.
            uce_tcp_reno (bool, optional): use TCP Reno congestion control. Defaults to False.
        '''
        self.lmbda = lmbda
        self.mu = mu
        self.theta = theta
        self.size = size
        self.is_exp_drop = is_exp_drop

        self.use_tcp_reno = use_tcp_reno
        self.cc = TCPReno() if use_tcp_reno else None

        # create instance of queue
        self.queue = Queue(self.size)

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

        # process packets until specified num of packet is reached
        while packet_id < n_packet:
            cwnd = self.cc.get_cwnd() if self.use_tcp_reno else 1

            for _ in range(cwnd):
                if packet_id >= n_packet:
                    break

                # generate new packet and insert into queue
                packet = Packet(packet_id, cur_time, self.mu, self.theta, self.is_exp_drop)
                print(f"[{cur_time:.4f}] [{len(self.queue.priority_queue)}] Packet {packet_id} ARRIVAL")

                self.queue.insert_packet(packet)

                # log the current time and queue state
                # self.queue.log_stats(cur_time)

                # advance sim time and increment packet id
                cur_time += rnd.exponential(1 / self.lmbda)
                packet_id += 1

            # handle jobs in the queue
            while self.queue.event_queue and cur_time >= self.queue.event_queue[0][0]:
                self.queue.handle_jobs(self.cc)

        # compute and return blocking and dropping probabilities
        return self.queue.n_block / n_packet, self.queue.n_drop / n_packet, (self.queue.service_log[-1], self.queue.drop_log[-1], self.queue.block_log[-1])

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
        plt.plot(sorted_times, sorted_drop, label="Dropped Packets")
        plt.plot(sorted_times, sorted_block, label="Blocked Packets", color='red')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Count')
        plt.title('Cumulative Packet Events Over Time')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(results_folder, "cumulative_packet_events_over_time.png"))

        if self.cc:
            zipped = sorted(zip(self.cc.time_log, self.cc.cwnd_log, self.cc.state_log))
            times, cwnds, states = zip(*zipped)
            plt.figure(figsize=(12, 4))
            plt.plot(times, cwnds, color='black', alpha=0.4, label='TCP cwnd')

            # Fill background based on TCP state
            last_time = times[0]
            last_state = states[0]
            ax = plt.gca()
            for t, s in zip(times[1:], states[1:]):
                if s != last_state:
                    ax.axvspan(last_time, t, alpha=0.2,
                               color={'Slow Start': 'blue', 'Additive Increase': 'yellow', 'Multiplicative Decrease': 'red'}.get(last_state, 'gray'),
                               label=last_state if last_state not in ax.get_legend_handles_labels()[1] else "")
                    last_time = t
                    last_state = s
            # Mark final span
            ax.axvspan(last_time, times[-1], alpha=0.2,
                       color={'Slow Start': 'blue', 'Additive Increase': 'yellow', 'Multiplicative Decrease': 'red'}.get(last_state, 'gray'),
                       label=last_state if last_state not in ax.get_legend_handles_labels()[1] else "")

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
    theta = 1 # deadline rate or fixed deadline time (θ)
    lmbda = 10 # arrival rate (λ)
    n_packet = 20

    queue_size = 5 
    is_exp_drop = False
    
    # init simulator
    sim = Simulator(lmbda=lmbda, mu=mu, theta=theta, size=queue_size, is_exp_drop=is_exp_drop, use_tcp_reno=True)

    # run sim
    sim_pb, sim_pd, sim_counts = sim.run(n_packet=n_packet)
    sim_service, sim_drop, sim_block = sim_counts
    sim.plot_results()
    
    # print results
    print(f"\n=== Simulation Results for M/M/1 Queue ===")
    print("Simulation Parameters:")
    print(f"  - Arrival Rate (λ): {lmbda}")
    print(f"  - Service Rate (μ): {mu}")
    print(f"  - Deadline Rate (θ): {theta} (Exp: {is_exp_drop})")
    print(f"  - Queue Size: {'Infinite' if queue_size is None else queue_size}")
    print(f"  - Total Packets: {n_packet}")

    print("\nSimulation Statistics:")
    print(f"  - Blocked Rate: {sim_pb:.4f} ({sim_pb * 100:.2f}%)")
    print(f"  - Dropped Rate: {sim_pd:.4f} ({sim_pd * 100:.2f}%)")
    print(f"  - Service Count: {sim_service}")
    print(f"  - Drop Count: {sim_drop}")
    print(f"  - Block Count: {sim_block}")
    print(f"  - Total Packets Processed: {sim_service + sim_drop + sim_block}")

if __name__ == "__main__":
    main()