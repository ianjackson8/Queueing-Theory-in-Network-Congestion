'''
Queueing Theory in Network Congestion Control

Author: Ian Jackson
Version: v0.1
'''

#== Imports ==#
import argparse
import random
import math
import heapq

from enum import Enum
from queue import PriorityQueue

random.seed(42) # random seed for reproducibility

#== Global Variables ==#
ARRIVAL_RATE = 1.0
SERVICE_RATE = 1.0
NUM_SERVERS = 1
SIM_TIME = 10000.0
BUFFER_SIZE = 20
RTT = 1.0

event_queue = []
queue_buffer = []

cur_time = 0
server_busy = False

#== Classes ==#
class TCP_STATE(Enum):
    SLOW_START = 1
    CONGESTION_AVOIDANCE = 2
    CONGESTION_DETECTION = 3

class EVENT_TYPE(Enum):
    ARRIVAL = 1
    DEPARTURE = 2
    TIMEOUT = 3
    MONITOR = 4

class TCP_Flow:
    def __init__(self, flow_id: int, cwnd: float, ssthresh: float):
        self.flow_id = flow_id
        self.cwnd = cwnd
        self.ssthresh = ssthresh
        # self.state = state
        # self.arrival_rate = arrival_rate

    def update_arrival_rate(self):
        return self.cwnd / RTT
    
    def on_ack(self):
        if self.cwnd < self.ssthresh:
            self.cwnd *= 2
        else:
            self.cwnd += 1

    def on_drop(self):
        self.ssthresh = max(self.cwnd // 2, 1)
        self.cwnd = 1

class Packet:
    packet_counter = 0

    def __init__(self, arrival_time: float, flow_id: int, size: float):
        self.id = Packet.packet_counter
        Packet.packet_counter += 1
        self.arrival_time = arrival_time
        self.flow_id = flow_id
        self.size = size

class Event():
    def __init__(self, time: float, type: EVENT_TYPE, packet: Packet):
        self.time = time
        self.type = type
        self.packet = packet

    def __lt__(self, other):
        return self.time < other.time

#== Methods ==#
def schedule_event(event: Event):
    heapq.heappush(event_queue, event)

def exp_time(rate: float):
    return random.expovariate(rate) if rate > 0 else float('inf')

def schedule_next_arrival(flow_obj: TCP_Flow):
    global cur_time
    rate = flow_obj.update_arrival_rate()

    inter_arrival = exp_time(rate)
    arrival_time = cur_time + inter_arrival

    if arrival_time <= SIM_TIME:
        packet = Packet(arrival_time=arrival_time, flow_id=flow_obj.flow_id, size=1.0)
        schedule_event(Event(time=arrival_time, type=EVENT_TYPE.ARRIVAL, packet=packet))

def start_service():
    global server_busy, cur_time

    if not server_busy and queue_buffer:
        server_busy = True

        service_time = exp_time(SERVICE_RATE)
        departure_time = cur_time + service_time

        departing_packet = queue_buffer[0]
        schedule_event(Event(time=departure_time, type=EVENT_TYPE.DEPARTURE, packet=departing_packet))

#== Main Execution ==#
def main():
    # initialize simulation state
    flow = TCP_Flow(flow_id=0, cwnd=1.0, ssthresh=16.0)

    # logging / stats
    total_dropped_packets = 0
    served_packets = 0
    cum_queue_length = 0
    num_samples = 0

    # main sim setup
    schedule_next_arrival(flow)

    # main event loop
    while event_queue:
        event = heapq.heappop(event_queue)
        cur_time = event.time

        if cur_time > SIM_TIME:
            break

        cum_queue_length += len(queue_buffer)
        num_samples += 1

        if event.type == EVENT_TYPE.ARRIVAL:
            if (BUFFER_SIZE is not None) and (len(queue_buffer) >= BUFFER_SIZE):
                total_dropped_packets += 1
                flow.on_drop()
            else:
                queue_buffer.append(event.packet)
                start_service()
        elif event.type == EVENT_TYPE.DEPARTURE:
            if queue_buffer and queue_buffer[0].id == event.packet.id:
                queue_buffer.pop(0)
                served_packets += 1

            server_busy = False
            flow.on_ack()
            start_service()

    avg_queue_length = cum_queue_length / num_samples if num_samples > 0 else 0.0
    drop_rate = total_dropped_packets / (served_packets + total_dropped_packets) if (served_packets + total_dropped_packets) > 0 else 0.0

    print("=== Simulation Results ===")
    print(f"Simulation Time            : {cur_time:.2f} seconds")
    print(f"Total Packets Served       : {served_packets}")
    print(f"Total Packets Dropped      : {total_dropped_packets}")
    print(f"Drop Rate                  : {drop_rate:.3f}")
    print(f"Average Queue Length       : {avg_queue_length:.2f}")
    print("")
    print("TCP Flow Statistics:")
    print(f"  Final cwnd               : {flow.cwnd}")
    print(f"  Final ssthresh           : {flow.ssthresh}")

if __name__ == "__main__":
    main()
