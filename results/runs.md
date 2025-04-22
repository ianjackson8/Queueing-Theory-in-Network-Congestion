# M/M/1 Runs
## Experiment Set 1: Varying Arrival Rate ($\lambda$) to Induce Congestion

| Experiment | TCP Variant | λ (Arrival Rate) | μ (Service Rate) | Queue Size | θ (Deadline) | Loss Rate (%)  | Throughput | Avg Latency  |
|------------|-------------|------------------|------------------|------------|--------------|----------------|------------|--------------|
| E1.1.R     | Reno        | 5                | 10               | 5          | 1            | 00.00%         | 0.0000     |              |
| E1.2.R     | Reno        | 10               | 10               | 5          | 1            | 00.00%         | 0.0000     |              |
| E1.3.R     | Reno        | 20               | 10               | 5          | 1            | 00.00%         | 0.0000     |              |
| E1.1.C     | Cubic       | 5                | 10               | 5          | 1            | 00.00%         | 0.0000     |              |
| E1.2.C     | Cubic       | 10               | 10               | 5          | 1            | 00.00%         | 0.0000     |              |
| E1.3.C     | Cubic       | 20               | 10               | 5          | 1            | 00.00%         | 0.0000     |              |

## Experiment Set 2: Varying Queue Size

| Experiment | TCP Variant | λ  | μ  | Queue Size | θ  | Loss Rate (%)  | Throughput | Avg Latency  |
|------------|-------------|----|----|------------|----|----------------|------------|--------------|
| E2.1.R     | Reno        | 15 | 10 | 1          | 1  |                |            |              |
| E2.2.R     | Reno        | 15 | 10 | 5          | 1  |                |            |              |
| E2.3.R     | Reno        | 15 | 10 | 10         | 1  |                |            |              |
| E2.1.C     | Cubic       | 15 | 10 | 1          | 1  |                |            |              |
| E2.2.C     | Cubic       | 15 | 10 | 5          | 1  |                |            |              |
| E2.3.C     | Cubic       | 15 | 10 | 10         | 1  |                |            |              |

## Experiment Set 3: Varying Deadline Sensitivity ($\theta$)

| Experiment | TCP Variant | λ  | μ  | Queue Size | θ    | Loss Rate (%)  | Throughput | Avg Latency  |
|------------|-------------|----|----|------------|------|----------------|------------|--------------|
| E3.1.R     | Reno        | 12 | 10 | 5          | 0.1  |                |            |              |
| E3.2.R     | Reno        | 12 | 10 | 5          | 1    |                |            |              |
| E3.3.R     | Reno        | 12 | 10 | 5          | 10   |                |            |              |
| E3.1.C     | Cubic       | 12 | 10 | 5          | 0.1  |                |            |              |
| E3.2.C     | Cubic       | 12 | 10 | 5          | 1    |                |            |              |
| E3.3.C     | Cubic       | 12 | 10 | 5          | 10   |                |            |              |

## Experiment Set 4: Short vs. Long Flow Durations

| Experiment | TCP Variant | λ  | μ  | Queue Size | n_packet | θ  | Loss Rate (%)  | Throughput | Avg Latency  |
|------------|-------------|----|----|------------|----------|----|----------------|------------|--------------|
| E4.1.R     | Reno        | 10 | 10 | 5          | 50       | 1  |                |            |              |
| E4.2.R     | Reno        | 10 | 10 | 5          | 300      | 1  |                |            |              |
| E4.1.C     | Cubic       | 10 | 10 | 5          | 50       | 1  |                |            |              |
| E4.2.C     | Cubic       | 10 | 10 | 5          | 300      | 1  |                |            |              |

## Experiment Set 5: Random Early Drop Enabled

| Experiment | TCP Variant | λ  | μ  | Queue Size | is_exp_drop  | θ  | Loss Rate (%)  | Throughput | Avg Latency  |
|------------|-------------|----|----|------------|--------------|----|----------------|------------|--------------|
| E5.1.R     | Reno        | 15 | 10 | 5          | True         | 1  |                |            |              |
| E5.1.C     | Cubic       | 15 | 10 | 5          | True         | 1  |                |            |              |
