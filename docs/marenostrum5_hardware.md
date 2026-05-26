# MareNostrum 5 System Overview

MareNostrum 5 is a pre-exascale EuroHPC supercomputer supplied by Bull SAS that combines Lenovo ThinkSystem SD650 V3 and Eviden BullSequana XH3000 architectures, providing two partitions with different technical characteristics.

---

## MareNostrum 5 GPP (General Purpose Partition)

The MareNostrum 5 GPP is a general-purpose system housing **6,408 nodes** based on Intel Sapphire Rapids (4th Generation Intel Xeon Scalable Processors), along with an additional **72 nodes** featuring Intel Sapphire Rapids HBM (High Bandwidth Memory).

This configuration results in a total of **726,880 processor cores** and **1.75PB of main memory**.

> **Note:** In every node, there's a small amount of memory reserved for the operating system and GPFS, so there are limits at a Slurm level of how much memory is assigned to every core.

### Node Specifications

| Node Type | Node Count | Cores per Node | Main Memory per Node | Usable Memory per Core |
|-----------|------------|----------------|----------------------|------------------------|
| GPP | 6,192 | 112 | 256 GiB | 2 GB |
| GPP-HighMem | 216 | 112 | 1,024 GiB | 9 GB |
| GPP-Data | 10 | 112 | 2,048 GiB | 18 GB |
| GPP-HBM | 72 | 112 | 128 GiB | 1 GB |

### Hardware Details

#### GPP, GPP-HighMem Nodes

- **CPU:** 2x Intel Xeon Platinum 8480+ 56C 2GHz
- **Memory:**
  - GPP: 16x DIMM 16GB 4800MHz DDR5
  - GPP-HighMem: 16x DIMM 64GB 4800MHz DDR5
- **Storage:** 960GB NVMe local storage
- **Network:** ConnectX-7 NDR200 InfiniBand (shared by two nodes, **100Gb/s bandwidth per node**)

#### GPP-Data Nodes

- **CPU:** 2x Intel Xeon Platinum 8480+ 56C 2GHz
- **Memory:** 32x DIMM 64GB 4800MHz DDR5
- **Storage:** 960GB NVMe local storage
- **Network:** ConnectX-7 NDR200 InfiniBand (**200Gb/s bandwidth per node**)

#### GPP-HBM Nodes

- **CPU:** 2x Intel Xeon CPU Max 9480 56C 1.9GHz
- **Memory:** 8x Die 16GB 3200MHz HBM2
- **Storage:** 960GB NVMe local storage
- **Network:** ConnectX-7 NDR200 InfiniBand (shared by two nodes, 100Gb/s bandwidth per node)

> **Note:** Memory units are written as "GB" for simplicity, but they are in fact GiB. A GPP node has 263,812,180 kB of memory.

---

## MareNostrum 5 ACC (Accelerated Partition)

The MareNostrum 5 ACC accelerated block comprises **1,120 nodes** based on Intel Xeon Sapphire Rapids processors and NVIDIA Hopper GPUs, offering a total (CPUs + GPUs) of **680,960 compute units**.

### ACC Node Specifications

- **CPU:** 2x Intel Xeon Platinum 8460Y+ 40C 2GHz (80 cores per node)
- **GPU:** 4x NVIDIA Hopper H100 64GB HBM2
- **Memory:** 16x DIMM 32GB 4800MHz DDR5 (512GB main memory per node, 6.25GB usable per core)
- **Storage:** 480GB NVMe local storage
- **Network:** 4x ConnectX-7 NDR200 InfiniBand (**800Gb/s bandwidth per node**)

---

## InfiniBand Network Topology

The MareNostrum 5 InfiniBand topology is a **three-layer fat-tree** with a total of **324 switches** (model QM9790).

### Network Layout

- **3 GPP islands**
- **1 storage island**
- **7 ACC islands**

### Layer Structure

| Layer | Description |
|-------|-------------|
| Layer 1 | Switches directed to the nodes |
| Layer 2 | Connects node switches to switch cores |
| Layer 3 | Core switches |

### Routing

- **Advanced routing** and **adaptive routing** configurations available
- **Up/Down routing** with fault tolerance

---

## Performance Implications for ML Workloads

### Storage

- **960GB NVMe per node** (480GB on ACC) - Use for staging datasets to avoid NFS/GPFS contention
- Copy data to `$TMPDIR` or local NVMe at job startup for better I/O performance

### Network

- GPP nodes share 100Gb/s between pairs (~50Gb/s effective)
- GPP-Data nodes have dedicated 200Gb/s
- ACC nodes have 800Gb/s - ideal for distributed GPU training

### Memory

- Standard GPP: 256GB/node (2GB/core usable)
- HighMem available for memory-intensive workloads
- HBM nodes for bandwidth-sensitive applications

---

## SLURM Queues and Job Submission

MareNostrum 5 uses Slurm for batch processing. All jobs require specifying an account (`--account`) and QoS (`--qos`).

### Viewing Available Resources

```bash
# List your available accounts (unixgroups)
bsc_project list

# List available queues and their limits
bsc_queues
```

### Standard Queues (QoS)

#### GPP Partition

| Queue | Max Nodes (Cores) | Wallclock | Slurm QoS Name |
|-------|-------------------|-----------|----------------|
| BSC | 125 (14,000) | 48h | `gp_bsc{case,cs,es,ls}` |
| Data | 4 (448) | 72h | `gp_data` |
| Debug | 32 (3,584) | 2h | `gp_debug` |
| EuroHPC | 800 (89,600) | 72h | `gp_ehpc` |
| HBM | 50 (5,600) | 72h | `gp_hbm` |
| Interactive | 1 (32) | 2h | `gp_interactive` |
| RES Class A | 200 (22,400) | 72h | `gp_resa` |
| RES Class B | 200 (22,400) | 48h | `gp_resb` |
| RES Class C | 50 (5,600) | 24h | `gp_resc` |
| Training | 32 (3,584) | 48h | `gp_training` |

#### ACC Partition (GPU)

| Queue | Max Nodes (Cores) | Wallclock | Slurm QoS Name |
|-------|-------------------|-----------|----------------|
| BSC | 25 (2,000) | 48h | `acc_bsc{case,cs,es,ls}` |
| Debug | 8 (640) | 2h | `acc_debug` |
| EuroHPC | 100 (8,000) | 72h | `acc_ehpc` |
| Interactive | 1 (40) | 2h | `acc_interactive` |
| RES Class A | 100 (8,000) | 72h | `acc_resa` |
| RES Class B | 100 (8,000) | 48h | `acc_resb` |
| RES Class C | 10 (800) | 24h | `acc_resc` |
| Training | 4 (320) | 48h | `acc_training` |

> **Note:** Each BSC QoS has a limit of 320 nodes for GPP and 80 nodes for ACC partition. This limit refers to the total number of nodes being used by all running jobs in that queue.

### Essential SBATCH Directives

```bash
#SBATCH --qos={qos}              # Queue (required)
#SBATCH --account={account}       # Slurm account (required)
#SBATCH --time={DD-HH:MM:SS}      # Max runtime
#SBATCH --ntasks={number}         # Number of processes
#SBATCH --cpus-per-task={number}  # Threads per process
#SBATCH --nodes={number}          # Number of nodes
#SBATCH --chdir={pathname}        # Working directory
#SBATCH --output={filename}       # stdout file
#SBATCH --error={filename}        # stderr file
#SBATCH --constraint=highmem      # Request high-memory nodes
```

### GPU Jobs (ACC Partition)

```bash
#SBATCH --gres=gpu:{1-4}         # GPUs per node (1-4)
```

**Important:** For each GPU requested, you must request 20 CPUs (each ACC node has 80 CPUs and 4 GPUs).

**GPU Examples:**

```bash
# Single GPU, single MPI task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:1

# Three GPUs, six MPI tasks
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:3

# Eight GPUs across two nodes (4 per node)
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=20
#SBATCH --gres=gpu:4
```

### High-Memory Nodes

Request high-memory nodes (216 available, 1TB RAM each):

```bash
#SBATCH --constraint=highmem
```

**Note:** High-memory nodes have significantly longer queue times. Consider using standard nodes with reduced processes per node instead.

### Interactive Jobs

```bash
# GPP interactive session
salloc -A {account} -q gp_interactive -t 00:10:00 -n 1 -c 4

# ACC interactive session with GPUs
salloc -A {account} -q acc_bsccase -n 2 -c 20 --gres=gpu:2
```

### Job Arrays

```bash
#SBATCH --array=1-10
```

Environment variables available:
- `SLURM_ARRAY_JOB_ID`: Initial job ID of the array
- `SLURM_ARRAY_TASK_ID`: Current array index

### Job Status Codes

| Code | Meaning |
|------|---------|
| `CD` | COMPLETED - Job finished |
| `CG` | COMPLETING - Job finishing, some processes still active |
| `F` | FAILED - Job terminated with non-zero exit |
| `PD` | PENDING - Waiting for resources |
| `R` | RUNNING - Job allocated and running |
| `PR` | PREEMPTED - Job terminated by preemption |

### Common Reason Codes

| Code | Meaning |
|------|---------|
| `Priority` | Higher priority jobs in queue |
| `Resources` | Waiting for resources |
| `QOSGrpNodeLimit` | All nodes for QoS in use |
| `QOSGrpCpuLimit` | All CPUs for QoS in use |
| `InvalidAccount` | Invalid account specified |
| `InvaldQoS` | Invalid QoS specified |

### Resource Accounting

- **Core hours:** 1 core running for 1 hour = 1 core-hour
- **Full GPP node:** 112 cores × 1 hour = 112 core-hours
- **GPU jobs:** Accounted based on CPUs requested (20 CPUs per GPU)
  - Example: 2 full ACC nodes (160 CPUs) for 12 hours = 1,920 core-hours

### Job Priority Factors

1. **Job size:** Larger jobs (more cores) get higher priority
2. **Wait time:** Jobs gain priority the longer they wait
3. **Fair share:** Groups with less usage get higher priority

Check your fair-share score:
```bash
sshare -la
```
