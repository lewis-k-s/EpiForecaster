# Synthetic dataset Zarr layout plan

Status: phase 1 implemented; mobility BitRound (10 bits) implemented and
smoke-tested (2026-10-27). Branch: `agent/synthetic-zarr-layout`.

This plan replaces the processed synthetic NetCDF4 file with a sharded Zarr v3
store that is packed into one file. The goals are:

- parallel writes during preprocessing;
- fast per-run reads during dataset construction;
- one DVC object per dataset, as with NetCDF;
- a smaller payload, if the precision validation in section 6 permits it.

The real dataset (`real_with_id.zarr`, 202 MB) is not in scope. It stays as it
is.

## 1. Measured facts

The facts are from the b354b27 reactive dataset (198 runs × 455 dates × 945
regions, 117.7 GB) and MN5 job 48106107.

| Item | Value |
| --- | --- |
| `mobility` share of the stored bytes | 114.3 GB of 117.7 GB (97%) |
| Six latent arrays | about 1.5 GB |
| Observations, masks, ages | about 1.9 GB |
| Coordinates and run scalars | less than 1 MB |
| Mobility per run: raw / stored (zlib level 4 with shuffle) | 1.63 GB / 0.58 GB |
| h5py decode of one run of mobility | 7.0 s |
| Dataset construction in the trainer | about 30 s per run, one run at a time |
| Trainer memory after 43 runs | 30.8 GB, about 0.72 GB per run |
| NetCDF write on Nord4 (one thread, HDF5 is not thread-safe) | about 4 h |

Mobility structure in one run:

- 26.5% of the entries are positive. The positive pattern does not change over
  time, but it is different between runs.
- Positive values go from 5.7e-20 to 0.68. The values are `log1p(fraction)`.
- Every daily matrix is different. Runs do not share mobility.
- A Reactive run is close to a row-scaled Baseline run, but it is not equal to
  one: the standard deviation of the ratio per origin and date is up to 1e-2.
  Do not factor runs.

Codec measurements on one run of mobility (1.63 GB raw, one thread):

| Encoding | MB per run | 198 runs | Decode |
| --- | ---: | ---: | ---: |
| zlib 4 + shuffle (current) | 577 | 114 GB | 7.0 s |
| Blosc zstd 5 + shuffle | 567 | 112 GB | 0.6 s |
| zstd 5, positive entries only | 389 | 77 GB | — |
| BitRound 12 bits + zstd 5 | 376 | 74 GB | 0.5 s |
| BitRound 10 bits + zstd 5 | 349 | 69 GB | 0.5 s |
| BitRound 10 bits, positive entries only | 232 | 46 GB | — |
| BitRound 7 bits + zstd 5 | 265 | 52 GB | 0.4 s |
| BitRound 7 bits, positive entries only | 184 | 36 GB | — |

The maximum relative errors are 1.2e-4 for 12 bits, 4.9e-4 for 10 bits and
3.9e-3 for 7 bits. The fixed cost of dimensions and coordinates is close to
zero, so the size grows linearly with the number of runs, at about 0.59 GB per
run.

## 2. Why one packed file

Commit c53ac42 moved the processed output from a Zarr directory to NetCDF4. A
Zarr directory becomes a DVC `.dir` output, and the cost of that output grows
with the number of files:

- DVC hashes every file;
- DVC writes one cache object and one link for every file;
- `dvc push` and `dvc pull` move one object for every file;
- the `.dir` manifest lists every file;
- GPFS metadata operations grow with the number of files.

An unsharded layout with the current chunks (1, 30, 100, 100) would make about
317,000 files for mobility alone.

The plan therefore uses Zarr v3 sharding, which gives about 8,000 files, and
packs the store into one uncompressed ZIP64 file: `<dataset_name>.zarr.zip`. DVC
then tracks one file with one MD5, as it does for NetCDF. Publication, staging,
receipts and the `.dvc` pointers keep their single-file form.

A prototype with zarr 3.1.5 and xarray 2025.10.1 confirmed four things:

- one shard file is made for each variable and run;
- parallel `to_zarr(region=...)` writes by separate processes do not conflict;
- a store packed with `zipfile.ZIP_STORED` opens with
  `xr.open_zarr(zarr.storage.ZipStore(path, mode="r"))`;
- values read back exactly.

## 3. Layout

### Contract

- **Format:** Zarr v3, with `zarr_format=3` and consolidated metadata.
- **Dimensions, coordinates, variable names and attributes:** the same as the
  current NetCDF output. Consumers keep the same xarray contract.
- **Store attributes:** `layout_version = "synthetic-zarr-v1"`, the codec, and
  the BitRound mantissa bits of each variable.

### Shards and chunks

| Variable group | Shape | Shard | Inner chunk |
| --- | --- | --- | --- |
| `mobility` | (run, date, origin, destination) | (1, 455, 945, 945) | (1, 65, 945, 945), 232 MB raw |
| Run × date × region arrays (observations, latents, masks, ages) | (run, date, region) | (1, 455, 945) | the same as the shard |
| `temporal_covariates`, `mobility_time_mask` | (run, date, ·) | (1, 455, ·) | the same as the shard |
| Run-level arrays (`population`, `valid_targets`, `biomarker_data_start`, run scalars) | (run, ·) | one chunk for all runs | the same as the shard |
| Coordinates | — | one chunk | — |

Notes on the layout:

- One shard holds one run of one variable. A worker that writes run *i* is
  therefore the only writer of every shard it touches.
- Seven inner chunks in each mobility shard let one run decode on several
  threads. Each inner chunk stays under the 2 GB Blosc buffer limit.
- Run-level arrays are small. The coordinator writes them once, after the
  workers finish, so the workers do not write to a shared shard.
- There are about 40 per-run variables × 198 runs, so about 8,000 shards plus
  metadata.

### Codecs

- **Compressor:** Blosc zstd, level 5, byte shuffle, for all floating-point
  arrays. On mobility it gives a slightly better ratio than the current zlib,
  and it decodes about 12 times faster.
- **Integer masks and ages:** Blosc zstd level 5 also replaces the current
  uncompressed storage. Today these arrays are stored at 0.9×, because chunk
  overhead makes them larger than the raw data.
- **BitRound:** apply it explicitly in the preprocessing code, before the
  write, and record the mantissa bits in the variable attributes. Any Zarr
  reader can read the result. The default is no rounding; the b354b27 config
  rounds mobility to 10 bits (section 10).
- **Data type:** keep float32 for continuous values.
  - Do not store float16, because positive synthetic flows go down to 5.7e-20
    and float16 makes them zero.
  - Do not store bfloat16. It is not a core Zarr v3 data type, and it needs
    `ml_dtypes`, which is not in the environment. BitRound with 7 mantissa
    bits on float32 gives bfloat16 precision with the full float32 exponent
    range, and the compressor removes the zero bits.

### Optional later step: positive entries only

If mobility stores only its positive entries, with a positive-pattern bitmask
per run (112 KB each), the size goes down a further 33%. This changes the
dataset contract, because the loader must expand the data back to dense
matrices. Do it only after the dense layout is in use, and only if the size is
still a problem.

## 4. Parallel write

1. Write the template metadata with `to_zarr(compute=False)` into a directory
   store on node-local NVMe. The existing `node_local_output` path from commit
   602344e does this.
2. A process pool with W workers does the per-run work. For each run, a worker:
   - reads the raw run;
   - applies the existing transforms;
   - runs the per-run QA;
   - writes `region={"run_id": slice(i, i + 1)}`;
   - returns its QA summary.
3. The coordinator then does the remaining steps in order:
   - write the run-level arrays;
   - merge the per-run QA into `qa.json`;
   - consolidate the metadata;
   - pack the store into the ZIP64 file on local disk;
   - compute sha256;
   - copy the file into the attempt directory on GPFS.

The expected memory is about 6 GB per worker: one dense run (1.6 GB) plus
temporary arrays. A GPP node (256 GiB, 960 GB of NVMe) runs W = 16 with a
safety margin.

The estimated time is 198 runs / 16 workers × about 30 s, which is about 6
minutes. The current write takes about 4 hours.

The raw EpiSim corpus is one 134 GB NetCDF file. Several processes can read it
read-only at the same time with h5py. Section 5 makes the raw input use the
same layout.

## 5. EpiSim upstream coordination

Ask the EpiSim session (`codex/synthetic-observation-contract`) to do the
following:

- Write the raw synthetic corpus in the same layout:
  - Zarr v3, with one shard for each variable and run;
  - Blosc zstd, level 5;
  - packed into one uncompressed ZIP64 file;
  - written in parallel by run.
- Keep the variable names, the run identifiers and the handoff manifest
  contract.
- Record the codec, the layout version and any rounding in the store
  attributes.
- Agree the BitRound decision (section 6) before the generator applies any
  rounding. The raw corpus stays lossless until the validation passes.

During the change, EpiForecaster preprocessing reads both `.nc` and
`.zarr.zip` raw inputs. The suffix selects the reader, as
`open_processed_dataset` does now.

## 6. Precision validation

The two corpora do not use the same mobility scale:

| Corpus | Stored value | Range | Storage |
| --- | --- | --- | --- |
| Real (`real_with_id.zarr`) | `log1p(trips)` | 0 to 11.6 | float16, Blosc lz4 |
| Synthetic | `log1p(fraction)` | 0 to 0.68 | float32, zlib |

The loader converts both to float16 (`STORAGE_DTYPES["continuous"]`). Real
mobility therefore already has float16 precision: 10 mantissa bits, and values
below about 6e-8 become zero.

- BitRound with 10 or more bits does not change what the model sees from real
  data.
- BitRound with 7 bits changes both corpora.

Candidates: no rounding, 12, 10 and 7 mantissa bits.

1. **Input fidelity, on both corpora.** Compare the original and the rounded
   mobility after the loader transforms. Report:
   - edge-mask changes at `mobility_min_flow_raw`;
   - top-K neighbour set changes at `max_neighbors = 20`;
   - the maximum and median relative error of the transformed weights;
   - the imported-risk error, where it is enabled.
2. **Model fidelity, on real data.** Evaluate one fixed production checkpoint
   on real data with the original and the rounded mobility. Compare the
   canonical evaluation outputs paired by target, region, horizon and date.
   Declare the tolerance before the run. This check is necessary only for
   candidates below 10 bits.
3. **Decision rule:** use the smallest number of mantissa bits that passes
   both checks. Record it in the preprocessing config and in the store
   attributes.

## 7. Loader memory (dependency)

The trainer builds and keeps every run at startup. At about 0.72 GB per run,
198 runs need about 142 GB. The single-GPU ACC allocation has 125,000 MiB
(about 131 GB), so the full reactive pretraining run will run out of memory
before the first batch.

A smaller file does not solve this. The loader must build runs when the
curriculum needs them:

- a bounded cache of active runs, sized from the curriculum `active_runs`
  setting;
- background construction of the next runs in a process pool;
- release of inactive runs.

Fast per-run decode (section 3) and parallel construction make this practical.
This work is separate from the layout change, but the full pretraining run
needs both.

## 8. Phases and acceptance

| Phase | Work | Acceptance |
| --- | --- | --- |
| 1 | Zarr writer (template, parallel workers, run-level arrays, pack, sha256); `.zarr.zip` reader dispatch; QA and publication accept `.zarr.zip` | Unit tests on the minimal synthetic corpus: the parallel output equals the serial output; the ZIP reopens; the DVC pointer has one MD5 |
| 2 | Rebuild b354b27 as Zarr without rounding, on GPP with W = 16 | Values identical to the NetCDF dataset; write time and size recorded |
| 3 | Precision validation (section 6) | Mantissa bits decided and recorded |
| 4 | EpiSim raw output in the same layout | A raw corpus is accepted by the handoff QA; parallel read in preprocessing |
| 5 | Lazy loader (section 7) | 198-run pretraining starts within the ACC memory limit; peak RSS measured on a compute node |

## 9. Phase 1 result

`python -m data.preprocess.convert_netcdf` converted the b354b27 NetCDF dataset
on `alogin3` (MN5 job 48106107, `acc_interactive`, SATA SSD, 12 workers):

| Item | Value |
| --- | --- |
| Runs written and verified by read-back | 198 of 198 |
| Conversion time (write, read-back, pack) | 1403 s |
| Packed size | 114.33 GB (NetCDF: 117.83 GB) |
| ZIP members | 5,997 |
| One run of mobility, NetCDF / packed Zarr | 6.9-9.2 s / 1.5-2.8 s |
| Peak memory, 12 workers | about 43 GB |

Spot checks after packing: dimensions, variable set, attributes, run IDs,
three runs of mobility, and the `latent_I_true`, `cases_mask`, `population`,
`valid_targets` and `temporal_covariates` arrays are identical to the NetCDF
dataset.

The packed file was on node-local disk only and was not published. A
published Zarr dataset must come from a workflow run (phase 2).

The per-run time (about 85 s for each worker) is much higher than the
single-thread mobility codec cost (about 18 s). Profile the worker before
phase 2: the source NetCDF read through Dask and the per-variable region
writes are the likely costs.

## 10. Mobility rounding and training smoke

`preprocessing.bitround_keepbits` rounds named float32 variables before the
write (round half to even). The read-back compares against the rounded values,
and the bits are in the variable attribute `bitround_keepbits` and the store
attribute `zarr_bitround_keepbits_json`. The b354b27 config rounds `mobility`
to 10 bits. That is the float16 mantissa that the trainer uses
(`STORAGE_DTYPES["continuous"]`), so the model input is the same except at
float16 subnormals. The float32 exponent range stays, so small positive flows
stay positive.

Test on the first 8 runs of b354b27, converted on `alogin3` (MN5 job
48127780, 8 workers):

| Item | Lossless | BitRound 10 |
| --- | ---: | ---: |
| Packed size | 4.63 GB | 2.89 GB (-38%) |
| Conversion time | 61 s | 58 s |
| Validation joint loss after 200 steps | 0.7258 | 0.7246 |
| Validation mean target CRPS | 0.3401 | 0.3396 |

Input fidelity on 3 runs, with the trainer's transforms
(`mobility_min_flow_raw = 0`, clip at 8, float16 cast):

- edge-mask changes: 0;
- relative error of the stored values: maximum 4.9e-4, median 1.7e-4;
- equal float16 weights: 99.5%; the largest difference is 6e-8, which is one
  float16 subnormal step;
- changed top-20 incoming-neighbour sets: 4e-5 to 1e-4 of the
  (date, destination) pairs.

The trainer built each run from the packed store in about 11 s (NetCDF: about
30 s). It trained at about 406 samples/s, without compilation, on one H100.

Real data is not affected. Its mobility is already float16 and this rounding
applies only to synthetic outputs. Rounding below 10 bits still needs the
model-fidelity check in section 6.

### Full b354b27 rebuild

Run `b354b27-zarr-br10-20261027` (job 48128720, GPP, source d98807b)
rebuilt the 198-run dataset from the raw NetCDF with `bitround_keepbits:
{mobility: 10}` and 12 write workers.

| Item | Value |
| --- | ---: |
| Preprocessing step | 28 min 33 s, peak 62.7 GiB |
| Publication step | 4 min 38 s |
| Packed size | 71,369,022,411 bytes (NetCDF: 117,834,036,904) |
| DVC md5 | `8308952213e3700693e349a9b2371399` |

Validation job 48158598 compared the packed dataset with the published
NetCDF dataset (md5 `946dae85…`). Dimensions, variables, coordinates and
static arrays are identical. Every per-run variable of all 198 runs is
identical, and mobility equals BitRound(NetCDF mobility, 10). The only new
attributes are the `zarr_*` store attributes and the mobility attribute
`bitround_keepbits`. The check took 10 minutes with 12 worker processes.

Open the NetCDF reference in each worker after the workers start (spawn).
An earlier attempt forked 20 workers after the parent had opened the file
through HDF5, and no worker completed a run in 3 hours.

## 11. Open decisions

- Inner chunk length along `date` for mobility (65 proposed). This value sets
  the parallelism of a single-run decode.
- Whether `qa.json` reads back the full packed store or uses the per-run QA
  from the workers. The per-run QA avoids a second full read.
- Whether phase 5 also stores the kept graph weights sparsely (top-K per
  target node) to reduce the 396 MB per run in memory.
