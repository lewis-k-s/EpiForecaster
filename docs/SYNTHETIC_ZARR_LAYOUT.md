# Synthetic dataset Zarr layout

Processed synthetic datasets are sharded Zarr v3 stores packed into one
uncompressed ZIP64 file (`<dataset_name>.zarr.zip`, layout version
`synthetic-zarr-v1`). The code is in `data/preprocess/zarr_layout.py`.

The layout gives parallel writes during preprocessing, fast per-run reads
during dataset construction, and one DVC object per dataset. The `mobility`
variable is about 97% of the stored bytes, so its encoding sets the size.

The real dataset (`real_with_id.zarr`) is not in this layout.

## 1. Why one packed file

An earlier change moved the processed output from a Zarr directory to NetCDF4. A
Zarr directory becomes a DVC `.dir` output, and the cost of that output grows
with the number of files:

- DVC hashes every file;
- DVC writes one cache object and one link for every file;
- `dvc push` and `dvc pull` move one object for every file;
- the `.dir` manifest lists every file;
- GPFS metadata operations grow with the number of files.

An unsharded layout with the current chunks (1, 30, 100, 100) would make about
317,000 files for mobility alone.

This layout uses Zarr v3 sharding, which gives about 8,000 files, and
packs the store into one uncompressed ZIP64 file: `<dataset_name>.zarr.zip`. DVC
then tracks one file with one MD5, as it does for NetCDF. Publication, staging,
receipts and the `.dvc` pointers keep their single-file form.

A prototype with zarr 3.1.5 and xarray 2025.10.1 confirmed four things:

- one shard file is made for each variable and run;
- parallel `to_zarr(region=...)` writes by separate processes do not conflict;
- a store packed with `zipfile.ZIP_STORED` opens with
  `xr.open_zarr(zarr.storage.ZipStore(path, mode="r"))`;
- values read back exactly.

## 2. Layout

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
  rounds mobility to 10 bits.
- **Data type:** keep float32 for continuous values.
  - Do not store float16, because positive synthetic flows go down to 5.7e-20
    and float16 makes them zero.
  - Do not store bfloat16. It is not a core Zarr v3 data type, and it needs
    `ml_dtypes`, which is not in the environment. BitRound with 7 mantissa
    bits on float32 gives bfloat16 precision with the full float32 exponent
    range, and the compressor removes the zero bits.

### Possible later step: positive entries only

If mobility stored only its positive entries, with a positive-pattern bitmask
per run (112 KB each), the size goes down a further 33%. This changes the
dataset contract, because the loader must expand the data back to dense
matrices. This is not done.

## 3. Parallel write

1. Write the template metadata with `to_zarr(compute=False)` into a directory
   store on node-local NVMe. The `node_local_output` option does this.
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

Each worker needs about 6 GB: one dense run (1.6 GB) plus temporary arrays.
Size `preprocessing.write_workers` from the node memory.

## 4. BitRound

`preprocessing.bitround_keepbits` rounds named float32 variables before the
write (round half to even). The read-back compares against the rounded values.
The bits are in the variable attribute `bitround_keepbits` and the store
attribute `zarr_bitround_keepbits_json`. The default is no rounding.

The trainer stores continuous values as float16 (`STORAGE_DTYPES["continuous"]`),
which has 10 mantissa bits. Rounding mobility to 10 bits therefore does not
change the model input except at float16 subnormals, and the float32 exponent
range keeps small positive flows positive. Rounding below 10 bits changes the
model input and needs a model-fidelity check before use.

Do not store float16 mobility in the processed file: positive synthetic flows
go down to 5.7e-20 and float16 makes them zero.

## 5. Conversion

`python -m data.preprocess.convert_netcdf` converts an earlier NetCDF output
to this layout without changing its values. Open the NetCDF source in each
worker after the workers start (spawn). Forking workers after the parent has
opened the file through HDF5 can stall every worker.
