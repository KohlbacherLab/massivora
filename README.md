# Massivora

A superfast pipeline for proteome-wide evolutionary coupling screens — a scalable
framework for massive sequence co-evolution analysis.

Massivora ships a native (C++/optional CUDA) core built on **NLopt** and **Eigen**,
exposed to Python via pybind11. GPU acceleration (via **CuPy**) is **optional**.

---

## Installation

There are multiple ways to install this package. But the conda installation is recommended.

### 1. Conda (recommended)

Conda is the primary, most convenient path:

- CPU only:

```bash
# use -n to customize the environment name
conda env create -f environment.yml
conda activate massivora

pip install . --no-build-isolation
```

- With CUDA compatible GPU available:

```bash
# use -n to customize the environment name
conda env create -f environment.yml
conda activate massivora

conda install -c conda-forge cupy
pip install . --no-build-isolation
```

The CUDA backend is enabled **automatically** when a CUDA compiler (`nvcc`) is
on `PATH` — no extra flag needed. Force it on/off with
`-C cmake.define.ENABLE_CUDA=ON` / `=OFF`.

### 2. pip

Dependencies: `eigen`, `nlopt`. If they are not installed, pip will try to install them. If you wish to use the analysis module of massivora, you also need to install `openstructure` manually.

- CPU only:

```bash
pip install .
```

- With CUDA GPU available:

Note that you have to determine your CUDA toolkit version here. If you have CUDA 12 installed, for example, use `[cuda12]` in the install command:

```bash
pip install ".[cuda12]"
```

The `[cuda12]` extra installs the matching CuPy wheel, and the CUDA backend is
built **automatically** because `nvcc` is detected — you no longer need to pass
`-C cmake.define.ENABLE_CUDA=ON`. (Force the build on/off with that flag set to
`ON`/`OFF` if you want to override the auto-detection.)

### 3. Docker (production)

Two images are provided.

**GPU image** (CUDA-enabled):

```bash
docker build -t massivora:gpu .
docker run --rm --gpus all massivora:gpu massivora --help
```

**CPU-only image** (smaller, no CUDA/CuPy):

```bash
docker build -f Dockerfile.cpu -t massivora:cpu .
docker run --rm massivora:cpu massivora --help
```

## System configuration

Machine/cluster-specific settings live in a **system config** at `~/.massivora.yml`.
It is created automatically (from a bundled default) the first time you run
massivora after installation, and is read from there afterwards. Edit it with:

```sh
massivora sysconf
```

This opens `~/.massivora.yml` in your editor (`$VISUAL`/`$EDITOR`, falling back to
`vi`), creating it from the bundled default if it doesn't exist yet.

When submitting to SLURM, massivora automatically uses the python/conda environment that is **currently activated** when you submit the job (`CONDA_PREFIX`), so just `conda activate` your env before running `massivora batch ...`.

For a detailed description of the fields in the system configuration file, see the documentation.

## Usage

### Pairwise coupling calculation

Prepare all the Uniprot ID of your proteins and put them into a text file, one protein per line. Then create a new project with this command:

```sh
massivora new <project_name>
```

Then you will have a config file `project_name.yml` under the directory `project_name`. Put your list of proteins to `project_name/data/proteins.txt`. You may change this path in the yml file.

Massivora relies on `JackHMMER` to do homologue search, you need to manually specify the `jackhmmer_binary` and the search `database` in the project configuration yml file under `align` section.

After the configure is done, run this command to run homologue search for proteins on the local machine:

```sh
massivora run align project_name.yml
```

If you are running Massivora on a SLURM cluster, use the `batch` command:

```sh
massivora batch align project_name.yml
```

Massivora will start to run the alignment for each protein in your text file, and create a cronjob to monitor the status. After the job has been finished, or failed too many times, the cronjob will be removed.

When homologue search has been finished, run this command to perform the pairwise coupling calculation:

```sh
massivora run couple project_name.yml
```

Same as homologue search, you can also use `batch` to run Massivora on SLURM cluster:

```sh
massivora batch couple project_name.yml
```

#### Faster couplings: `--fast-approximate`

By default the model is fitted on every column of the concatenated alignment,
including columns that are almost entirely gaps. `--fast-approximate` fits it only
on columns whose gap fraction is below `align.col_gap_threshold`:

```sh
massivora run couple project_name.yml --fast-approximate
```

Deep alignments are dominated by insert columns, so this typically removes 65–90%
of them. Cost scales with the square of the column count, so on a sample of 48
human protein pairs it gave a **median 8.0× speedup** (range 2.3–13.1×).

It also repairs the sequence reweighting. Identity is measured across all columns,
so when most of them are near-empty, gap-versus-gap agreement dominates and every
sequence looks like a neighbour of every other: on that sample the effective
sequence count was a median of **1.3 without the flag and 296 with it**.

**This changes the couplings.** It is an approximation, not a free speedup — the
dropped columns are dropped from the model. On the same sample, rank correlation
against a full run was 0.93 and roughly a quarter of the top-L predicted contacts
differed. Validate against your own benchmark before using it for published
results. Scores produced this way carry `fast_approximate` and `kept_columns`
attributes in the output Zarr array, so positions can be mapped back to the
original alignment.