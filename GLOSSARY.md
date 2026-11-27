# EpiForecaster

Forecasting of epidemiological observations for public-health situational awareness. A model infers a latent SIRHD epidemic from sparse observations and maps it back to forecasts of those observations.

## Language

### Epidemic model

**Compartment**:
One of the five population states Susceptible, Infected, Hospitalized, Recovered and Dead.

**Latent state**:
The fractions of a region's population in each compartment at one time step. The fractions sum to one.
_Avoid_: true infection rate, hidden state

**Latent flow**:
The fraction of a region's population that moves between compartments in one time step: infection flow (S to I), hospitalization flow (I to H) and death flow (into D).
_Avoid_: flux

**Rates**:
The time-varying transmission, recovery, direct mortality, hospitalization, hospital recovery and hospital mortality rates that drive the latent state.

**Roll-forward**:
The step-by-step advance of the latent state under the rates, from an initial latent state.
_Avoid_: simulation, integration

**Reproduction number**:
The transmission rate times the susceptible fraction, divided by the total rate of removal from the Infected compartment.
_Avoid_: Rt (in prose)

### Targets and signals

**Target**:
An observed quantity that the model forecasts and is scored on: wastewater viral load, hospital admissions, reported cases or deaths.
_Avoid_: series, signal, proxy signal

**Wastewater viral load**:
The viral concentration measured in wastewater at a wastewater plant.
_Avoid_: ww, biomarker (in prose)

**Hospital admissions**:
The number of patients admitted to hospital, reported as weekly totals.
_Avoid_: hosp, hospitalizations

**Signal**:
An input stream that the model reads but is never scored on, such as mobility.
_Avoid_: feature stream, series

**Observation head**:
The part of the model that forecasts one target from the latent state and observation context.
_Avoid_: head (alone, in prose), output

**Observed point**:
A reported value of a target on one of its reporting days.

**Imputed point**:
A value filled into a gap in a target. It is never scored.

**Reporting cadence**:
The expected interval between observed points of a target: daily for reported cases and deaths, weekly for hospital admissions and wastewater viral load.

**Supervised**:
Describes a target in a window whose observed points in history and horizon meet the observed-fraction thresholds; only supervised targets are scored.
_Avoid_: eligible, active head

### Observation

**Delay kernel**:
The distribution of the delay between a latent quantity and its clinical observation.

**Shedding kernel**:
The distribution of viral shedding into wastewater after infection.

**Latent source**:
The latent quantity that an observation head reads through its kernel: infected prevalence or the infection, hospitalization or death flow.
_Avoid_: relevant source, observation source

**Mechanistic path**:
The route from the latent state through a kernel to a forecast.

**Mechanistic location**:
The part of a forecast location derived from the latent state through a kernel.

**Residual contribution**:
The part of a forecast location that the model adds from observation context for effects outside the SIRHD dynamics, such as reporting noise.

**Black-box model**:
A model whose observation heads all have the mechanistic path disabled, so every forecast is the residual contribution alone.
_Avoid_: residual-only network

**Ascertainment**:
The ratio of observed counts to the latent flow that produces them.

### Forecast window

**Origin**:
The last day of history; forecasts are made from it.
_Avoid_: forecast date, t0

**History**:
The days up to and including the origin that the model sees.
_Avoid_: input window, lookback

**Horizon**:
The forecast days after the origin.

**Window**:
One sample: a history and a horizon around one origin.
_Avoid_: forecast window (for the whole sample)

**Nowcast**:
The forecast of a target at the origin itself. It is not scored, but a jump between the nowcast and the first horizon day shows a discontinuity in forecast scale.

**Anchor**:
The last observed value of a target in the history. Forecast locations are expressed relative to it.
_Avoid_: delta, baseline

### Spatial data

**Region**:
The spatial unit of the forecast; in Catalonia, a municipality.
_Avoid_: node, municipality (in model prose)

**Target region**:
A region whose targets are forecast and scored.
_Avoid_: target node

**Contiguity**:
The relation between regions that share a border.

**Region embedding**:
A learned, static vector for each region that encodes its features, contiguity and mobility flows.
_Avoid_: geospatial embedding, node embedding

**Wastewater plant**:
A treatment plant whose catchment covers one or more regions.
_Avoid_: EDAR, plant (alone)

**Fold**:
One fixed group of target regions in the fold manifest.

**Fold manifest**:
The persisted assignment of every target region to exactly one fold.

**Role**:
The part a fold plays in one training: test, validation or train. One fold takes the test role, the next fold takes the validation role and the rest take the train role.
_Avoid_: split

**Temporal holdout**:
A held-out date range, named separately from any fold.
_Avoid_: split, test set (for a date range)

**Region set**:
A named, static subset of the fold manifest's regions, used to stratify evaluation.

### Datasets and scenarios

**Dataset**:
A processed, model-ready collection of scenarios, real or synthetic.
_Avoid_: corpus (for processed data)

**Corpus**:
The raw output of one version of the synthetic generator, before preprocessing.

**Scenario**:
One epidemic over all regions and dates of a dataset. A synthetic dataset holds many simulated scenarios; the real dataset holds one observed scenario.
_Avoid_: run, synthetic run, simulation

**Mobility profile**:
A base mobility series shared by several synthetic scenarios. Each scenario owns its observations, latent state and intervention-driven mobility reductions, and shares the base mobility of its profile.

**Reactive scenario**:
A synthetic scenario whose interventions respond to its own epidemic in a closed loop.

**No-policy scenario**:
The twin of a reactive scenario, on the same mobility profile, without interventions.
_Avoid_: baseline twin, baseline scenario

### Evaluation

**Selection metric**:
The validation score that chooses the checkpoint kept from training.

**Endpoint**:
The test score on which a campaign compares its conditions.
_Avoid_: primary metric

**Baseline model**:
A simple statistical forecaster, such as last-observed persistence, that sets a skill floor.

**Latent plausibility indicator**:
A ground-truth-free check of whether the inferred latent state is epidemiologically plausible.

**Latent information indicator**:
A ground-truth-free check of whether the latent quantity behind a forecast tracks the observations and drives the forecast.

### Experiments

**Campaign**:
A checked-in declaration of conditions, model seeds and folds that are trained and evaluated together.

**Condition**:
One configuration variant within a campaign.

**Reference condition**:
The condition of a campaign against which every other condition is compared.
_Avoid_: baseline, baseline condition

**Campaign cell**:
One combination of condition, model seed and fold within a campaign.
_Avoid_: identity

**Model seed**:
The seed of weight initialization and sampling in one training, paired across the conditions of a campaign.

**Ablation**:
A campaign whose conditions compare the presence and absence of one factor.
_Avoid_: experiment

**HPO study**:
A hyperparameter search. It is not a campaign.
_Avoid_: study (alone)

**Training**:
One fit of one model, from initial weights or a checkpoint to its final checkpoint.
_Avoid_: training run

**Pretraining**:
A training on a synthetic dataset.

**Fine-tuning**:
A training on the real dataset that starts from a pretraining's checkpoint.
_Avoid_: transfer (except as a campaign name)

**From-scratch training**:
A training on the real dataset from initial weights.

**Curriculum training**:
A training that samples synthetic and real scenarios together.

**Trial**:
One training and evaluation of a campaign cell or of an HPO study's sampled configuration. A campaign cell may need several trials when trials fail.
_Avoid_: attempt

### MN5 workflow

**Run**:
One versioned MN5 workflow run of one workflow kind, with its own run root, pinned source bundle and record of inputs and outputs.
_Avoid_: job, deployment

**Workflow kind**:
What a run does, fixed when the run is created.

**Train run**:
A run whose workflow kind is `train`; it holds one training.

**Stage**:
One step of a run's workflow.

**Execution**:
One Slurm allocation's pass over a run.
_Avoid_: execution attempt, attempt (alone), job

**Stage attempt**:
One try of one stage, such as preprocessing or one cross-validation fold.
_Avoid_: attempt (alone)

**Promotion**:
Making a run's output durable as versioned, content-addressed data outside the run.
_Avoid_: publish, publication

**Pinned input**:
A file from outside a run that the run copies in and identifies by content hash before any stage uses it.

**Init checkpoint**:
The pinned input from which a training starts.

**Login node**:
The MN5 host through which work is submitted.
_Avoid_: endpoint
