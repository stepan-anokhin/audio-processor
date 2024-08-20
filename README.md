# Audio Transformers

[![Tests](https://github.com/stepan-anokhin/audio-transformers/actions/workflows/tests.yaml/badge.svg)](https://github.com/stepan-anokhin/audio-transformers/actions/workflows/tests.yaml)
[![Coverage Status](https://coveralls.io/repos/github/stepan-anokhin/audio-transformers/badge.svg?branch=master)](https://coveralls.io/github/stepan-anokhin/audio-transformers?branch=master)
[![Licence: MIT](https://img.shields.io/pypi/l/audio-transformers)](https://github.com/stepan-anokhin/audio-transformers/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/audio-transformers.svg?style=flat)](https://pypi.org/project/audio-transformers/)
![Python version support](https://img.shields.io/pypi/pyversions/audio-transformers)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/ambv/black)

A python library for audio signals transformations.

## Setup

### Prerequisites

Requires `ffmpeg`, and `Python >= 3.10`

### Installation

```shell
pip install audio-transformers
```

## Command-Line Interface

### List Available Transformations

Run:

```shell
audio transform list
```

Output:

```
Name               Description
-----------------  --------------------------------------------------
BandPass           Apply band-pass filter.
BandStop           Apply band-stop filter.
GaussianNoise      Add gaussian noise to the signal.
HighPass           Apply high-pass filter.
Inversion          Inverse waveform polarity by multiplying it by -1.
LowPass            Apply low-pass filter.
PitchShift         Pitch shift transformation.
SpeedPerturbation  Speed perturbation transformer.
```

### Show Transformation Parameters

Run:

```shell
audio transform params TRANSFORMATION
```

For example:

```shell
audio transform params PitchShift
```

Output:

```
Name             Type    Default    Description
---------------  ------  ---------  --------------------------------------
shift            float              Pitch shift in octaves.
fft_window_size  float   0.1        Short Time FFT window size in seconds.
```

### Transform Audio File

Command format variants:

```shell
audio transform file INPUT_PATH OUTPUT_PATH TRANSFORMATION *OPTIONS
audio transform file INPUT_PATH OUTPUT_PATH --config=CONFIG_PATH
```

For example to specify transformation via CLI args run:

```shell
audio transform file path/to/input.opus path/to/output.wav PitchShift --shift=0.5
```

Output:

```
2024-08-20 18:07:24,897 INFO Processing file path/to/input.opus -> path/to/output.wav
  5%|█████              | 21.1M/453M [00:00<00:11, 36.4Msamples/s] 
```

Otherwise, you can specify transformation in a config file.
For example if `task.yaml` contains the following definitions:

```yaml
transforms:
  - type: PitchShift
    params:
      shift: 0.2
  - type: SpeedPerturbation
    params:
      speed_factor: 0.5
```

You can run:

```shell
audio transform file path/to/input.opus path/to/output.wav --config=task.yaml
```

The `output.wav` will have pitch shifted by `+0.2` octaves relative to `input.opus`
and will be stretched twice (with no additional significant pitch perturbations).

### Transform Dataset

Command format:

```shell
audio transform files --config=FILE
```

Config will have additional attributes:

```yaml
input_root: "path/to/INPUT/data/root"
input_pattern: "**/*.opus"
output_root: "path/to/OUTPUT/data/root"
output_pattern: "{reldir}/{name}.opus"
transforms:
  - type: PitchShift
    params:
      shift: 0.5
  - type: SpeedPerturbation
    params:
      speed_factor: 0.5
```

* `input_root` is a root directory for input dataset
* `input_pattern` is input file path pattern relative to the `input_root`
* `output_root` is a root directory for output files
* `output_pattern` output file pattern relative to the output root. It will be recalculated for each input file. You can
  use curly braces `{something}` to substitute the corresponding input file path elements. The following elements are
  supported:
    * `{relpath}` - full input path relative to the input root
    * `{reldir}` - input file directory relative to the input root
    * `{name}` - input file name without extension
    * `{ext}` - input file extension

### Public Datasets

The `audio` tool supports downloading public STT datasets for testing purpose.

#### Listing Public Datasets

Run:

```shell
audio datasets list
```

Output:

```
Name                                   Format    Size      Archive Size
-------------------------------------  --------  --------  --------------
radio_v4_and_public_speech_5percent    opus      65.8 GB   11.4 GB
audiobook_2                            opus      162.0 GB  25.8 GB
radio_2                                opus      154.0 GB  24.6 GB
public_youtube1120                     opus      237.0 GB  19.0 GB
asr_public_phone_calls_2               opus      66.0 GB   9.4 GB
public_youtube1120_hq                  opus      31.0 GB   4.9 GB
asr_public_stories_2                   opus      9.0 GB    1.4 GB
tts_russian_addresses_rhvoice_4voices  opus      80.9 GB   12.9 GB
public_youtube700                      opus      75.0 GB   12.2 GB
asr_public_phone_calls_1               opus      22.7 GB   3.2 GB
```

#### Download Public Datasets

Run for example:

```shell
audio datasets download public_lecture_1 data/lecture_dataset
```

Output (intermediate):

```yaml
2024-08-20 18:27:23,344 INFO     Downloading dataset 'public_lecture_1' (122.5 MB) to data/lecture_dataset
90%|████████████████████████████████   | 110M/123M [00:43<00:15, 3.4Mbytes/s]
```

## Development

The project requires [Poetry](https://python-poetry.org/) and `Python >= 3.10`

Clone:

```shell
git clone git@github.com:stepan-anokhin/audio-transformers.git
```

Then:

```shell
cd audio-transformers
```

Install dependencies:

```shell
poetry install
```

Run tests:

```shell
poetry run pytest
```

The project uses [Black](https://pypi.org/project/black/) code style. Run style check:

```shell
poetry run black --check --line-length 120 audio_transformers tests
```

Run linter:

```shell
poetry run flake8 audio_transformers tests --count --max-complexity=10 --max-line-length=120 --statistics
```

### Project Structure

Packages:

* `audio_transformers/core` - implementations audio transformations
* `audio_transformers/io` - input/output logic (using `ffmpegio`)
* `audio_transformers/cli` - CLI tool implementation
* `audio_transformers/cli/handlers` - CLI subcommand handlers
* `audio_transformers/utils` - misc utilities
* `tests` - unit-tests and integration tests
