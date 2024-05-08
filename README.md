# MetaVoice-LoRA

This repo is heavily based on https://github.com/metavoiceio/metavoice-src
## What We Do
1. We add LoRA to the existing finetuning pipeline and do some experiments to verify the effectiveness of LoRA.
2. We use whisperx to preprocess the audio files to generate the training data.
3. We use wespeaker to evaluate the similarity between the generated audio and the original audio.

**Directory Structure**
|- assets/
|- data	/	audio files and captions
|- datasets/	path to audio files and captions in cs
|- fam/
	|- llm/
		|- adapters/		flattened interleaved decoding
		|- config/			finetune config
		|- layers/			layer and block definition
		|- loaders/			process audio and text data
		|- mixins/			mixin class for inference
		|- preprocessing/	pad and interleave tokens
	|- quantiser/	speaker information embedding
	|- telemetry	/	monitor
|- tests/	e2e testing

|- eval.py	compare scores
|- makeDataset.py	generate data with  whisperx
```

## Installation

**Pre-requisites:**
- GPU VRAM >=12GB
- Python >=3.10,<3.12
- pipx ([installation instructions](https://pipx.pypa.io/stable/installation/))

**Environment setup**
```bash
# install ffmpeg
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz
wget https://johnvansickle.com/ffmpeg/builds/ffmpeg-git-amd64-static.tar.xz.md5
md5sum -c ffmpeg-git-amd64-static.tar.xz.md5
tar xvf ffmpeg-git-amd64-static.tar.xz
sudo mv ffmpeg-git-*-static/ffprobe ffmpeg-git-*-static/ffmpeg /usr/local/bin/
rm -rf ffmpeg-git-*

# install rust if not installed (ensure you've restarted your terminal after installation)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Project dependencies installation
1. [Using poetry](#using-poetry-recommended)
2. [Using pip/conda](#using-pipconda)

#### Using poetry (recommended)
```bash
# install poetry if not installed (ensure you've restarted your terminal after installation)
pipx install poetry

# disable any conda envs that might interfere with poetry's venv
conda deactivate

# if running from Linux, keyring backend can hang on `poetry install`. This prevents that.
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring

# pip's dependency resolver will complain, this is temporary expected behaviour
# full inference & finetuning functionality will still be available
poetry install && poetry run pip install torch==2.2.1 torchaudio==2.2.1 peft
```

#### Using pip/conda
NOTE 1: When raising issues, we'll ask you to try with poetry first.
NOTE 2: All commands in this README use `poetry` by default, so you can just remove any `poetry run`.

```bash
pip install -r requirements.txt
pip install torch==2.2.1 torchaudio==2.2.1 peft
pip install -e .
```

## Finetuning

In order to finetune, we expect a "|"-delimited CSV dataset of the following format:

```csv
audio_files|captions
./data/audio.wav|./data/caption.txt
```

We need whisperx to preprocess the audio files. You can install it according to the instructions [here](https://github.com/m-bain/whisperX).
Then you can preprocess your audio files via:
```bash
whisperx "example.wav" --model small --align_model WAV2VEC2_ASR_LARGE_LV60K_960H --batch_size 4
```
Now, we can use scripts to convert the preprocessed audio files to the format we need for finetuning:
You need to modify the path in the script to point to the directory where the preprocessed audio files are stored.
```python
wav_file_path = 'path/to/preprocessed/audio/files'
srt_file_path = 'path/to/srt/files'
output_dir = 'path/to/output/directory'
csv_file_path = 'path/to/csv/file'
```
```bash
python makeDataset.py
```

Try it out using our sample datasets via:
```bash
poetry run finetune --train path/to/train/csv/file --val ./path/to/val/csv/file
```

Once you've trained your model, you can use it for inference via:
```bash
poetry run python -i fam/llm/fast_inference.py --first_stage_path ./lora.pt
```

We can evaluate the similarity between the generated audio and the original audio using wespeaker:
```bash
python eval.py
```

## Results
We have conducted experiments on the Trump dataset. The results are shown in the following table:

Metavoice (zero-shot): 0.872

XTTS2 (zero-shot): 0.593

StyleTTS2 (zero-shot): 0.565

OpenVoice (zero-shot): 0.920

XTTS2 (fine-tune): 0.888

Metavoice (last layer finetune): 0.875

Metavoice (LoRA): 0.890

Metavoice (fine-tune): 0.892


Compared to full-parameter finetuning, LoRA can achieve similar performance with fewer parameters. And LoRA can save a lot of disk space:
We set rank = 10 and alpha = 10 in lora. The results are shown in the following table:
```
trainable params: 1,966,080 || all params: 1,245,157,376 || trainable%: 0.15789811295307302

Metavoice (full-parameter finetuning): 4.7G
Metavoice (LoRA): 7.9M
```
