## Automated Robocall Campaign Detection using Audio Embeddings

This repository contains a Proof of Concept (PoC) of uncovering robocall campaigns from raw robocall recordings based on audio similarity. The example code demonstrate the following:

- How to compute audio embeddings using two pre-trained (and fine-tuned) models using Wav2Vec2 and WavLM (on CPU and GPU)
- How to aggregate the embeddings into robocall campaigns

## Dataset Details

To demonstrate this code, the dataset from **Robocall Audio from the FTC’s Project Point of No Entry** ([GitHub link](https://github.com/wspr-ncsu/robocall-audio-dataset)) is used.


## How to run this example?

- Extract the raw audio recordings in `FTC-raw-audio-ppone-normalized.zip` ([Google Drive link](https://drive.google.com/file/d/1ku__-U2VQleHfeuxA06b06cqP9w16ha-/view?usp=sharing)) or download audio files from [robocall-audio-dataset](https://github.com/wspr-ncsu/robocall-audio-dataset).
- Install the relevant dependencies
- Run the example code `Robocall_Campaign_Detection_GPU_and_CPU.py`

## Questions?

The example code is part of the paper titled "*Characterizing Robocalls with Multiple Vantage Points*". The paper was published at the IEEE Security & Privacy 2025 conference.

Please refer to the paper for additional details (evaluation, scaling, etc). If you found this artifact useful, please cite the paper!
