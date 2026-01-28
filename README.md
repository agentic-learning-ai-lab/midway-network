# Midway Network: Learning Representations for Recognition and Motion from Latent Dynamics
### [Paper](https://arxiv.org/abs/2510.05558) | [Website](https://agenticlearning.ai/midway-network)

This project hosts the code for implementing the Midway Network (ICLR 2026) architecture for self-supervised learning of visual representations for recognition and motion from videos.

> [**Midway Network: Learning Representations for Recognition and Motion from Latent Dynamics**](https://arxiv.org/abs/2510.05558)<br>
> [Christopher Hoang](https://www.chrishoang.com), [Mengye Ren](https://mengyeren.com)<br>
> International Conference on Learning Representations 2026<br>
> *arXiv ([arXiv 2510.05558](https://arxiv.org/abs/2510.05558))*

## Pretrained models

<table>
  <tr>
    <th colspan="1">data</th>
    <th colspan="1">backbone</th>
    <th colspan="1">epochs</th>
    <th colspan="2">download</th>
  </tr>
  <tr>
    <td>BDD100K</td>
    <td>300</td>
    <td>ViT-S</td>
    <td><a href="https://huggingface.co/agentic-learning-ai-lab/Midway-Network/blob/main/midway-bdd-vit-s-ep300.pth">full checkpoint</a></td>
    <td><a href="configs/exp/midway_bdd.yaml">configs</a></td>
  </tr>
  <tr>
    <td>BDD100K</td>
    <td>300</td>
    <td>ViT-B</td>
    <td><a href="https://huggingface.co/agentic-learning-ai-lab/Midway-Network/blob/main/midway-bdd-vit-b-ep300.pth">full checkpoint</a></td>
    <td><a href="configs/exp/midway_bdd_vit_base.yaml">configs</a></td>
  </tr>
  <tr>
    <td>Walking Tours Venice</td>
    <td>100</td>
    <td>ViT-S</td>
    <td><a href="https://huggingface.co/agentic-learning-ai-lab/Midway-Network/blob/main/midway-wt-venice-vit-s-ep100.pth">full checkpoint</a></td>
    <td><a href="configs/exp/midway_wt_venice.yaml">configs</a></td>
  </tr>
</table>

## Code Structure

```
.
├── configs                   # directory in which all experiment '.yaml' configs are stored
├── src                       # the package
│   ├── main.py               #   main training loop for midway network
│   ├── midway.py             #   model definition
│   ├── utils.py              #   shared utilities
│   ├── vision_transformer.py #   encoder definition
│   └── datasets              #   datasets, data loaders
└── main.py                   # entrypoint to launch Midway Network pre-training locally or on SLURM cluster
```

**Config files:**
Note that all experiment parameters are specified in config files (as opposed to command-line-arguments). See the [configs/exp](configs/exp) directory for example config files.

## Launching Midway Network pre-training
[submit.py](submit.py) is an entrypoint script for launching experiments with [submitit](https://github.com/facebookincubator/submitit) and [hydra](https://hydra.cc).
The actual implementation is in [src/main.py](src/main.py), which parses the experiment config file and runs Midway Network pre-training.

### Training
Here is an example of how to run Midway Network ViT-S WT-Venice pre-training on a local, 2 GPU machine with config [configs/exp/midway_wt_venice.yaml](configs/exp/midway_wt_venice.yaml):

```
torchrun --standalone --nnodes=1 --nproc-per-node=2 submit.py \
    compute=local \
    exp=midway_bdd \
    name='midway-wt-venice-local'
```

Here is an example of how to run Midway Network ViT-B BDD pre-training on a SLURM cluster with 2 GPUs with config [configs/exp/midway_bdd_vit_base.yaml](configs/exp/midway_bdd_vit_base.yaml):
```
python submit.py \
    compute/greene=2x1 compute/greene/node=ah \
    compute.timeout=1700 \
    compute.cpus_per_task=20 \
    exp=midway_bdd_vit_base \
    name='midway-bdd-vit-b-slurm'
```

See the [scripts](scripts) directory for example scripts.
Note: Use [scripts/decode_walking_tours.py](scripts/decode_walking_tours.py) to extract a Walking Tours (or any other) video into PNG frames so that it is compatiable with the [decoded_walking_tours](src/datasets/wt.py) dataloader. 

## Evaluation

We use [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) to evaluate on semantic segmentation and the evaluation setup in [CroCo v2](https://github.com/naver/croco) to evaluate on optical flow.

---

### Requirements
* Python 3.10 (or newer)
* PyTorch 2.2.0
* torchvision 0.17.1 ([build from source, for video_reader](https://github.com/pytorch/vision?tab=readme-ov-file#unstable-video-backend))
* ffmpeg 5.1.2 (from conda-forge, for video_reader)
* Other dependencies: decord, ffprobe-python, flow-vis, hydra-core, numpy, scipy, timm==0.3.2, wandb

Importing this version of `timm` will raise an import error, see [here](https://github.com/huggingface/pytorch-image-models/issues/420) for a fix.

We provide an example [environment.yaml](environment.yaml) file.

--- 

## License
See the [LICENSE](./LICENSE) file for details about the license under which this code is made available.

## Citation
If you find this repository useful in your research, please consider giving a star :star: and a citation:
```
@inproceedings{hoang:2026:midway-network,
  title={Midway Network: Learning Representations for Recognition and Motion from Latent Dynamics}, 
    author={Chris Hoang and Mengye Ren},
  booktitle={International Conference on Learning Representations},  
  year={2026}
}
```