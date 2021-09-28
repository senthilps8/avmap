# Audio-Visual Floorplan Reconstruction

![AV-Map Model](http://www.cs.cmu.edu/~spurushw/publication/avmap/teaser.png)

This is the code accompanying the work:  

Audio-Visual Floorplan Reconstruction<br/>
*Senthil Purushwalkam, Sebastian Vicenc Amengual Gari, Vamsi Krishna Ithapu, Carl Schissler, Philip Robinson, Abhinav Gupta, Kristen Grauman<br/>
arXiv preprint arXiv:2012.15470 (2020) <br/>*
[Webpage](http://www.cs.cmu.edu/~spurushw/projects/avmap.html) | [Paper](https://arxiv.org/abs/2012.15470) <br/>

## Cite

If you find this repository useful in your own research, please consider 
citing both papers:

```
@article{purushwalkam2020audio,
  title={Audio-visual floorplan reconstruction},
  author={Purushwalkam, Senthil and Gari, Sebastian Vicenc Amengual and Ithapu, Vamsi Krishna and Schissler, Carl and Robinson, Philip and Gupta, Abhinav and Grauman, Kristen},
  journal={arXiv preprint arXiv:2012.15470},
  year={2020}
}

@inproceedings{chen2020soundspaces,
    title = {SoundSpaces: Audio-Visual Navigation in 3D Environments},
    author = {Chen, Changan and Jain, Unnat and Schissler, Carl and Gari, Sebastia Vicenc Amengual and Al-Halah, Ziad and Ithapu, Vamsi Krishna and Robinson, Philip and Grauman, Kristen},
    year = {2020},
    booktitle={ECCV},
}
```


## Prerequisites
The code has been tested using Python v3.7.9, PyTorch v1.7.1 and 
Hydra v1.0.4.

Choose a project directory (`$PROJ_DIR`) where you plan to store all the data.

### Downloading the data
We release the set of rendered images and audio clips that were 
used to train and test our models.
This data can be downloaded from [here](https://drive.google.com/file/d/1gUNU_NgqUbWMyhGIRhsK_BYftSiF0kZE/view?usp=sharing).
Untar this data in `$PROJ_DIR`.

### Download the SoundSpaces dataset
We use the ambisonic impulse response data for Matterport3D released in 
the [SoundSpaces dataset](https://soundspaces.org/). Follow this link and 
download the dataset to the same `$PROJ_DIR` directory chosen above.


### Setup
Edit the `project_dir` entry in `configs/avmap/environment/default.yaml` to 
point to the `$PROJ_DIR` containing the above downloaded datasets.

## Training a model

### Device Generated Audio Setting

```bash
# RGB + Audio Model 
PYTHONPATH=. python main.py logging.name=devgen_rgba  model.rgb_model.use_model=True model.audio_model.use_model=True data/audio_clip=freq_sweep_signal data.source_at_receiver=True
# RGB Only Model 
PYTHONPATH=. python main.py logging.name=devgen_rgb  model.rgb_model.use_model=True model.audio_model.use_model=False data/audio_clip=freq_sweep_signal data.source_at_receiver=True
# Audio Only Model 
PYTHONPATH=. python main.py logging.name=devgen_a  model.rgb_model.use_model=False model.audio_model.use_model=True data/audio_clip=freq_sweep_signal data.source_at_receiver=True
```

### Environment Generated Audio Setting

```bash
# Set data.n_sources > 20 for all room setting
# RGB + Audio Model 
PYTHONPATH=. python main.py logging.name=envgen_rgba  model.rgb_model.use_model=True model.audio_model.use_model=True data/audio_clip=env_gen data.source_at_receiver=False data.n_sources=100
# Audio Only Model 
PYTHONPATH=. python main.py logging.name=envgen_a  model.rgb_model.use_model=False model.audio_model.use_model=True data/audio_clip=env_gen data.source_at_receiver=False  data.n_sources=100
```

## Testing a model

Models can be tested using the same commands as above by 
appending `environment.evaluate_path=<path_to_checkpoint> model.pool_steps=True`
 at the end of the corresponding training command. 



