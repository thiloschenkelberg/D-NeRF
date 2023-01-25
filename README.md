# [[D-NeRF]](https://github.com/albertpumarola/D-NeRF) with [[Tiny Cuda Neural Network]](https://github.com/NVlabs/tiny-cuda-nn) and [[Multiresolution Hash Encoding]](https://nvlabs.github.io/instant-ngp/).

<img src='https://www.albertpumarola.com/images/2021/D-NeRF/teaser2.gif' align="right" width=400>

## D-NeRF: Neural Radiance Fields for Dynamic Scenes
#### [[Project]](https://www.albertpumarola.com/research/D-NeRF/index.html)[ [Paper]](https://openaccess.thecvf.com/content/CVPR2021/papers/Pumarola_D-NeRF_Neural_Radiance_Fields_for_Dynamic_Scenes_CVPR_2021_paper.pdf) 

[D-NeRF](https://www.albertpumarola.com/research/D-NeRF/index.html) is a method for synthesizing novel views, at an arbitrary point in time, of dynamic scenes with complex non-rigid geometries. We optimize an underlying deformable volumetric function from a sparse set of input monocular views without the need of ground-truth geometry nor multi-view images.

This project is an extension of [NeRF](http://www.matthewtancik.com/nerf) enabling it to model dynmaic scenes. The code heavily relays on [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). 

![D-NeRF](https://www.albertpumarola.com/images/2021/D-NeRF/model.png)

### Installation
```
git clone https://github.com/albertpumarola/D-NeRF.git
cd D-NeRF
conda create -n dnerf python=3.6
conda activate dnerf
pip install -r requirements.txt
cd torchsearchsorted
pip install .
cd ..
```

### Download Pre-trained Weights
 You can download the pre-trained models from [drive](https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/25sveotbx2x7wap/logs.zip?dl=0). Unzip the downloaded data to the project root dir in order to test it later. See the following directory structure for an example:
```
├── logs 
│   ├── mutant
│   ├── standup 
│   ├── ...
```

### Download Dataset
 You can download the datasets from [drive](https://drive.google.com/file/d/19Na95wk0uikquivC7uKWVqllmTx-mBHt/view?usp=sharing) or [dropbox](https://www.dropbox.com/s/0bf6fl0ye2vz3vr/data.zip?dl=0). Unzip the downloaded data to the project root dir in order to train. See the following directory structure for an example:
```
├── data 
│   ├── mutant
│   ├── standup 
│   ├── ...
```

### Demo
We provide simple jupyter notebooks to explore the model. To use them first download the pre-trained weights and dataset.

| Description      | Jupyter Notebook |
| ----------- | ----------- |
| Synthesize novel views at an arbitrary point in time. | render.ipynb|
| Reconstruct mesh at an arbitrary point in time. | reconstruct.ipynb|
| Quantitatively evaluate trained model. | metrics.ipynb|

### Test
First download pre-trained weights and dataset. Then, 
```
python run_dnerf.py --config configs/mutant.txt --render_only --render_test
```
This command will run the `mutant` experiment. When finished, results are saved to `./logs/mutant/renderonly_test_799999` To quantitatively evaluate model run `metrics.ipynb` notebook

### Train
First download the dataset. Then,
```
conda activate dnerf
export PYTHONPATH='path/to/D-NeRF'
export CUDA_VISIBLE_DEVICES=0
python run_dnerf.py --config configs/mutant.txt
```

### Citation
If you use this code or ideas from the paper for your research, please cite our paper:
```
@article{pumarola2020d,
  title={D-NeRF: Neural Radiance Fields for Dynamic Scenes},
  author={Pumarola, Albert and Corona, Enric and Pons-Moll, Gerard and Moreno-Noguer, Francesc},
  journal={arXiv preprint arXiv:2011.13961},
  year={2020}
}
```

## Tiny CUDA Neural Networks ![](https://github.com/NVlabs/tiny-cuda-nn/workflows/CI/badge.svg)

This is a small, self-contained framework for training and querying neural networks. Most notably, it contains a lightning fast ["fully fused" multi-layer perceptron](https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/fully-fused-mlp-diagram.png) ([technical paper](https://tom94.net/data/publications/mueller21realtime/mueller21realtime.pdf)), a versatile [multiresolution hash encoding](https://raw.githubusercontent.com/NVlabs/tiny-cuda-nn/master/data/readme/multiresolution-hash-encoding-diagram.png) ([technical paper](https://nvlabs.github.io/instant-ngp/assets/mueller2022instant.pdf)), as well as support for various other input encodings, losses, and optimizers.

### Usage

Tiny CUDA neural networks have a simple C++/CUDA API:

```cpp
#include <tiny-cuda-nn/common.h>

// Configure the model
nlohmann::json config = {
	{"loss", {
		{"otype", "L2"}
	}},
	{"optimizer", {
		{"otype", "Adam"},
		{"learning_rate", 1e-3},
	}},
	{"encoding", {
		{"otype", "HashGrid"},
		{"n_levels", 16},
		{"n_features_per_level", 2},
		{"log2_hashmap_size", 19},
		{"base_resolution", 16},
		{"per_level_scale", 2.0},
	}},
	{"network", {
		{"otype", "FullyFusedMLP"},
		{"activation", "ReLU"},
		{"output_activation", "None"},
		{"n_neurons", 64},
		{"n_hidden_layers", 2},
	}},
};

using namespace tcnn;

auto model = create_from_config(n_input_dims, n_output_dims, config);

// Train the model
GPUMatrix<float> training_batch_inputs(n_input_dims, batch_size);
GPUMatrix<float> training_batch_targets(n_output_dims, batch_size);

for (int i = 0; i < n_training_steps; ++i) {
	generate_training_batch(&training_batch_inputs, &training_batch_targets); // <-- your code

	float loss;
	model.trainer->training_step(training_batch_inputs, training_batch_targets, &loss);
	std::cout << "iteration=" << i << " loss=" << loss << std::endl;
}

// Use the model
GPUMatrix<float> inference_inputs(n_input_dims, batch_size);
generate_inputs(&inference_inputs); // <-- your code

GPUMatrix<float> inference_outputs(n_output_dims, batch_size);
model.network->inference(inference_inputs, inference_outputs);
```

### License and Citation

This framework is licensed under the BSD 3-clause license. Please see `LICENSE.txt` for details.

If you use it in your research, we would appreciate a citation via
```bibtex
@misc{tiny-cuda-nn,
    Author = {Thomas M\"uller},
    Year = {2021},
    Note = {https://github.com/nvlabs/tiny-cuda-nn},
    Title = {Tiny {CUDA} Neural Network Framework}
}
```
