![1](picture\1.gif ) ![2](picture\2.gif )
# About

We propose the Enhance-NERF model, which can reconstruct outdoor scenes within an hour using a single graphics card. This MLP-based NERF method, built in PyTorch, enables fast model construction, expansion and deployment. In the tank and temple test, compared to other models, Enhance-NERF achieves better reconstruction of object materials and textures, effectively removing the fog-like noise in the background and providing clearer reconstruction results. Meanwhile, facing the uneven overexposure caused by one-sided light in outdoor scenes, Enhance-NERF averages the lighting intensity of the scene, making the boundary between foreground and background clearer.

To better simulate image capture scenarios, we create the Mix Sample dataset by simply shooting various scenes without deliberately planning the image capture scheme. Under this condition, despite the complex lighting behavior in dense scenes, we still succeed in restoring diffuse and specular reflections in the scene while eliminating chromatic fog-like interference. In synthesizing new perspectives, we achieve good restoration of object shape and distribution of ray lights in the scene. The main components of our model can be used interchangeably with other models to enhance the lighting performance of other NERF-based models.

![3](picture\2.png) 

Enhance-Nerf program using [nerfstudio](https://arxiv.org/abs/2302.04264)


# Quickstart


## 1. Installation: Setup the environment

### Prerequisites

You must have an NVIDIA video card with CUDA installed on the system. This library has been tested with version 11.3 of CUDA. You can find more information about installing CUDA [here](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)

### Create environment

Nerfstudio requires `python >= 3.7`. We recommend using conda to manage dependencies. Make sure to install [Conda](https://docs.conda.io/en/latest/miniconda.html) before proceeding.

```bash
conda create --name nerfstudio -y python=3.8
conda activate nerfstudio
python -m pip install --upgrade pip
```

### Dependencies

Install pytorch with CUDA (this repo has been tested with CUDA 11.3 and CUDA 11.7) and [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn)

For CUDA 11.3:

```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

For CUDA 11.7:

```bash
pip install torch torchvision functorch --extra-index-url https://download.pytorch.org/whl/cu117
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

See [Dependencies](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md#dependencies)  in the Installation documentation for more.

### Installing nerfstudio

Easy option:

```bash
pip install nerfstudio
```

**OR** if you want the latest and greatest:

```bash
git clone https://github.com/nerfstudio-project/nerfstudio.git
cd nerfstudio
pip install --upgrade pip setuptools
pip install -e .
```

**OR** if you want to skip all installation steps and directly start using nerfstudio, use the docker image:

See [Installation](https://github.com/nerfstudio-project/nerfstudio/blob/main/docs/quickstart/installation.md) - **Use docker image**.

### Train with Enhance-NeRF
```
ns-train sm --data XXX
```
sm means StitchMonster for we borrow many papers' part to enhance our model.

### Mix sample
comming soon



# Citation

You can find a paper writeup of the framework on [arXiv](https://arxiv.org/abs/2306.05303).

If you use this library or find the documentation useful for your research, please consider citing:

```
@article{Tan2023EnhanceNeRF,
  title={Enhance-NeRF: Multiple Performance Evaluation for Neural Radiance Fields},
  author={Qianqiu Tan and Tao Liu and Y. G. Xie and Shuwan Yu and Baohua Zhang},
  journal={arXiv preprint arXiv:2306.05303},
  year={2023}
}
```
