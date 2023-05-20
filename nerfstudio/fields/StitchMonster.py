"Filed  of mine"
"""
基础的idea：
在原有的basemodel前加入固定好的参数
加入一些插件让nerf变干净


解决的问题：
1。模型的撰写，cleannerf影响下的模型，nerfw+niceslam
2。体素导出工具，AI奎的nerf-pl的vol导出模型

"""
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from torch.nn import init

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    LinearHead,
    PredNormalsFieldHead,
    RGBFieldHead,
    RGBTUNEHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP, conv_onet
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn


class StitchMonster(Field):

    """MYNeRF Field

        1,更大
        2.更干净
        3.更快
        3.appearance embedding 编码每一张图，代表每一张图
        4.参考nice-slam的前段部分：不希望每一次都更新全部

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        base_mlp_num_layers: Number of layers for base MLP.
        base_mlp_layer_width: Width of base MLP layers.
        head_mlp_num_layers: Number of layer for output head MLP.
        head_mlp_layer_width: Width of output head MLP layers.

        position_encoding: Position encoder.
        direction_encoding: Direction encoder.

        skip_connections: Where to add skip connection in base MLP.
        use_integrated_encoding: Used integrated samples as encoding input.
        spatial_distortion: Spatial distortion.

    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        # 等待
        base_mlp_num_layers: int = 3,
        base_mlp_num_width: int = 64,
        head_mlp_num_layers: int = 2,
        head_mlp_layer_width: int = 32,
        position_encoding: Encoding = HashEncoding(),
        direction_encoding: Encoding = SHEncoding(),
        skip_connections: Tuple[int] = (4,),
        appearance_embedding_dim: int = 32,
        spatial_distortion: SpatialDistortion = SceneContraction(order=float("inf")),
        use_average_appearance_embedding: bool = False,
        num_nerf_samples_per_ray: int = 48,
    ) -> None:
        super().__init__()
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(
            self.num_images, self.appearance_embedding_dim
        )
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.position_encoding = position_encoding
        self.direction_encoding = direction_encoding
        self.geo_feat_dim = self.direction_encoding.get_out_dim()
        self.use_average_appearance_embedding = use_average_appearance_embedding

        ##三种mlp模型
        self.shared_decoders_position = conv_onet(
            in_dim=32,
            hidden_size=32,
            n_blocks=5,
            skips=[2],
            out_activation=nn.ReLU(),
        )
        self.shared_decoders_dirctions = conv_onet(
            in_dim=93,
            hidden_size=32,
            n_blocks=5,
            skips=[2],
            out_activation=nn.ReLU(),
        )

        self.mlp_base = MLP(
            in_dim=self.shared_decoders_position.get_out_dim(),
            num_layers=base_mlp_num_layers,
            layer_width=base_mlp_num_width,
            skip_connections=skip_connections,
            out_activation=nn.ReLU(),
        )
        """
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim()
            + self.embedding_appearance.get_out_dim()
            + 16,
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.Sigmoid(),
            out_dim=3,
        )
        """
        self.mlp_head = MLP(
            in_dim=self.mlp_base.get_out_dim()
            + self.embedding_appearance.get_out_dim()
            + self.shared_decoders_position.get_out_dim(),
            num_layers=head_mlp_num_layers,
            layer_width=head_mlp_layer_width,
            out_activation=nn.Sigmoid(),
        )

        # self.load_pretrain()
        # set heads
        self.field_output_density = DensityFieldHead(
            in_dim=self.mlp_base.get_out_dim(), activation=trunc_exp
        )
        self.field_output_RGB = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())
        self.field_output_RGB_withoutDir = RGBFieldHead(
            in_dim=self.mlp_base.get_out_dim()
        )
        self.field_output_RGB_withDir = RGBFieldHead(in_dim=self.mlp_head.get_out_dim())
        self.fied_RGBTUNE = RGBTUNEHead(in_dim=self.mlp_base.get_out_dim())
        self.fied_RGBTUNE2 = RGBTUNEHead(in_dim=self.mlp_head.get_out_dim())
        self.Linear = LinearHead(
            in_dim=self.direction_encoding.get_out_dim(), out_dim=93
        )
        # self._position = LinearHead(in_dim=self.direction_encoding.get_out_dim(), out_dim=32)
        # self._dirctions = LinearHead(in_dim=93, out_dim=32)
        # TODO 改成双线性差值

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb
            )

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]

        encoded_xyz = self.position_encoding(positions)
        # 中间有一段高频输入的固定格式

        encoded_xyz = self.shared_decoders_position(encoded_xyz)
        base_mlp_out = self.mlp_base(encoded_xyz)

        density = self.field_output_density(base_mlp_out)
        """
        test = density
        test.view(4096, self.num_nerf_samples_per_ray)
        print(test.shape)
        """
        # TODO 让density更加的clean

        return density, base_mlp_out

    def clean_density(self, density):
        return density

    def load_pretrain(
        self,
    ):
        """
        Load parameters of pretrained ConvOnet checkpoints to the decoders.
        """
        ckpt = torch.load("nerfstudio/nerfstudio/field_components/coarse.pt")
        ckpt2 = torch.load("nerfstudio/nerfstudio/field_components/middle_fine.pt")
        coarse_dict = {}
        fine_dict = {}
        for key, val in ckpt["model"].items():
            if ("decoder" in key) and ("encoder" not in key) and ("output" not in key):
                key = key[8:]
                coarse_dict[key] = val

        for key, val in ckpt2["model"].items():
            if ("decoder" in key) and ("encoder" not in key):
                if "fine" in key:
                    key = key[8 + 5 :]
                    fine_dict[key] = val

        self.shared_decoders_dirctions.load_state_dict(fine_dict, strict=False)
        self.shared_decoders_position.load_state_dict(coarse_dict, strict=False)

        for name, value in self.shared_decoders_position.named_parameters():
            value.requires_grad = False

        for name, value in self.shared_decoders_dirctions.named_parameters():
            value.requires_grad = False

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()

        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)

        encoded_dir = self.direction_encoding(directions_flat)  # 纬度 16
        encoded_dir = self.Linear(encoded_dir)
        encoded_dir = self.shared_decoders_dirctions(encoded_dir)

        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim),
                    device=directions.device,
                )

        outputs = {}

        tensor = torch.cat(
            [
                encoded_dir,
                density_embedding.view(-1, self.mlp_base.get_out_dim()),  # type:ignore
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,  # type:ignore
        )

        # RGBwithDir = self.mlp_head(tensor)

        # TODO
        tensor = self.mlp_head(tensor)
        RGBwithDir = self.field_output_RGB_withDir(
            tensor.view(-1, self.mlp_head.get_out_dim())
        )
        RGBTune2 = self.fied_RGBTUNE2(tensor.view(-1, self.mlp_head.get_out_dim()))

        RGBwithoutDir = self.field_output_RGB_withoutDir(
            density_embedding.view(-1, self.mlp_base.get_out_dim())
        )
        RGBTune = self.fied_RGBTUNE(
            density_embedding.view(-1, self.mlp_base.get_out_dim())
        )

        # TODO
        thefinal1 = (1 - RGBTune) * RGBwithDir + (RGBTune * RGBwithoutDir)
        thefinal2 = (1 - RGBTune2) * RGBwithDir + (RGBTune2 * RGBwithoutDir)
        thefinal = (thefinal1 + thefinal2) / 2

        outputs[FieldHeadNames.RGB] = thefinal1.view(*outputs_shape, -1).to(directions)
        # outputs[FieldHeadNames.RGB] = RGBwithDir.view(*outputs_shape, -1).to(directions)
        outputs[FieldHeadNames.RGB0] = RGBwithDir.view(*outputs_shape, -1).to(
            directions
        )
        outputs[FieldHeadNames.RGB1] = RGBwithoutDir.view(*outputs_shape, -1).to(
            directions
        )

        return outputs


if __name__ == "__main__":
    cd = StitchMonster(
        aabb=SceneBox.aabb,
        num_images=48,
    )
