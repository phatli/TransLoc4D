import torch
import torch.nn as nn
from MinkowskiEngine.modules.resnet_block import BasicBlock, Bottleneck
from os.path import isfile


from .minkloc import MinkLoc
from .ptcnet import PTC_Net, PTC_Net_L
from .layers.eca_block import ECABasicBlock
from .layers.pooling_wrapper import PoolingWrapper
from .minkfpn import MinkFPN
from .transloc3dfpn import TransLoc3DFPN

def get_model(params, device, weights_path=None):
    if params.model_params.input_representation == "RVI":
        in_channels = 2
    else:
        in_channels = 1
    if params.model_params.model == 'PTC-Net':
        model = PTC_Net(params.model_params)
    elif params.model_params.model == 'PTC-Net-L':
        model = PTC_Net_L(params.model_params)
    elif params.model_params.model == 'MinkLoc':
        block_module = create_resnet_block(params.model_params.block)
        backbone = MinkFPN(
            in_channels=in_channels,
            out_channels=params.model_params.feature_size,
            num_top_down=params.model_params.num_top_down,
            conv0_kernel_size=params.model_params.conv0_kernel_size,
            block=block_module,
            layers=params.model_params.layers,
            planes=params.model_params.planes,
        )
        vlad_init = False
        pooling = PoolingWrapper(
            pool_method=params.model_params.pooling,
            in_dim=params.model_params.feature_size,
            output_dim=params.model_params.output_dim,
            vlad_init=vlad_init,
            num_clusters=int(params.model_params.num_clusters),
        )
        model = MinkLoc(
            backbone=backbone,
            pooling=pooling,
            normalize_embeddings=params.model_params.normalize_embeddings,
            self_att=params.model_params.self_att,
            add_FTU=params.model_params.add_FTU,
        )
    elif params.model_params.model == "TransLoc3d":
        block_module = create_resnet_block(params.model_params.block)
        backbone = TransLoc3DFPN(
            in_channels=in_channels,
            conv0_out_channels=params.model_params.conv0_out_channels,
            conv1_out_channels=params.model_params.conv1_out_channels,
            conv0_kernel_size=params.model_params.conv0_kernel_size,
            conv0_stride=params.model_params.conv0_stride,
            conv1_kernel_size=params.model_params.conv1_kernel_size,
            conv1_stride=params.model_params.conv1_stride,
            num_attn_layers=params.model_params.num_attn_layers,
            global_channels=params.model_params.global_channels,
            local_channels=params.model_params.local_channels,
            num_centers=params.model_params.num_centers,
            num_heads=params.model_params.num_heads,
            out_channels=params.model_params.out_channels,
        )
        vlad_init = False
        pooling = PoolingWrapper(
            pool_method=params.model_params.pooling,
            in_dim=params.model_params.feature_size,
            output_dim=params.model_params.output_dim,
            vlad_init=vlad_init,
            num_clusters=int(params.model_params.num_clusters),
        )
        model = MinkLoc(
            backbone=backbone,
            pooling=pooling,
            normalize_embeddings=params.model_params.normalize_embeddings,
            self_att=False,
            add_FTU=False,
        )
    else:
        raise NotImplementedError("Model type not supported.")

    if weights_path:
        if isfile(weights_path):
            checkpoint = torch.load(
                weights_path, map_location=torch.device("cpu"), encoding="latin1")
            if 'net' in checkpoint:
                model.load_state_dict(checkpoint['net'])
            else:
                model = model_selective_init(model, checkpoint)
            print(f"==> Loaded checkpoint from {weights_path}")
        else:
            print(
                f"==> No checkpoint found at '{weights_path}', please move the checkpoint to this path")

    model.to(device)
    return model


def create_resnet_block(block_name: str) -> nn.Module:
    if block_name == "BasicBlock":
        block_module = BasicBlock
    elif block_name == "Bottleneck":
        block_module = Bottleneck
    elif block_name == "ECABasicBlock":
        block_module = ECABasicBlock
    else:
        raise NotImplementedError(
            "Unsupported network block: {}".format(block_name))

    return block_module


def model_selective_init(nn_model, checkpoint):
    pretrained_dict = (
        checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    )

    # For pretrained models, there are some keys to be replaced due to the key_name changes:
    for key in list(pretrained_dict.keys()):
        pretrained_dict[key.replace("pool.conv_sa1", "pool.conv_app")] = (
            pretrained_dict.pop(key)
        )

    model_dict = nn_model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape
    }
    redundent_keys_dict = {}
    redundent_keys_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k not in model_dict or model_dict[k].shape != pretrained_dict[k].shape
    }
    missing_keys_dict = {}
    missing_keys_dict = {
        k: v
        for k, v in model_dict.items()
        if k not in pretrained_dict
        or pretrained_dict[k].shape != model_dict[k].shape
    }
    # 2. overwrite entries in the existing state dict
    print("=> params of {} will be loaded".format(pretrained_dict.keys()))
    print(
        "=> params of {} in loaded pretrained_dict are redundent and will be omitted".format(
            redundent_keys_dict.keys()
        )
    )
    print(
        "=> params of {} in nn_model.dict are missing and will not be initialized".format(
            missing_keys_dict.keys()
        )
    )
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    keys = nn_model.load_state_dict(model_dict)
    print("=> keys missing and redundant:{}".format(keys))
    # model = model.to(device)
    return nn_model
