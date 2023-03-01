
from typing import Tuple, Union
import torch.nn as nn
import torchvision


def get_model(name: str, headless: bool = False) -> Union[nn.Module, Tuple[nn.Module, int]]:
    """Get classical neural network model

    Parameters
    ----------
    name: Model name as in torchvision
    headless: Specify if last classifier layer should removed

    Returns
    -------
    model: Parametric module to be learned
    fan_in: Number of output neurons (returned if headless = True)
    """
    model = torchvision.models.__dict__[name]()

    if headless:
        match name[:3]:
            case "ale":
                fan_in = model.classifier[1].in_features
                model.classifier = nn.Identity()
            case "con" | "eff" | "mna" | "mob":
                fan_in = model.classifier[-1].in_features
                model.classifier = nn.Identity()
            case "den":
                fan_in = model.classifier.in_features
                model.classifier = nn.Identity()
            case "goo" | "inc" | "reg" | "res" | "wid" | "shu":
                fan_in = model.fc.in_features
                model.fc = nn.Identity()
            case "swi":
                fan_in = model.head.in_features
                model.head = nn.Identity()
            case "vit":
                fan_in = model.heads.head.in_features
                model.heads.head = nn.Identity()
            case "vgg":
                fan_in = model.classifier[0].in_features
                model.classifier = nn.Identity()
            case _:
                raise NotImplementedError(f"Headless option is not implemented for model {name}.")
        return model, fan_in

    return model


def preprocessing(name: str) -> torchvision.transforms._presets.ImageClassification:
    """Get function to transform input to map specific model specifications

    Parameters
    ----------
    name: Model name as in torchvision

    Returns
    -------
    preprocess: Function to map raw inputs into the right size for the specified model
    """
    match name:
        case "alexnet":
            weight_name = "AlexNet_Weights"

        case "convnext_tiny":
            weight_name = "ConvNeXt_Tiny_Weights"
        case "convnext_small":
            weight_name = "ConvNeXt_Small_Weights"
        case "convnext_base":
            weight_name = "ConvNeXt_Base_Weights"
        case "convnext_large":
            weight_name = "ConvNeXt_Large_Weights"

        case "densenet121":
            weight_name = "DenseNet121_Weights"
        case "densenet161":
            weight_name = "DenseNet161_Weights"
        case "densenet169":
            weight_name = "DenseNet169_Weights"
        case "densenet201":
            weight_name = "DenseNet201_Weights"

        case "efficientnet_b0":
            weight_name = "EfficientNet_B0_Weights"
        case "efficientnet_b1":
            weight_name = "EfficientNet_B1_Weights"
        case "efficientnet_b2":
            weight_name = "EfficientNet_B2_Weights"
        case "efficientnet_b3":
            weight_name = "EfficientNet_B3_Weights"
        case "efficientnet_b4":
            weight_name = "EfficientNet_B4_Weights"
        case "efficientnet_b5":
            weight_name = "EfficientNet_B5_Weights"
        case "efficientnet_b6":
            weight_name = "EfficientNet_B6_Weights"
        case "efficientnet_b7":
            weight_name = "EfficientNet_B7_Weights"
        case "efficientnet_v2_s":
            weight_name = "EfficientNet_V2_S_Weights"
        case "efficientnet_v2_m":
            weight_name = "EfficientNet_V2_M_Weights"
        case "efficientnet_v2_l":
            weight_name = "EfficientNet_V2_L_Weights"

        case "GoogLeNet":
            weight_name = "GoogLeNet_Weights"

        case "inception_v3":
            weight_name = "Inception_V3_Weights"

        case "mnasnet0_5":
            weight_name = "MNASNet0_5_Weights"
        case "mnasnet0_75":
            weight_name = "MNASNet0_75_Weights"
        case "mnasnet1_0":
            weight_name = "MNASNet1_0_Weights"
        case "mnasnet1_3":
            weight_name = "MNASNet1_3_Weights"

        case "mobilenet_v2":
            weight_name = "MobileNet_V2_Weights"
        case "mobilenet_v3_large":
            weight_name = "MobileNet_V3_Large_Weights"
        case "mobilenet_v3_small":
            weight_name = "MobileNet_V3_Small_Weights"

        case "regnet_y_400mf":
            weight_name = "RegNet_Y_400MF_Weights"
        case "regnet_y_800mf":
            weight_name = "RegNet_Y_800MF_Weights"
        case "regnet_y_1_6gf":
            weight_name = "RegNet_Y_1_6GF_Weights"
        case "regnet_y_3_2gf":
            weight_name = "RegNet_Y_3_2GF_Weights"
        case "regnet_y_8gf":
            weight_name = "RegNet_Y_8GF_Weights"
        case "regnet_y_16gf":
            weight_name = "RegNet_Y_16GF_Weights"
        case "regnet_y_32gf":
            weight_name = "RegNet_Y_32GF_Weights"
        case "regnet_y_128gf":
            weight_name = "RegNet_Y_128GF_Weights"
        case "regnet_x_400mf":
            weight_name = "RegNet_X_400MF_Weights"
        case "regnet_x_800mf":
            weight_name = "RegNet_X_800MF_Weights"
        case "regnet_x_1_6gf":
            weight_name = "RegNet_X_1_6GF_Weights"
        case "regnet_x_3_2gf":
            weight_name = "RegNet_X_3_2GF_Weights"
        case "regnet_x_8gf":
            weight_name = "RegNet_X_8GF_Weights"
        case "regnet_x_16gf":
            weight_name = "RegNet_X_16GF_Weights"
        case "regnet_x_32gf":
            weight_name = "RegNet_X_32GF_Weights"

        case "resnet18":
            weight_name = "ResNet18_Weights"
        case "resnet34":
            weight_name = "ResNet34_Weights"
        case "resnet50":
            weight_name = "ResNet50_Weights"
        case "resnet101":
            weight_name = "ResNet101_Weights"
        case "resnet152":
            weight_name = "ResNet152_Weights"
        case "resnext50_32x4d":
            weight_name = "ResNeXt50_32X4D_Weights"
        case "resnext101_32x8d":
            weight_name = "ResNeXt101_32X8D_Weights"
        case "resnext101_64x4d":
            weight_name = "ResNeXt101_64X4D_Weights"
        case "wide_resnet50_2":
            weight_name = "Wide_ResNet50_2_Weights"
        case "wide_resnet101_2":
            weight_name = "Wide_ResNet101_2_Weights"

        case "shufflenet_v2_x0_5":
            weight_name = "ShuffleNet_V2_X0_5_Weights"
        case "shufflenet_v2_x1_0":
            weight_name = "ShuffleNet_V2_X1_0_Weights"
        case "shufflenet_v2_x1_5":
            weight_name = "ShuffleNet_V2_X1_5_Weights"
        case "shufflenet_v2_x2_0":
            weight_name = "ShuffleNet_V2_X2_0_Weights"

        case "squeezenet1_0":
            weight_name = "SqueezeNet1_0_Weights"
        case "squeezenet1_1":
            weight_name = "SqueezeNet1_1_Weights"

        case "swin_t":
            weight_name = "Swin_T_Weights"
        case "swin_s":
            weight_name = "Swin_S_Weights"
        case "swin_b":
            weight_name = "Swin_B_Weights"

        case "vgg11":
            weight_name = "VGG11_Weights"
        case "vgg11_bn":
            weight_name = "VGG11_BN_Weights"
        case "vgg13":
            weight_name = "VGG13_Weights"
        case "vgg13_bn":
            weight_name = "VGG13_BN_Weights"
        case "vgg16":
            weight_name = "VGG16_Weights"
        case "vgg16_bn":
            weight_name = "VGG16_BN_Weights"
        case "vgg19":
            weight_name = "VGG19_Weights"
        case "vgg19_bn":
            weight_name = "VGG19_BN_Weights"

        case "vit_b_16":
            weight_name = "ViT_B_16_Weights"
        case "vit_b_32":
            weight_name = "ViT_B_32_Weights"
        case "vit_l_16":
            weight_name = "ViT_L_16_Weights"
        case "vit_l_32":
            weight_name = "ViT_L_32_Weights"
        case "vit_h_14":
            weight_name = "ViT_H_14_Weights"

        case _:
            raise NotImplementedError(f"No implementation for model {name}.")

    return torchvision.models.__dict__[weight_name].DEFAULT.transforms()
