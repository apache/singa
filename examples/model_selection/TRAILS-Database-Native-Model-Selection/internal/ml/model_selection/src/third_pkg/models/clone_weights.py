import torch
import torch.nn as nn


def copy_conv(module, init):
    assert isinstance(module, nn.Conv2d), "invalid module : {:}".format(module)
    assert isinstance(init, nn.Conv2d), "invalid module : {:}".format(init)
    new_i, new_o = module.in_channels, module.out_channels
    module.weight.copy_(init.weight.detach()[:new_o, :new_i])
    if module.bias is not None:
        module.bias.copy_(init.bias.detach()[:new_o])


def copy_bn(module, init):
    assert isinstance(module, nn.BatchNorm2d), "invalid module : {:}".format(module)
    assert isinstance(init, nn.BatchNorm2d), "invalid module : {:}".format(init)
    num_features = module.num_features
    if module.weight is not None:
        module.weight.copy_(init.weight.detach()[:num_features])
    if module.bias is not None:
        module.bias.copy_(init.bias.detach()[:num_features])
    if module.running_mean is not None:
        module.running_mean.copy_(init.running_mean.detach()[:num_features])
    if module.running_var is not None:
        module.running_var.copy_(init.running_var.detach()[:num_features])


def copy_fc(module, init):
    assert isinstance(module, nn.Linear), "invalid module : {:}".format(module)
    assert isinstance(init, nn.Linear), "invalid module : {:}".format(init)
    new_i, new_o = module.in_features, module.out_features
    module.weight.copy_(init.weight.detach()[:new_o, :new_i])
    if module.bias is not None:
        module.bias.copy_(init.bias.detach()[:new_o])


def copy_base(module, init):
    assert type(module).__name__ in [
        "ConvBNReLU",
        "Downsample",
    ], "invalid module : {:}".format(module)
    assert type(init).__name__ in [
        "ConvBNReLU",
        "Downsample",
    ], "invalid module : {:}".format(init)
    if module.conv is not None:
        copy_conv(module.conv, init.conv)
    if module.bn is not None:
        copy_bn(module.bn, init.bn)


def copy_basic(module, init):
    copy_base(module.conv_a, init.conv_a)
    copy_base(module.conv_b, init.conv_b)
    if module.downsample is not None:
        if init.downsample is not None:
            copy_base(module.downsample, init.downsample)
        # else:
        # import pdb; pdb.set_trace()


def init_from_model(network, init_model):
    with torch.no_grad():
        copy_fc(network.classifier, init_model.classifier)
        for base, target in zip(init_model.layers, network.layers):
            assert (
                type(base).__name__ == type(target).__name__
            ), "invalid type : {:} vs {:}".format(base, target)
            if type(base).__name__ == "ConvBNReLU":
                copy_base(target, base)
            elif type(base).__name__ == "ResNetBasicblock":
                copy_basic(target, base)
            else:
                raise ValueError("unknown type name : {:}".format(type(base).__name__))
