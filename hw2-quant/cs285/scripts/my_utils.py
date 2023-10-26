import os
import torch
from torch import nn
from torch.quantization import quantize_dynamic
from torch.quantization import quantize_fx

# returns model size in MB
def get_model_size(mdl):
    torch.save(mdl.state_dict(), "tmp.pt")
    size = os.path.getsize("tmp.pt")/1e6
    os.remove('tmp.pt')
    return size


def shrink_that_thang(mdl, debug_print=False):
    if (debug_print):
        print("Original model size: %.2f MB" % get_model_size(mdl))

    ## EAGER MODE
    # model_quantized = quantize_dynamic(
    # model=mdl, qconfig_spec={nn.Linear}, dtype=torch.qint8, inplace=False)


    ## FX MODE
    qconfig_dict = {"": torch.quantization.default_dynamic_qconfig}  # An empty key denotes the default applied to all modules
    model_prepared = quantize_fx.prepare_fx(mdl, qconfig_dict)
    model_quantized = quantize_fx.convert_fx(model_prepared)

    if (debug_print):
        print("Quantized model size: %.2f MB" % get_model_size(model_quantized))
    
    return model_quantized
