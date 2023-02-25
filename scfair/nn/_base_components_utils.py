import torch
from torch import nn



def set_requires_grad(nn_module: nn.Module, req_grad: bool):
    for m in nn_module.modules():
        for par in m.parameters():
            par.requires_grad = req_grad
            
            
# TODO: implement set_requires_grad for remaining classes if necessary


def FCLayers_set_requires_grad(fc_class, req_grad):
    set_requires_grad(fc_class.fc_layers, req_grad)
            
        

def Encoder_set_requires_grad(enc_class, req_grad):
    set_requires_grad(enc_class.encoder, req_grad)
    set_requires_grad(enc_class.mean_encoder, req_grad)
    set_requires_grad(enc_class.var_encoder, req_grad)

    
    
def DecoderSCVI_set_requires_grad(decscvi_class, req_grad):
    set_requires_grad(decscvi_class.px_decoder, req_grad)
    set_requires_grad(decscvi_class.px_scale_decoder, req_grad)
    set_requires_grad(decscvi_class.px_r_decoder, req_grad)
    set_requires_grad(decscvi_class.px_dropout_decoder, req_grad)
    
    
    
def LinearDecoderSCVI_set_requires_grad(lindecscvi_class, req_grad):
    set_requires_grad(lindecscvi_class.factor_regressor, req_grad)
    set_requires_grad(lindecscvi_class.px_dropout_decoder, req_grad)    
    
    
    
def Decoder_set_requires_grad(dec_class, req_grad):
    set_requires_grad(dec_class.decoder, req_grad)
    set_requires_grad(dec_class.mean_decoder, req_grad)
    set_requires_grad(dec_class.var_decoder, req_grad)
    
    
