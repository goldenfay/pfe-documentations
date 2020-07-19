"""
    This module defines model's state_dict keys matches when trying to load an extern pretrained model.
    **NOTE:** these dictionnaries are subject oriented and this module is not generic.
"""
import torch
import re,os,glob,inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

def define_equivalence_dict(key_list, couples_equivalence_list) -> dict:
    equivalence_dict = dict()
    for el in key_list:
        corresponding = el
        for key, coresponding_key in couples_equivalence_list:
            corresponding = re.sub(key, coresponding_key,corresponding) #corresponding.raplace(key, coresponding_key)

        if corresponding!='':
            equivalence_dict[el] = corresponding

    return equivalence_dict

def get_dict_match(model_name):
    external_model_statedict=torch.load(os.path.join(currentdir,'frozen','external',model_name+'.pth'), map_location='cpu')
    if model_name=='CSRNet':
        match_dict= define_equivalence_dict(list(external_model_statedict),
                                           [('CCN.', ''), ('frontend', 'frontEnd'), ('backend', 'backEnd'),(r'gs\.gaussian.*','')])
        # match_dict= define_equivalence_dict(list(external_model_statedict),
        #                                    [('CCN.module.', ''), ('frontend', 'frontEnd'), ('backend', 'backEnd'),(r'gs\.gaussian.*','')])

    elif model_name=='SANet':
        match_dict= define_equivalence_dict(list(external_model_statedict),
                                           [('CCN.module.', ''), ('branch1x1', 'branch_1x1'), ('branch3x3', 'branch_3x3'), ('branch5x5', 'branch_5x5'), ('branch7x7', 'branch_7x7')])

    return external_model_statedict,match_dict                                    
# CSRNet_DICT_MATCH = {
#     'CCN.output_layer.weight': 'output_layer.weight',
#     'CCN.output_layer.bias': 'output_layer.bias',
#     'CCN.frontend.0.weight': 'frontEnd.0.weight',
#     'CCN.frontend.0.bias': 'frontEnd.0.bias',
#     'CCN.frontend.2.weight': 'frontEnd.2.weight',
#     'CCN.frontend.2.bias': 'frontEnd.2.bias',
#     'CCN.frontend.5.weight': 'frontEnd.5.weight',
#     'CCN.frontend.5.bias': 'frontEnd.5.bias',
#     'CCN.frontend.7.weight': 'frontEnd.7.weight',
#     'CCN.frontend.7.bias': 'frontEnd.7.bias',
#     'CCN.frontend.10.weight': 'frontEnd.10.weight',
#     'CCN.frontend.10.bias': 'frontEnd.10.bias',
#     'CCN.frontend.12.weight': 'frontEnd.12.weight',
#     'CCN.frontend.12.bias': 'frontEnd.12.bias',
#     'CCN.frontend.14.weight': 'frontEnd.14.weight',
#     'CCN.frontend.14.bias': 'frontEnd.14.bias',
#     'CCN.frontend.17.weight': 'frontEnd.17.weight',
#     'CCN.frontend.17.bias': 'frontEnd.17.bias',
#     'CCN.frontend.19.weight': 'frontEnd.19.weight',
#     'CCN.frontend.19.bias': 'frontEnd.19.bias',
#     'CCN.frontend.21.weight': 'frontEnd.21.weight',
#     'CCN.frontend.21.bias': 'frontEnd.21.bias',
#     'CCN.backend.0.weight': 'backEnd.0.weight',
#     'CCN.backend.0.bias': 'backEnd.0.bias',
#     'CCN.backend.2.weight': 'backEnd.2.weight',
#     'CCN.backend.2.bias': 'backEnd.2.bias',
#     'CCN.backend.4.weight': 'backEnd.4.weight',
#     'CCN.backend.4.bias': 'backEnd.4.bias',
#     'CCN.backend.6.weight': 'backEnd.6.weight',
#     'CCN.backend.6.bias': 'backEnd.6.bias',
#     'CCN.backend.8.weight': 'backEnd.8.weight',
#     'CCN.backend.8.bias': 'backEnd.8.bias',
#     'CCN.backend.10.weight': 'backEnd.10.weight',
#     'CCN.backend.10.bias': 'backEnd.10.bias'
# }
# SANet_DICT_MATCH = define_equivalence_dict(list(torch.load('08-SANet_all_ep_57_mae_42.4_mse_85.4.pth', map_location='cpu')),
#                                            [('CCN.module.', ''), ('branch1x1', 'branch_1x1'), ('branch3x3', 'branch_3x3'), ('branch5x5', 'branch_5x5'), ('branch7x7', 'branch_7x7')])

['CCN.module.frontend.0.weight',
 'CCN.module.frontend.0.bias',
 'CCN.module.frontend.2.weight',
 'CCN.module.frontend.2.bias',
 'CCN.module.frontend.5.weight',
 'CCN.module.frontend.5.bias',
 'CCN.module.frontend.7.weight',
 'CCN.module.frontend.7.bias',
 'CCN.module.frontend.10.weight',
 'CCN.module.frontend.10.bias',
 'CCN.module.frontend.12.weight',
 'CCN.module.frontend.12.bias',
 'CCN.module.frontend.14.weight',
 'CCN.module.frontend.14.bias',
 'CCN.module.frontend.17.weight',
 'CCN.module.frontend.17.bias',
 'CCN.module.frontend.19.weight',
 'CCN.module.frontend.19.bias',
 'CCN.module.frontend.21.weight',
 'CCN.module.frontend.21.bias',
 'CCN.module.backend.0.weight',
 'CCN.module.backend.0.bias',
 'CCN.module.backend.2.weight',
 'CCN.module.backend.2.bias',
 'CCN.module.backend.4.weight',
 'CCN.module.backend.4.bias',
 'CCN.module.backend.6.weight',
 'CCN.module.backend.6.bias',
 'CCN.module.backend.8.weight',
 'CCN.module.backend.8.bias',
 'CCN.module.backend.10.weight',
 'CCN.module.backend.10.bias',
 'CCN.module.output_layer.weight',
 'CCN.module.output_layer.bias']