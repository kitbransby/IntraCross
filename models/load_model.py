from models.intracross_model import IntraCross
from models.sidebranch_detector import SideBranchDetector
from models.calcium_detector import CalciumDetector

def load_model(config, device):
    if config['MODEL'] == 'intracross':
        model = IntraCross(config).to(device)
    elif config['MODEL'] == 'sidebranch_detector':
        if config['INFERENCE']:
            model = SideBranchDetector(num_classes=2,
                                            trainable_backbone_layers=5,
                                            encoder_weights=config['ENCODER_WEIGHTS'],
                                            rpn_pre_nms_top_n_test=config['PRE_NMS_TOP_N_TEST'], # test
                                            box_nms_thresh=config['NMS_IOU'],
                                            box_score_thresh=config['CONF_THRESH']
                                            ).to(device)
        else:
            model = SideBranchDetector(num_classes=2,
                                            trainable_backbone_layers=5,
                                            encoder_weights=config['ENCODER_WEIGHTS'],
                                            ).to(device)
            
    elif config['MODEL'] == 'calcium_detector':
        model = CalciumDetector(num_classes=config['NUM_CLASSES'], input_dim=config['INPUT_DIM'], encoder_weights=config['ENCODER_WEIGHTS']).to(device)
    else:
        print('Model not recognised')
    return model