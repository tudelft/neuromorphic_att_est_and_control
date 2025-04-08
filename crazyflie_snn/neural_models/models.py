from crazyflie_snn.neural_models.snn import SNN, OneLayerSNN, IntegrateSNN, RSNN, OneLayerRSNN, FullRSNN, OneLayerSoftRSNN
from crazyflie_snn.neural_models.ann import ANN


models = {
    "SNN": SNN,
    "OneLayerSNN": OneLayerSNN,
    "IntegrateSNN": IntegrateSNN,
    "RSNN": RSNN,
    "OneLayerRSNN": OneLayerRSNN,
    # "IWTANeuron": IWTANeuron,
    "ANN": ANN,
    "FullRSNN": FullRSNN,
    "OneLayerSoftRSNN": OneLayerSoftRSNN
}