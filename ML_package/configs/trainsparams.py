CSRNet_PARAMS={"lr":1e-6,
            "momentum":0.95,
            "maxEpochs":1000,
            "criterionMethode":'L2Loss',
            "optimizationMethod":'SGD',
            'batch_size': 1
            }
SANet_PARAMS={"lr":1e-5,
            "momentum":0.95,
            "maxEpochs":1000,
            "criterionMethode":'L2Loss',
            "optimizationMethod":'Adam',
            'batch_size': 1
            }
MCNN_PARAMS={"lr":1e-6,
            "momentum":0.95,
            "maxEpochs":1000,
            "criterionMethode":'MSELoss',
            "optimizationMethod":'SGD',
            'batch_size': 1
            }
CCNN_PARAMS={"lr":1e-6,
            "momentum":0.95,
            "maxEpochs":1000,
            "criterionMethode":'MSELoss',
            "optimizationMethod":'Adam',
            'batch_size': 8
            }