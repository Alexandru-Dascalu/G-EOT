
hyper_params = {'BatchSize': 6,
                'NumSubnets': 50,
                'SimulatorSteps': 1,
                'GeneratorSteps': 1,
                'ImageShape': [299, 299],

                # constants related to learning rate and loss
                'PenaltyWeight': 0.001,
                'LearningRate': 1e-3,
                'DecayRate': 0.95,
                'DecayAfter': 300,
                'LayerRegularisationWeight': 1e-4 * 0.5,

                # hyper params related to number of steps
                'ValidateAfter': 300,
                'TestSteps': 100,
                'WarmupSteps': 1000,
                'TotalSteps': 40000,

                # renderer settings for object pose
                'MinCameraDistance': 1.8,
                'MaxCameraDistance': 2.3,
                'MinTranslationX': -0.05,
                'MaxTranslationX': 0.05,
                'MinTranslationY': -0.05,
                'MaxTranslationY': 0.05,

                # image post-processing settings
                'MinBackgroundColour': 0.1,
                'MaxBackgroundColour': 1.0,
                'PrintError': False,
                'PrintErrorAddMin': -0.15,
                'PrintErrorAddMax': 0.15,
                'PrintErrorMultMin': 0.7,
                'PrintErrorMultMax': 1.3,
                'PhotoError': False,
                'PhotoErrorAddMin': -0.15,
                'PhotoErrorAddMax': 0.15,
                'PhotoErrorMultMin': 0.5,
                'PhotoErrorMultMax': 2.0,
                'GaussianNoiseStdDev': 0.1
                }
