{
	"data":{
		"filename": 
		// "sequence_length": 50, 
		// "training_rate": 0.9,
		// "normalise": true
	},
	"training": {
		"epochs": 2,
		"batch_size": 32
	},
	"model": {
		"loss": "mse",
		"optimiser": "adam",
		"sav_dir": "some_random_dir_to_save_model",
		"layers":[
				{
					"type": "Flatten"
				},
				{
					"type": "Normalisation"
				},
				{
					"type": "Dense",
					"neurons": 256,
					"activation": "relu"
				},
				{
					"type": "Regularisation",
					"l2_penalisation": 0.1
				},
				{
					"type": "Dense",
					"neurons": 128,
					"activation": "relu"
				},
				{
					"type": "Dropout",
					"rate": 0.2
				},
				{
					"type": "Normalisation", 
				},
				{
					"type": "Dense",
					"neuron": 64,
					"activation": "relu"
				},
				{
					"type": "Regularisation",
					"l2_penalisation": 0.05
				},
				{
					"type": "Normalisation"
				},
				{
					"type": "Dense",
					"neurons": 32,
					"activation": "relu"
				},
				{
					"type": "dense",
					"neurons": 1,
					"activation": "linear"
				}
			]
	}
}