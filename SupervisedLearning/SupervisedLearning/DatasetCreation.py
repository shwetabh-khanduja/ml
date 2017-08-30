import SupervisedLearning as sl

def CreateCreditScreeningDatasetInstance(
	raw_datset_root,
	train_split=80,
	random_states=[0,1,2,3,4],
	noise_percs=[0,10,15,20,25,30,50,70],
	train_size_percs=[20,30,40,50,60,70,80,90,100],
	imb_percs=[5,10,20,30,40,50,70,90]):
	"""
	Each random state will produce new data set for the given (noise,imb,train_size) settings
	"""
	for r in random_states:
		sl.GenerateCreditScreeningDataSetSplits(raw_datset_root,r,train_split,100-train_split,r,noise_percs=noise_percs,imbalance_percs=imb_percs,train_size_percs=train_size_percs)

def CreateCharacterRecognitionDatasetInstance(
	raw_datset_root,
	train_split_perc=80,
	random_states=[0,1,2,3,4],
	noise_percs=[0,10,15,20,25,30,50,70],
	train_size_percs=[20,30,40,50,60,70,80,90,100],
	imb_percs=[5,10,20,30,40,50,70,90]):
	"""
	Each random state will produce new data set for the given (noise,imb,train_size) settings
	"""
	for r in random_states:
		sl.GenerateVowelRecognitionDataSetSplits(raw_datset_root,r,train_split_perc,100-train_split_perc,r,noise_percs=noise_percs,imbalance_percs=imb_percs,train_size_percs=train_size_percs)
		