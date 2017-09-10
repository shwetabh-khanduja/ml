class NeuralNetworkRunConfig():
	
	def __init__(self, **kwargs):
		"""
		Since we are doing grid search over the other
		parameters, everytime, here we are only passing
		the earlystopping criteria values
		"""
		self.early_stopping = [False,True]
		self.early_stopping_idx = 0

	def GetNextConfigAlongWithIdentifier(self):
		if(self.early_stopping_idx == 2):
			return None
		id = "earlystop-{0}".format(str(self.early_stopping[self.early_stopping_idx]))
		val = self.early_stopping[self.early_stopping_idx]
		self.early_stopping_idx = self.early_stopping_idx + 1
		return {"earlystopping":val,"id" : id}



