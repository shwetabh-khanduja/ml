class DecisionTreeRunConfig:
	"""
	For decision tree the allowed parameter values are:
	prune=[True,False] -- based on this -R for reduced error pruning or -U for no pruning flag will be set
	numfolds=[3,4..] decides the validation set size -- -N flag will be set using this
	minleafsize=[2,3,4] decides the min num of instances per leaf -- -M flag will be set
	trainset=<path> sets -t
	testset=<path> if null -no-cv will be set and train set will be evaluated otherwise sets -T option
	class=last sets the class column

	In the constructor only numfolds,minleafsize and pruce must be passed in
	"""
	def __init__(self, **kwargs):
		"""
		kwargs is a dictionary of parameters with
		key as the parameter name and value as the list
		of parameter values.
		first options in prune must true if there
		keys are prune,numfolds,minleafsize
		"""
		self.prune_vals = kwargs["prune"]
		self.numfolds=kwargs["numfolds"]
		self.minleafsize=kwargs["minleafsize"]
		self.prune_idx=0
		self.numfolds_idx=0
		self.minleafsize_idx=0
		self.total_prune_options = len(self.numfolds) * len(self.minleafsize) if (self.prune_vals[0] == True) else False

	def GetNextConfigAlongWithIdentifier(self):
		if(self.prune_idx == len(self.prune_vals)):
			return None
		prune=self.prune_vals[self.prune_idx]
		self.prune_idx = self.prune_idx + 1
		options = {}
		options['prune'] = prune
		options['minleafsize'] = self.minleafsize[0]
		if(prune):
			options['numfolds'] = self.numfolds[0]
			id = "prune-{0}_numfolds-{1}_minleafsize-{2}".format(prune,options["numfolds"],options["minleafsize"])
		else:
			id="prune-False_numfolds-0_minleafsize-{0}".format(options["minleafsize"])
			
		options["id"] = id
		return options

	def GenerateWekaCommandline(self, options):
		"""
		1. wekajar
		2. modeloutputfile
		3. predictionoutputfile
		4. trainset
		5. testset
		6. numfolds
		7. minleafsize
		8. prune
		9. class
		10. attrs
		"""
		cmdline_template_no_pruning_train = "java -classpath \"{0}\" weka.classifiers.trees.J48 -t \"{1}\" -d \"{2}\" -no-cv -M {3} -c {4} -U -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {5}{6}\""
		cmdline_template_pruning_train = "java -classpath \"{0}\" weka.classifiers.trees.J48 -t \"{1}\" -d \"{2}\" -no-cv -M {3} -c {4} -N {7} -R -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {5}{6}\""
		cmdline_template_test = "java -classpath \"{0}\" weka.classifiers.trees.J48 -T \"{1}\" -l \"{2}\" -classifications \"weka.classifiers.evaluation.output.prediction.CSV -file {3}{4}\""

		attrs_flag = " -p "+options["attrs"] if("attrs" in options) else ""

		if("testset" in options):
			return cmdline_template_test.format(options["wekajar"],options["testset"],options["modeloutputfile"],options["predictionoutputfile"],attrs_flag)
		elif(options["prune"]):
			return cmdline_template_pruning_train.format(options["wekajar"],options["trainset"],options["modeloutputfile"],options["minleafsize"],options["class"],options["predictionoutputfile"],attrs_flag,options["numfolds"])
		else:
			return cmdline_template_no_pruning_train.format(options["wekajar"],options["trainset"],options["modeloutputfile"],options["minleafsize"],options["class"],options["predictionoutputfile"],attrs_flag)