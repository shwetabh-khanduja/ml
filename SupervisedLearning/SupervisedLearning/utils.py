import numpy as np;
import matplotlib.pyplot as plt;
import os;

def WriteTextToFile(file,text):
    f = open(file,'w')
    f.write(text)
    f.close()

def WriteTextArrayToFile(file,textArray):
    WriteTextToFile(file,'\n'.join(textArray))

def ReadLinesFromFile(file):
	f = open(file,'r')
	lines = [line.rstrip('\n') if(line != '\n') else line for line in f.readlines()]
	f.close()
	return lines

def ReadLineFromFile(file,line_idx):
	"""
	line_idx starts from 0
	so line number would be line idx + 1
	"""
	f = open(file,'r')
	index = 0
	line_at_idx = None
	for line in f.readlines():
		if(index == line_idx):
			line_at_idx = line
			break
		index = index + 1
	f.close()
	return line_at_idx

def SaveDataPlot(y1,
				y2 = None,
				x = None,
				filename=None,
				dispose_fig=True,
				x_axis_name="",
				y1_axis_name="",
				y2_axis_name="",
				title="",
				y1_plot_color='b',
				y2_plot_color='r',
				fig = None,
				ax1 = None,
				ax2 = None):
    
	if(x is None):
		x_count = y1.size
		if(y2 is not None and y2.size > x_count):
			x_count = y2.size
		x = [i + 1 for i in range(x_count)]

	fig, ax1 = plt.subplots()
	ax2 = ax1.twinx()

	ax1.plot(x, y1, y1_plot_color)
	ax1.set_xlabel(x_axis_name)
	ax1.set_ylabel(y1_axis_name, color=y1_plot_color)

	if y2 is not None:
		ax2.plot(x, y2, y2_plot_color)
		ax2.set_ylabel(y2_axis_name, color=y2_plot_color)

	if(title != "" or title is not None):
		plt.title(title)

	if filename is not None:
		fig.savefig(filename)

	if dispose_fig is True:
		plt.close(fig)
	else:
		return [fig,ax1,ax2]

def SaveDataSubplots(
				y1_matrix,
				nrows=1,
				ncols=1,
				y2_matrix = None,
				x_matrix = None,
				filename=None,
				dispose_fig=True,
				x_axis_name="",
				y1_axis_name="",
				y2_axis_name="",
				title="",
				y1_plot_color='b-',
				y2_plot_color='r-',
				):
	"""
	Creates a grid of size nrows X ncols subplots
	y1[y2] : each column wii be a new time series
	title : list
	"""
	assert(y1_matrix.shape[1] == nrows * ncols and (y2_matrix is None or y2_matrix.shape[1] == nrows * ncols) and (x_matrix is None or y2_matrix.shape[1] == nrows * ncols))
	fig,axs1 = plt.subplots(nrows,ncols)
	axs1 = [axs1] if nrows == 1 and ncols == 1 else axs1.flatten()
	axs2 = [ax.twinx() for ax in axs1]
	f = lambda g : g if isinstance(g,list) else [g for i in range(y1_matrix.shape[1])]

	x_axis_name = f(x_axis_name)
	y1_axis_name = f(y1_axis_name)
	y2_axis_name = f(y2_axis_name)
	title = f(title)

	for i in range(y1_matrix.shape[1]):
		y1 = y1_matrix[:,i]
		y2 = None if y2_matrix is None else y2_matrix[:,i]
		ax1 = axs1[i]
		ax2 = axs2[i]

		if(x_matrix is None):
			x_count = y1.size
			if(y2 is not None and y2.size > x_count):
				x_count = y2.size
			x = np.arange(x_count) + 1

		ax1.plot(x, y1, y1_plot_color)
		ax1.set_xlabel(x_axis_name[i])
		ax1.set_ylabel(y1_axis_name[i], color=y1_plot_color)

		if y2 is not None:
			ax2.plot(x, y2, y2_plot_color)
			ax2.set_ylabel(y2_axis_name[i], color=y2_plot_color)

		if(title[i] != ""):
			ax1.set_title(title[i])
			#plt.title(title)

	if filename is not None:
		fig.savefig(filename)

	if dispose_fig is True:
		plt.close(fig)
	else:
		return [fig,axs1,axs2]

def SaveHistogram(data,nbins,title="",x_axis_name="",y_axis_name="", file=None, show=False):
	f = plt.figure()
	plt.hist(data,nbins)
	plt.title(title)
	plt.xlabel(x_axis_name)
	plt.ylabel(y_axis_name)
	if(file is not None):
		f.savefig(file)
	if(show):
		plt.show()

def CreateDirectoryIfNotExists(path, is_file = True):
	dir = os.path.dirname(path) if(is_file) else path
	if not os.path.exists(dir):
		os.makedirs(dir)
	
def PreparePath(path,is_file=True):
	CreateDirectoryIfNotExists(path,is_file)
	return path

def Get_Subdirectories(a_dir):
    return [os.path.join(a_dir, name) for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

def ConcatToStr(delimiter,values_array):
	return delimiter.join([str(x) for x in values_array])