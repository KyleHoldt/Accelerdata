import numpy as np
from matplotlib import pyplot as plt
import scipy.fftpack
import traceback

class Data:
	'''
	
	-Example how to use:
	something = accelerdata.Data(Name_of_data_file)
	something.dataPlots()
	
	-Automatic truncation:
	--- Accepted region...
	---	starts when max and min magnitudes in a given window are within 10% of eachother
	---	ends when max amplitude is less than 10x the white noise amplitude
	
	-Manual truncation:
	--- Overrides automatic truncation.
	---	keyword: use=(StartTime,EndTime) in seconds
	---	ex: something = accelerdata.Data(Name_of_data_file,use=(StartTime,EndTime))
	
	-Figures are saved by default thus no ability to zoom.
	---If you want the ability to zoom in on the plots...
	---	keyword: save=False
	---	ex: something = accelerdata.Data(Name_of_data_file,save=False)
	
	
	Input File Format:
	'z0'	'z1'	'time'
	|	  |	  |
	|	  |	  |
	
	
	'''
	def __init__(self,filename,save=True,use=None,length=None,freq_gap=5,max_freq=None):
		self.save = save
		self.filename = filename
		self.titles = np.loadtxt(self.filename,dtype=str)
		self.title_z0 = self.titles[0][0]
		self.title_z1 = self.titles[0][1]
		
		self.data = np.loadtxt(self.filename,skiprows=1)
		self.z0_raw = self.data[:,0]
		self.z1_raw = self.data[:,1]
		self.time_raw = self.data[:,2] * 10e-7
		
		self.z0_long = self.z0_raw - np.mean(self.z0_raw)
		self.z1_long = self.z1_raw - np.mean(self.z1_raw)
		
		self.use = use
		if use == None:
			self.cut0_start,self.cut0_end = self.truncate(self.z0_long,self.title_z0)
			self.cut1_start,self.cut1_end = self.truncate(self.z1_long,self.title_z1)
		
		
		elif use != None:
			self.cut0_start, self.cut0_end = (self.get_index(self.time_raw,use[0]),self.get_index(self.time_raw,use[1]))
			self.cut1_start,self.cut1_end = self.cut0_start, self.cut0_end
		
		self.z0 = self.z0_long[self.cut0_start:self.cut0_end]
		self.z1 = self.z1_long[self.cut1_start:self.cut1_end]
		self.time = self.time_raw
		
		self.from_to = [[self.cut0_start,self.cut0_end],[self.cut1_start,self.cut1_end]]
		
		self.N = len(self.time_raw)
		self.dt = self.time_raw[1] - self.time_raw[0]
		self.fs = 1/self.dt
		self.T = max(self.time_raw) + self.dt
		self.df = 1/self.T
		
		
		
		
		self.length = length
		if length != None:
			pass
		
		self.freq_gap = freq_gap
		self.max_freq = max_freq
		
	
	def get_index(self,List,value):
		diff_List = List - value
		smallest = min(abs(diff_List))
		Where = np.where(diff_List == smallest)
		try:
			index = Where[0][0]
		
		except:
			Where = np.where(diff_List == -smallest)
			index = Where[0][0]
			#raise ValueError('No index found',Where)
		return index
		
	
	def fft(self,z,title,cut=False):
		
		freq = np.arange(0,int(self.fs),self.df)
		fft = abs(scipy.fftpack.fft(z))
		
		
		
		fft_freq = []
		for i in range(int(len(z)/2)):
			fft_freq.append([fft[i],freq[i]])
		
		fft_freq_sorted = sorted(fft_freq, key = lambda k: k[0])[::-1]
		
		search = fft_freq_sorted[:100]
		
		def remove(List_in):
			List = List_in.copy()
			for i in range(len(List)):
				test = List[i]
				for j in range(i+1,len(List)):
					check = List[j]
					if abs(test[1] - check[1])<self.freq_gap:
						List[j] = [0,0]
				
			
			List_new = []
			for each in List:
				if each != [0,0]:
					List_new.append(each)
			
			return List_new
		
		peak_freq = sorted(remove(search))
		freqs = [ peak_freq[i][1] for i in range(len(peak_freq))]
		
		if cut == False:
			max_freq = len(z)
		
		if cut == True:
			
			if self.max_freq == None:
				max_freq = int(1.5*max(freqs)/self.df)
			
			else:
				max_freq = self.max_freq/self.df
		
		
		fig = plt.figure()
		ax = fig.add_subplot(111)
		#ax.plot(freq[:len(fft)],fft)
		ax.plot(freq[:max_freq],fft[:max_freq])
		ax.set_ylim(top=1.1*max(fft[:max_freq]))
		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('Magnitude')
		if cut == True:
			for each in peak_freq:
				ax.annotate(str(round(each[1],2))+'\n',xy=(each[1],each[0]),horizontalalignment='center')
		
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		
		if cut == False:
			fig.suptitle(title+' full fft',y=1)
		
		elif cut == True:
			fig.suptitle(title+' relevant fft',y=1)
		
		fig.tight_layout()
		if self.save == True:
			fig.show()
			if cut == False:
				fig.savefig(title + ' full fft plot.png')
			
			elif cut == True:
				fig.savefig(title + ' relevant fft plot.png')
		
		elif self.save == False:
			fig.show()
		
	
	def Plot(self,t,z,title):
		N = len(z)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(t[-N:],z,'.',markersize=.5)
		ax.set_xlabel('Time [s]')
		ax.set_ylabel('Magnitude [mV]')
		fig.suptitle(title)
		ax.spines['right'].set_visible(False)
		ax.spines['top'].set_visible(False)
		if self.save == True:
			fig.show()
			fig.savefig(title + ' plot.png')
		
		elif self.save == False:
			fig.show()
	
	def truncate(self,z,name):
		first_noise = max(z[:100])
		last_noise = max(z[-5000:])
		
		try:
			i = np.where(z==max(z))[0][0]
			check = abs(max(z[i:i+500])/min(z[i:i+500]))
			while (check < .9) or (check > 1.1):
				i+=50
				check = abs(max(z[i:i+500])/min(z[i:i+500]))
		
		except Exception:
			print(traceback.format_exc())
			print('\nUnable to find truncation starting point for %s.'%name)
			i = 0
		
		try:
			j = np.where(abs(z-last_noise)>10)[0][-1] -  len(z)
			while abs(max(z[j-50:j]) / last_noise) < 10:
				j-=25
		
		except Exception:
			print(traceback.format_exc())
			print('\nUnable to find trunctation ending point for %s.'%name)
			j = len(z)
		
		return i,j
	
	def dataPlots(self):
		'''
		Plots the truncated and centered data from each accelerometer.
		'''
		self.Plot(self.time[self.cut0_start:self.cut0_end],self.z0,self.title_z0+' truncated data')
		self.Plot(self.time[self.cut1_start:self.cut1_end],self.z1,self.title_z1+' truncated data')
	
	def fftPlots(self):
		'''
		Plots the fft plot for each set of accelerometer data.
		'''
		self.fft(self.z0,self.title_z0)
		self.fft(self.z0,self.title_z0,cut=True)
		self.fft(self.z1,self.title_z1)
		self.fft(self.z1,self.title_z1,cut=True)
	
	def z0Plots(self):
		'''
		Plots the truncated and centered data and the fft for the first accelerometer.
		'''
		self.Plot(self.time_raw,self.z0_raw,self.title_z0+' raw data')
		self.Plot(self.time[self.cut0_start:self.cut0_end],self.z0,self.title_z0+' truncated data')
		self.fft(self.z0,self.title_z0)
		self.fft(self.z0,self.title_z0,cut=True)
	
	def z1Plots(self):
		'''
		Plots the truncated and centered data and the fft for the second accelerometer.
		'''
		self.Plot(self.time_raw,self.z1_raw,self.title_z1+' raw data')
		self.Plot(self.time[self.cut1_start:self.cut1_end],self.z1,self.title_z1+' truncated data')
		self.fft(self.z1,self.title_z1)
		self.fft(self.z1,self.title_z1,cut=True)
	
	def rawPlots(self):
		'''
		Plots the raw data from each accelerometer.
		'''
		self.Plot(self.time_raw,self.z0_raw,self.title_z0+' raw data')
		self.Plot(self.time_raw,self.z1_raw,self.title_z1+' raw data')
	
	def allPlots(self):
		'''
		Plots the raw data, the truncated and centered data, and the fft for each accelerometer.
		'''
		#self.rawPlots()
		self.z0Plots()
		self.z1Plots()
