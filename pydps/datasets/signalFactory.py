"""
Main class to generate signals to test with bpd
"""
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')


class SignalFactory():
	"""class bankSignal to generate signal to be test the bpd algorithm"""
	def __init__(self,size,sigma=6):
		"""
		Constructor wich fixs the size and the noise level

		"""

		assert(size>0), " The size of the signal must be positif"

		self.size=size;
		self.sigma=np.sqrt(sigma);

	def _plotCase(self,ini,nois):
		"""
		simple function to plot a case
		"""
		fig=plt.figure(figsize=(12,8))
		plt.plot(ini,label='initial')
		plt.plot(nois,label='noised',alpha=0.5)
		plt.legend(loc='best')
		plt.show()

	def _discreteSignal(self,dis,jumps):
		"""
		function to generate the signal given the discontinuities
		and the jumps wich must satisfy



		"""

		assert(len(dis)==len(jumps)), " dis and jumps must be the same size"
		if(max(dis)>self.size):
			" the discontinuities given must be inferior to the signal size"

		#initialisation of the signal
		sig=np.zeros((self.size,1))

		initial=0;

		for (i,val) in enumerate(jumps[:-1]):
			initial+=val;
			sig[dis[i]:dis[i+1]]=initial

		initial+=jumps[-1]
		sig[dis[-1]:]=initial;

		return (sig.T).squeeze();

	def discreteCase(self,dis,jumps,show=False):
		"""
		function to generate a signal and its noised coounterpart
		"""

		sig=self._discreteSignal(dis,jumps)
		noi=sig+np.random.normal(scale=self.sigma,size=len(sig))

		if(show):
			self._plotCase(sig,noi)

		return sig,noi

	def dichotomieCase(self,parts=4,minJump=10,show=False):
		"""
		generate a signal discontinuities that split the signal
		each time by half
		"""
		dis=[0]
		jumps=[100]
		for i in range(parts):
			dis.append(int(self.size-self.size/2**(i+1)))
			jumps.append(minJump)
		sig=self._discreteSignal(dis,jumps)
		noi=sig+np.random.normal(scale=self.sigma,size=len(sig))
		if(show):
			self._plotCase(sig,noi)

		return sig,noi


	def normalShapeCase(self,parts=5,minJump=10,show=False):
		"""
		function to generate a signal with a given parts  with the
		discontinuities in for the form of a normal shape (ie) the importance
		of the jumps decreases as we approach to the extremities
		"""

		assert(parts%2==0)," the number of parts must be odd"

		#discontinuities
		dis=np.linspace(0,self.size,parts+1).astype(int)[:-1]


		mid=(int)(parts/2);
		jumps=np.zeros(parts)
		jumps[0]=100


		#jumps
		for i in range(1,mid+1):
			jumps[i]=minJump*i

		for i in range(mid+1,parts):
			jumps[i]=jumps[i-1]-minJump


		ini=self._discreteSignal(dis,jumps)
		noi=ini+np.random.normal(size=self.size,scale=self.sigma)

		if(show):
			self._plotCase(ini,noi)
		return ini,noi

	def uniformCase(self, parts=3,minJump=10,show=False):
		"""
		function to generate a signal uniformly distributed with a given parts
		"""

		dis=np.linspace(0,self.size,parts+1).astype(int)
		dis=dis[:-1]

		jumps=np.zeros((parts))
		jumps[0]=100;
		for i in range(1,parts):
			jumps[i]=np.random.randint(minJump,20)

		sig=self._discreteSignal(dis,jumps)

		nois=sig+np.random.normal(size=self.size, scale=self.sigma)


		if(show):
			self._plotCase(sig,nois)
		return sig,nois

	def randomDis(self,disNum=10,jump=10,show=False):
		"""
		generate a precise number of discontinuities with a random distribution
		"""
		dis=np.zeros(disNum+1)
		dis[1:]=sorted(np.random.choice(np.arange(4,self.size),size=disNum,replace=False))

		jump=np.zeros(disNum+1)
		jump[0]=100; jump[1:]=10
		sig=self._discreteSignal(dis,jump)
		nois=sig+np.random.normal(size=self.size,scale=self.sigma)
		if(show):
			self._plotCase(sig,nois)

		return sig,nois

	def randomCase(self,x0=100,minPart=10,minJump=10,p=0.03,show=False):
		"""
		Generate a random case respecting that the number of discontinuities
		is inferior to maxDis and the jumps are also inferior to maxJumps
		Input:
		x0: initial value of the first plateau

		minJump= minimum jump (useful to assure convergence)
		p=  probability of a break
		"""
		sig=x0*np.ones(self.size)
		n=len(sig)
		breaks=np.random.uniform(size=n)<p;
		#ides of the breaks
		ids=np.where(breaks==True)[0];
		print(type(ids))
		jumps=np.random.uniform(low=minJump,high=2*minJump,size=ids.shape)
		signs=np.random.uniform(size=ids.shape)>0.7;
		jumps[signs]=-jumps[signs]

		for i in range(len(jumps)-1):
			sig[ids[i]:ids[i+1]]=x0+jumps[i];
			x0+=jumps[i];
		sig[ids[-1]:]=x0+jumps[-1];
		#nois=sig+np.random.normal(size=self.size,scale=self.sigma)
		nois=sig+np.random.laplace(size=self.size,scale=self.sigma)
		if(show):
			self._plotCase(sig,nois)

		return sig,nois

	def denseCase(self,numDis,minimum_distance=5,show=False):
		"""
		function to generate a worst case signal with
		the discontinuities condensed
		"""
		dis=np.array([r for r in range(0,(numDis*minimum_distance),minimum_distance)])
		jump=np.zeros(numDis); jump[0]=100; jump[1:]=10

		sig=self._discreteSignal(dis,jump)
		noi=sig+np.random.normal(scale=self.sigma,size=len(sig))

		if(show==True):
			self._plotCase(sig,noi)
		return sig,noi

	def uniformTest(self,S):
		"""
		Test that generate a set of unifom signal and test it against an algorithm
		"""

		pass

	def LevyProcess(self,x0=10,seed=None,std_jumps=4,std_noise=1,show=False):
		"""
		function to generate a random Levy process with same setup used on the article
		fast 1D regularization denoising
		"""
		if(seed!=None):
			np.random.seed=seed;

		ini=np.zeros(self.size);
		ini[0]=x0;
		A=np.random.uniform(size=self.size)>0.95;         #vector for choices
		Jumps=np.random.normal(scale=std_jumps,size=self.size)
		for i in range(1,self.size):
			if(A[i]):
				ini[i]=ini[i-1]+Jumps[i]
			else:
				ini[i]=ini[i-1]

		nois=ini+np.random.normal(scale=std_noise,size=self.size)

		if(show):
			self._plotCase(ini,nois)
		return (ini,nois)

	def stressTest(self,S):
		"""
		Test that generate a stress test, to test an algorithm
		"""

		pass

	def saveCase(self,sig):
		"""
		function to always save a case on sig file
		"""

		np.savetxt("sig",sig,header="%d"%len(sig),comments="")

	def pottsLab(self,type,noiseType='norm',nSamples=None,show=False):
		"""
		function to generate initial signal given some cases
		"""
		if(nSamples==None):
			nSamples=self.size
		x=np.linspace(-1,1,nSamples)
		y=np.zeros_like(x)
		if(type=='rect'):
			y=np.logical_and((x>-0.5),(x<0.5)).astype(float)
		if(type=='step' or type=='heaviside'):
			y=(x>=0).astype(float)
		if(type=='jumps'):
			y=np.zeros_like(x)
			for (i,x) in enumerate(x):
				if(x>-0.3):
					if((0.1 < x) & ( x < 0.4)):
						y[i]+=2;
					if(x<-0.1):
						y[i]+=1
		if(type=='equidistant'):
			n=8;
			step=int(nSamples/n);
			heights = np.array([3, 1, 7, 6, 5, 0,6,4]);
			for (i,h) in enumerate(heights):
				y[step*i:step*(i+1)]=h/7.0;

		if(noiseType=='norm'):
			noised=y+np.random.normal(scale=self.sigma,size=self.size)

		if(noiseType=='laplace'):
			noised=y+np.random.laplace(scale=self.sigma,size=self.size)
		if(show):
			plt.plot(y,'*',ms=2)
			plt.plot(noised,'.',ms=4)
			plt.ylim(-0.7,1.7)
			plt.savefig('test.pdf')
		return (y,noised)

if __name__ == '__main__':
	"""
	Test
	"""

	S=SignalFactory(1000,0.02)
	#sig,nois=S.normalShapeCase(6,minJump=10,show=True)
	#sig,noi=S.dichotomieCase(5,show=True)
	#sig,noi=S.randomDis(4,show=True)
	S.denseCase(6,minimum_distance=5,show=True)
	#S.saveCase(nois)
