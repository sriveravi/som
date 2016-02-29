from numpy import sqrt,sqrt,array,unravel_index,nditer,linalg,random,subtract,power,exp,pi,zeros,arange,outer,meshgrid,linspace,log,where, genfromtxt, argmin
import numpy as np
import random as r
from pylab import plot,axis,show,pcolor,colorbar,bone, xlabel, ylabel, legend
from collections import defaultdict
from warnings import warn
import copy as cop
#import time
import sys

"""
    Minimalistic implementation of the Self Organizing Maps (SOM).

    Giuseppe Vettigli 2013.
"""
#-----------------------------------------------------------------
#       SOM class below ( has composite minisom as an object)
#-----------------------------------------------------------------

class SOM(object):

    def __init__(self, file_name='idealInput.csv', columns = [0,1,3,4,5,6,7], width = 6, height = 5, 
                 sigma=1.2,  learning_rate=0.5, predThresh=np.inf, 
                 bmuFeatures=None, predMode='prot' ):
        # columns in file names are:
        # ['Antenna','Mouth','Back','Hand','Feet','Tail','Body','Color/Side','Label', 'category', 'accuracy', 'isLabeled']
        #    skipping feature 2 in default, because we aren't using back (so 7 features without acoustic label)
        #  Also, since feature 7( color) redundant, I use that to define 'side'
        #
        #       visual object only      = [0,1,3,4,5,6]
        #       visual object with side = [0,1,3,4,5,6,7] (side is 7)
        #       visual, side, acoustic  = [0,1,3,4,5,6,7,8]
        # Assume each feature coded binary (1 for A/0 for B), and it converts this to orthogonal representation

        # load data and normalization
        self.data = np.genfromtxt( file_name, delimiter=',',usecols=columns, skip_header=1)   
        self.data = binToOrthog( self.data) # make orthogonal representation
        self.data *= 2.0 
        self.data += -1.0  # -0.5 
        # get target labels
        self.target = np.genfromtxt(file_name,delimiter=',',usecols=(9),dtype=str, skip_header=1) # loading the labels
        #get infant accuracy
        self.acc = np.genfromtxt(file_name,delimiter=',',usecols=(10),dtype=float, skip_header=1) 
        self.somAcc = self.acc*np.nan
        
        # init som
        self.som = MiniSom(width, height, self.data.shape[1], sigma=sigma,learning_rate=learning_rate, bmuFeatures=bmuFeatures)
        #self.som.random_weights_init(self.data)
        self.initWeights()  
        self.dataIdx = 0 # to index what data sample for one shot learning
        
        # get idealized data
        self.ideal = np.genfromtxt( 'stimuli.csv', delimiter=',',usecols=columns, skip_header=1) 
        self.ideal = binToOrthog(self.ideal ) # make orthogonal representation
        self.ideal *= 2.0 
        self.ideal += -1.0  # -0.5 
        self.idealTarget = genfromtxt('stimuli.csv',delimiter=',',usecols=(9),dtype=str, skip_header=1) # loading the labels
               
        # index of ideal samples that are category prototypes
        self.protIdx = np.array([6,7]) # doesn't use this anymore (this was when using ideal protoypes, see getProt function)
        self.predThresh = predThresh  # when classifying by prototype, if distance larger than this then random decision
        self.predMode = predMode
        

    def pertWeights( self, scale=.2 ):     
        print( 'Adding noise to SOM weights')
        pertAmount = scale*(np.random.random_sample( self.som.weights.shape)-.5)
        self.som.weights = self.som.weights + pertAmount
    
    def pertData( self, p=.2, verbose=False):
        if verbose:
            print( 'Making %.2f percent of inputs 0.0' %(p*100))
        # randomly get proportion of indices to switch, then replace
        p = max( min(p,1), 0 ) # set in range [0,1]
        
   
#         if skipCols == None:
        
#         else:
#         sh = self.data.shape
#         noiseIndex = np.random.binomial(1,p,(sh[0], sh[1]-1 ) ) 
            
        noiseIndex = np.random.binomial(1,p, self.data.shape)  #ones at p proportion of samples
        self.data[noiseIndex ==1 ] = 0
        #return data
    
    # plot the thing
    def show( self, maxIdx=None, indices=None):
        print( 'Exemplar projection')
        som = self.som
        if maxIdx == None:
            maxIdx = len(self.data)            
        if indices ==None:            
            data= self.data[0:maxIdx]
            target = self.target
        else:
            data= self.data[indices]
            target= self.target[indices]
            
        bone()
        pcolor(som.distance_map().T) # plotting the distance map as background
        colorbar()
        t = zeros(len(target),dtype=int)
        t[target == 'A'] = 0
        t[target == 'B'] = 1
        # use different colors and markers for each label
        markers = ['o','s','D']
        colors = ['r','g','b']
        for cnt,xx in enumerate(data):
            w = som.winner(xx) # getting the winner
            # palce a marker on the winning position for the sample xx
            plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
                markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
        axis([0,som.weights.shape[0],0,som.weights.shape[1]])
        show() # show the figure
        
    def showIdeal( self):
        print( 'Ideal exemplar projection')
        som = self.som
        data= self.ideal
        target = self.idealTarget
        bone()
        pcolor(som.distance_map().T) # plotting the distance map as background
        colorbar()
        t = zeros(len(target),dtype=int)
        t[target == 'A'] = 0
        t[target == 'B'] = 1
        # use different colors and markers for each label
        markers = ['o','s','D']
        colors = ['r','g','b']
        for cnt,xx in enumerate(data):
            w = som.winner(xx) # getting the winner
            # palce a marker on the winning position for the sample xx
            plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None',
                markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)
        axis([0,som.weights.shape[0],0,som.weights.shape[1]])
        show() # show the figure        
        
    def train(self, iters = 30):
        self.som.train_random(self.data,iters)
#         print( 'Post training quantization error: %f' % self.som.quantization_error(self.data))

    """def trainOneGliozzi(self):
        # using gliozzi thing
        # init learning params
        self.som.sigma_gliozzi = linspace(1.2, 0.8, num=len(self.data))
        self.som._init_T(len(self.data))   
        
        #  will use the data idx to slowly reduce sigma, and learning rate in update function
        self.som.update_gliozzi(self.data[self.dataIdx],self.som.winner(self.data[self.som.dataIdx]), self.som.dataIdx)

        # update training index (for next time)
        self.dataIdx += 1
        if self.dataIdx >= len(self.data):
            self.dataIdx = 0 
    """
    
    def trainOne(self ):
        """ Trains the SOM picking samples from data one at a time 
            featIdx specifies subset of features to use for finding BMU
        """
        #self.running = True        
        d = self.dataIdx  # data index to train
        self.som._init_T(len(self.data))      
        self.som.update( self.data[d], self.som.winner(self.data[d]), d)  # update
        #self.running = False
        
        # update training index (for next time)
        self.dataIdx += 1
        if self.dataIdx >= len(self.data):
            self.dataIdx = 0        

    def predOne(self, verbose=False):
        # Do the next one prediction
        
        # get BMU index and weight 
        x = self.data[self.dataIdx]        
        bmuW = self.som.weights[self.som.winner(x)]
        protDist = []
        protLab = []
        
        if self.predMode == 'prot':
            # find distance to all prototypes  
            # print('Decision based on prototype')
            for prot,label in self.getProt():                       
                #protMapped = self.som.weights[self.som.winner(prot) ]
                protDist.append( linalg.norm( bmuW- prot )) # self.ideal[ idx ])) 
                protLab.append( label )             #  self.idealTarget[idx] )     

            # SOM decision  ( random if not close to any prototoype or first trial)
            if len( protDist ) >0:                       
                somLab = protLab[ argmin( protDist ) ]  # nearest category             
                # make random if not like either prototype
                if min(protDist) > self.predThresh:
                    somLab = ( 'A' if r.random() > .5 else 'B' )                
            else: # random on first trial
                somLab = ( 'A' if r.random() > .5 else 'B' )
                
        # decide based on the value of 'side' feature        
        elif  self.predMode == 'side':      
            # print( 'Decision based on side feature')
            # find distance to all prototypes (for just the side feature)                   
            for prot,label in self.getProt(idealProt = True ):                       
                protDist.append( linalg.norm( bmuW[-2:]- prot[-2:] )) # self.ideal[ idx ])) 
                protLab.append( label )             #  self.idealTarget[idx] )     
            # SOM decision  ( random if not close to any prototoype or first trial)
            if len( protDist ) >0:                       
                somLab = protLab[ argmin( protDist ) ]  # nearest category             
                # make random if not like either prototype
                if min(protDist) > self.predThresh:
                    somLab = ( 'A' if r.random() > .5 else 'B' )                
            else: # random on first trial
                somLab = ( 'A' if r.random() > .5 else 'B' )           
            
        else:
            print ('Invalid prediction mode entered.  Should be prot or side')
        
        # update SOM accuracy
        trueLab = self.target[self.dataIdx]
        self.somAcc[self.dataIdx] = (1 if somLab==trueLab else 0 )
        
        # get child label
        #  somAcc = (1 if predLab == trueLab else 0 )
        if self.acc[self.dataIdx ] == 1: # if accurate
            childLab = trueLab
        elif np.isnan(self.acc[self.dataIdx ]): # if NaN
            childLab = 'N'
        else: # if wrong
            childLab = 'AB'.strip(trueLab) 
    
        if verbose:
            print( 'Qnt error current Sample: %.4f' % linalg.norm(x- bmuW) )            
            print( 'SOM/Child/True Label: %s/%s/%s.' % (  somLab, childLab, trueLab)  )
            print( 'Project prototype distances: ' )
            print protDist, protLab
        return ( somLab, childLab ) 

    def predAll(self, reset = True, skip =0, verbose= True, reps=1 ):
        
        fullSomList = np.array([])
        fullChildList = np.array([])
        
        for numRepetitions in range(reps): # repeat for random initialization of som            
            # initialize som weights, and data index if reset=True
            if reset:
                self.initWeights()                
                self.dataIdx = 0 # to index what data sample for one shot learning
            
            # init for storing results
            somList = np.chararray(len(self.data)) #*np.nan
            childList = np.chararray(len(self.data)) #*np.nan
            # go through each trial and predict
            for idx in range(len(self.data)):          
                #predict
                somLab, childLab = self.predOne()
                somList[idx] = somLab
                childList[idx] = childLab
                # train
                self.trainOne()  #  don't use mod.trainOneGliozzi()

            # deal with missing values and skip first if requested
            somList = somList[skip:]  # kick out first skip samples
            childList = childList[skip:] # kick out first skip samples        
            somList = somList[ childList != 'N'] # N for nan
            childList = childList[ childList != 'N']
            
            # store in the big list
            fullSomList = np.concatenate( (fullSomList, somList), axis=0)
            fullChildList = np.concatenate( (fullChildList, childList), axis=0)
        
        # get how well model does
        if len(fullChildList) >0:
            scAgreement = float(np.sum( fullSomList==fullChildList))/len(fullSomList)
        else:
            scAgreement = np.nan
            
        if verbose:
            print 'SOM pred: ', fullSomList
            print 'Inf pred: ', fullChildList            
            print('SOM and infant agree %.2f of time' % scAgreement )
        return fullSomList == fullChildList

    
    
    def separation( self):
        """return the norm of the difference vector between the mean of projected idealized class
            A exemplars, and class B exemplars.  Essentially, the separation between those classes 
            May need to include a variance normalization in future. """
        A =  self.ideal[ self.idealTarget=='A']
        B =  self.ideal[ self.idealTarget=='B']
        projA = np.empty(A.shape) * np.nan
        projB = np.empty(B.shape) * np.nan
        # go through all catA
        for idx,x in enumerate(A):
            projA[idx] = self.som.weights[self.som.winner(x)]
        for idx,x in enumerate(B):
            projB[idx] = self.som.weights[self.som.winner(x)]
        diff = linalg.norm(projA.mean(axis=0)-projB.mean(axis=0) ) 
        # pooled variance estimate (not standard, since using average variance across dimensions for a class)
        pVar = pVar = (projA.var(axis=0).mean() + projB.var(axis=0).mean() )/2  
        return diff/np.sqrt( pVar)
    
    def getProt( self, idealProt = False ):
        prot = []
        label = []    
        
        # use true prototypes from ideal data
        if idealProt:
            for idx in self.protIdx:
                prot.append(self.ideal[idx,:])
                label.append(self.idealTarget[idx])        
        # estimate prototypes based in training data
        else:
            # get data/labels just until the current iteration (data is projected onto SOM)
            targTemp = self.target[:self.dataIdx]
            dataTemp = self.som.quantization( self.data[:self.dataIdx,:] ) # projected samples
            
            # loop through all labels, calc prototypes by average
            for t in np.unique(targTemp):    
                prot.append(np.mean( dataTemp[ targTemp==t, :], axis=0 ))  # mean of category
                label.append(t) # corresponding label                
        return zip(prot,label )   
        
    def initWeights(self, scale=1 ):
        """ Initializes the weights of the SOM randomly around 0 """        
        self.som.weights =scale*(np.random.random_sample( self.som.weights.shape)-.5)


    def showLearnCurves(self, window=6):
        curveSOM = slideAcc( self.somAcc, window )
        curveSubj = slideAcc( self.acc, window )
        plot( np.arange(len(curveSOM)) ,curveSOM, 'k.-', linewidth=3, label='SOM')
        plot( np.arange(len(curveSubj)), curveSubj, 'g--', linewidth=3, label='Person')
        plot( np.arange(0,50), np.ones(50)*.5,'r-', linewidth=2, label='chance')
        xlabel('trial')
        legend(loc=4)
        self.curveSOM = curveSOM
        self.curveSubj = curveSubj
        
    
    def meanLrnCurves( self, iters = 10, removeProp = 0  ):
        # backup ori data, init acc vector
        dataOri = cop.copy(self.data)
        accSum = np.zeros(len(self.acc))
        
        for i1 in range(iters):
            # train/predict SOM several times            
            self.data = cop.copy(dataOri)
            self.pertData(removeProp)  #not perfect data to train                      
            agreement = self.predAll( verbose=False, skip=0, reps=1)     
            accSum =accSum + self.somAcc
        self.somAcc = accSum/iters # average accuracy
        
        # return accuracy
        return self.somAcc
    
            
    
#-------------------------------------------------
#    a few general purpose methods
#------------------------------------------------------------------

# def getIdeal( file_name='stimuli.csv', columns = [0,1,2,3,4,5,6,7]): 
#     # return the ideal data for checking quantization error
#     data = np.genfromtxt( file_name, delimiter=',',usecols=columns, skip_header=1)     
#     return ( data[:7,:], data[7:,:] )


def slideAcc( acc, window=6 ):
    # return a sliding window average of accuracy 
    N = len(acc)
    wAcc = np.zeros((N))*np.nan
    for i1 in range(window,N+1):      
        a = acc[i1-window:i1]
        #print i1, a
        wAcc[i1-1] = np.nanmean(a) # get mean over window    
    return wAcc

def binToOrthog( data ):
    # take a binary data matrix and convert to orthogonal representation    
    N,d = data.shape # N,d : num samples, num features
    dataOut = np.ones((N, 2*d))*np.nan  
    dataOut[:,::2] = data
    dataOut[:,1::2] = 1-data    
    return dataOut



#-----------------------------------------------------------------
#       SOM2 class below (for 2 layer, inherits functionality of SOM)
#-----------------------------------------------------------------

class SOM2(SOM):
    def __init__(self,file_name='4750.csv', columns = [0,1,3,4,5,6], width = 6, height = 5, sigma=1.2, learning_rate=0.5, predThresh=np.inf, bmuFeatures=1, predMode='side' ):

        # essentially just a regular SOM, but puts the appropriate parameters
        SOM.__init__(self, file_name, columns, width, height, sigma,  learning_rate, predThresh, bmuFeatures, predMode )


#-----------------------------------------------------------------
#-----------------------------------------------------------------



class MiniSom:
    def __init__(self,x,y,input_len,sigma=1.0,learning_rate=0.5,random_seed=None, bmuFeatures=None):
        """
            Initializes a Self Organizing Maps.
            x,y - dimensions of the SOM
            input_len - number of the elements of the vectors in input
            sigma - spread of the neighborhood function (Gaussian), needs to be adequate to the dimensions of the map.
            (at the iteration t we have sigma(t) = sigma / (1 + t/T) where T is #num_iteration/2)
            learning_rate - initial learning rate
            (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T) where T is #num_iteration/2)
            random_seed, random seed to use.
        """
        if sigma >= x/2.0 or sigma >= y/2.0:
            warn('Warning: sigma is too high for the dimension of the map.')
        if random_seed:
            self.random_generator = random.RandomState(random_seed)
        else:
	    #print random_seed
            self.random_generator = random.RandomState(random_seed)
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.weights = self.random_generator.rand(x,y,input_len)*2-1 # random initialization
        self.weights = array([v/linalg.norm(v) for v in self.weights]) # normalization
        self.activation_map = zeros((x,y))
        self.neigx = arange(x)
        self.neigy = arange(y) # used to evaluate the neighborhood function
        self.neighborhood = self.gaussian
        self.x = x
        self.y = y 
        self.input_len = input_len
        self.tick = 0
        self.ticks = 0
        self.running = False
        self.paused = False
        self.bmuFeatures = bmuFeatures # for matching BMUs (finding winner) based on subset

    def _activate(self,x):
        """ Updates matrix activation_map, in this matrix the element i,j is the response of the neuron i,j to x """
        # if not matching based on subset of features
        # print( x.shape, self.weights.shape)
        if self.bmuFeatures == None: 
            s = subtract(x,self.weights) # x - w
        else:
            s = subtract(x[:-2],self.weights[:,:,:-2] )  # HERE MAGIC NUMBER NEED TO FIX!

        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.activation_map[it.multi_index] = linalg.norm(s[it.multi_index]) # || x - w ||
            it.iternext()
	
    def activate(self,x):
        """ Returns the activation map to x """
        self._activate(x)
        return self.activation_map


    def gaussian(self,c,sigma):
        """ Returns a Gaussian centered in c """
        d = 2*pi*sigma*sigma
        ax = exp(-power(self.neigx-c[0],2)/d)
        ay = exp(-power(self.neigy-c[1],2)/d)
        return outer(ax,ay) # the external product gives a matrix

    def diff_gaussian(self,c,sigma):
        """ Mexican hat centered in c (unused) """
        xx,yy = meshgrid(self.neigx,self.neigy)
        p = power(xx-c[0],2) + power(yy-c[1],2)
        d = 2*pi*sigma*sigma
        return exp(-(p)/d)*(1-2/d*p)

    def winner(self,x ):
        """ Computes the coordinates of the winning neuron for the sample x
		  """
        self._activate(x)
        return unravel_index(self.activation_map.argmin(),self.activation_map.shape)

    def update(self,x,win,t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
        """
        # print( 'in update, this is x going into quanitization error fct', x ) # HERE
        # q = self.quantization_error(x)  # originally
        # SR fixed a bug.  passing x to quantization_error function
        #   actually gets difference to individual features.
        # q = self.quantization_error([x])

        # eta(t) = eta(0) / (1 + t/T) 
        # keeps the learning rate nearly constant for the first T iterations and then adjusts it
        eta = self.learning_rate/(1+t/self.T)
        sig = self.sigma/(1+t/self.T) # sigma and learning rate decrease with the same rule
        g = self.neighborhood(win,sig)*eta # improves the performances
        it = nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])            
            # normalization
            self.weights[it.multi_index] = self.weights[it.multi_index] / linalg.norm(self.weights[it.multi_index])
            it.iternext()

           
    def update_gliozzi(self,x,win,t):
        """
            Updates the weights of the neurons.
            x - current pattern to learn
            win - position of the winning neuron for x (array or tuple).
            t - iteration index
        """
        # p. 722 Gliozzi et al. (2009)
        ## eta(t) = 1 / (1 + exp( - (quantization_error(d) - beta) / alpha))
        q = self.quantization_error(x)


        #alpha = 0.2
        #beta = .4
        #eta = 1 / (1 + exp( -( (q - beta) / alpha)))
        #eta = 0.25


 

	#From MIOsom_seqtrainBISPESATOESP1RETE.m
	#val_x1 = 0.7
	#val_x2 = 1
	#val_y1=0.5
	#val_y2=1
	#coeff_b= (log(val_y2) - log(val_y1))/ (val_x2 - val_x1)
	#coeff_a = val_y1 * exp(-coeff_b * val_x1)
        #eta = max(0.1, min(1, coeff_a*exp(coeff_b*sqrt(q))))

	# display x, q 	
        print "x = ", x, "qerr = ", q
	
	
	#From MIOsom_seqtrainBISPESATOESP1RETTOT.m
	#val_y1=0.5;
	#val_y2=1;
	#coeff_b= (log(val_y2) - log(val_y1))/ (val_x2 - val_x1);
	#coeff_a = val_y1 * exp(-coeff_b * val_x1);
	#lrate = 1/(1 + exp(-((sqrt(qerr)-val_05)/steepness)));
        r =   12
        val_05 = 0.1 + r * 0.05
        steepness = 0.1 + r * 0.05
        eta =  1 / (1 + exp(-((sqrt(q)-val_05)/steepness)));
        print "lrate = ", eta

        #sig = self.sigma
        #g = self.neighborhood(win,self.sigma_gliozzi[t])*eta # improves the performances

        g = self.neighborhood(win,self.sigma_gliozzi[t])*eta # improves the performances
        it = nditer(g, flags=['multi_index'])
        while not it.finished:
            # eta * neighborhood_function * (x-w)
            self.weights[it.multi_index] += g[it.multi_index]*(x-self.weights[it.multi_index])            
            # normalization
            #self.weights[it.multi_index] = self.weights[it.multi_index] / linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def quantization(self,data):
        """ Assigns a code book (weights vector of the winning neuron) to each sample in data. """
        q = zeros(data.shape)
        for i,x in enumerate(data):
            q[i] = self.weights[self.winner(x)]
        return q

    #def sample_weights_gliozzi(self,data):

    def weights_init_gliozzi(self,data):
      
        """ Initializes the weights of the SOM picking random samples from data """
        #it = nditer(self.activation_map, flags=['multi_index'])
        
        #self.sample_weights_gliozzi(data)
        #while not it.finished:
	    #for i in range(8):

	    #self.weights[it.multi_index][i] = self.random_generator.uniform(min(data[:,i]), max(data[:,i])) *0.3
	    #self.weights[it.multi_index][i] = data[:,i]
            ##self.weights[it.multi_index] = self.weights[it.multi_index]/linalg.norm(self.weights[it.multi_index])
            #it.iternext()

        it = nditer(self.activation_map, flags=['multi_index'])
        scale = 0.3
        while not it.finished:
            #for i in range(8):
            for i in range(self.input_len):
                col = data[:, i]
                self.weights[it.multi_index][i] = self.random_generator.uniform(min(col), max(col)) * scale
                 
            it.iternext()

    def random_weights_init(self,data):
        """ Initializes the weights of the SOM picking random samples from data """
        it = nditer(self.activation_map, flags=['multi_index'])
        while not it.finished:
            self.weights[it.multi_index] = data[int(self.random_generator.rand()*len(data)-1)]
            self.weights[it.multi_index] = self.weights[it.multi_index]/linalg.norm(self.weights[it.multi_index])
            it.iternext()

    def train_random(self,data,num_iteration):
        self.running = True
        """ Trains the SOM picking samples at random from data """
        self._init_T(num_iteration)      
        for iteration in range(num_iteration):
            rand_i = int(round(self.random_generator.rand()*len(data)-1)) # pick a random sample
            self.update(data[rand_i],self.winner(data[rand_i]),iteration)
            
        self.running = False

            
    def train_random_once(self,data):    
        self.running = True

        """ Trains the SOM picking samples at random from data """
        data_indices = range(len(data))
        self._init_T(len(data))      

        random.shuffle(data_indices)
        #print( data_indices ) # HERE

        for iteration, d in enumerate(data_indices):
            #print( data.shape ) #HERE
            self.update(data[d],self.winner(data[d]),iteration)
        self.running = False

            
    def train_gliozzi(self,data,num_iteration=1):    
        self.running = True

        """ Trains the SOM picking samples at random from data """
        data_indices = range(len(data))
        self.sigma_gliozzi = linspace(1.2, 0.8, num=len(data))
        self._init_T(len(data))      

        #while iteration < num_iteration:

        random.shuffle(data_indices)
        for iteration, d in enumerate(data_indices):
            self.update_gliozzi(data[d],self.winner(data[d]),iteration)
          
        self.running = False


    def train_batch(self,data,num_iteration):
        self.running = True
        """ Trains using all the vectors in data sequentially """
        self._init_T(len(data)*num_iteration)
        iteration = 0
        while iteration < num_iteration:
            idx = iteration % (len(data)-1)
            self.update(data[idx],self.winner(data[idx]),iteration)
            iteration += 1
        self.running = False

    def _init_T(self,num_iteration):
        """ Initializes the parameter T needed to adjust the learning rate """
        self.T = num_iteration/2 # keeps the learning rate nearly constant for the first half of the iterations

    def distance_map(self):
        """ Returns the average distance map of the weights.
            (Each mean is normalized in order to sum up to 1) """
        um = zeros((self.weights.shape[0],self.weights.shape[1]))
        it = nditer(um, flags=['multi_index'])
        while not it.finished:
            for ii in range(it.multi_index[0]-1,it.multi_index[0]+2):
                for jj in range(it.multi_index[1]-1,it.multi_index[1]+2):
                    if ii >= 0 and ii < self.weights.shape[0] and jj >= 0 and jj < self.weights.shape[1]:
                        um[it.multi_index] += linalg.norm(self.weights[ii,jj,:]-self.weights[it.multi_index])
            it.iternext()
        um = um/um.max()
        return um

    def activation_response(self,data):
        """ 
            Returns a matrix where the element i,j is the number of times
            that the neuron i,j have been winner.
        """
        a = zeros((self.weights.shape[0],self.weights.shape[1]))
        for x in data:
            a[self.winner(x)] += 1
        return a

    def quantization_error(self,data):
        """ 
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.            
        """
        
        # print( 'In quantization_error, this is data looped through', data)
        error = 0
        for x in data:
            # print( 'data vector being tested in quantization_error fct',x ) #HERE
            error += linalg.norm(x-self.weights[self.winner(x)])
        return error/len(data)
      
    def quantization_error_subset(self,data,n):
        """ 
            Returns the quantization error computed as the average distance between
            each input sample and its best matching unit.            
        """
        #n = cols -1
        
        tmp = data[0:n]
        data = tmp
        error = 0
        for x in data:
            error += linalg.norm(x-self.weights[self.winner(x)])
        return error/len(data)
  
    def win_map(self,data):
        """
            Returns a dictionary wm where wm[(i,j)] is a list with all the patterns
            that have been mapped in the position i,j.
        """
        winmap = defaultdict(list)
        for x in data:
            winmap[self.winner(x)].append(x)
        return winmap

'''
    def pertSomWeights( self, scale == None):
    	if scale == None:
    	scale = .5
    	print( 'Adding noise to SOM weights')

    	pertAmount = scale*(np.random.random_sample( self.som.weights.shape)-.5)
    	self.weights = self.weights + pertAmount

    def pertInputs( self,  widget=None, data=None ):
    	#if scale == None:
    	p = .2
        print( 'Making %f prop of inputs 0.5' %p)
    	#print( self.data.shape )

    	# randomly get indices to switch, then replace
    	noiseIndex = np.random.binomial(1,p, self.data.shape)  #ones at p proportion of samples
    	self.data[noiseIndex ==1 ] = .5
    	#print( self.data )
'''

### unit tests
from numpy.testing import assert_almost_equal, assert_array_almost_equal, assert_array_equal

class TestMinisom:
    def setup_method(self, method):
        self.som = MiniSom(5,5,1)
        for w in self.som.weights: # checking weights normalization
            assert_almost_equal(1.0,linalg.norm(w))
        self.som.weights = zeros((5,5)) # fake weights
        self.som.weights[2,3] = 5.0
        self.som.weights[1,1] = 2.0

    def test_gaussian(self):
        bell = self.som.gaussian((2,2),1)
        assert bell.max() == 1.0
        assert bell.argmax() == 12  # unravel(12) = (2,2)

    def test_win_map(self):
        winners = self.som.win_map([5.0,2.0])
        assert winners[(2,3)][0] == 5.0
        assert winners[(1,1)][0] == 2.0

    def test_activation_reponse(self):
        response = self.som.activation_response([5.0,2.0])
        assert response[2,3] == 1
        assert response[1,1] == 1

    def test_activate(self):
        assert self.som.activate(5.0).argmin() == 13.0  # unravel(13) = (2,3)
     
    def test_quantization_error(self):
        self.som.quantization_error([5,2]) == 0.0
        self.som.quantization_error([4,1]) == 0.5

    def test_quantization(self):
        q = self.som.quantization(array([4,2]))
        assert q[0] == 5.0
        assert q[1] == 2.0

    def test_random_seed(self):
        som1 = MiniSom(5,5,2,sigma=1.0,learning_rate=0.5,random_seed=1)
        som2 = MiniSom(5,5,2,sigma=1.0,learning_rate=0.5,random_seed=1)
        assert_array_almost_equal(som1.weights,som2.weights) # same initialization
        data = random.rand(100,2)
        som1 = MiniSom(5,5,2,sigma=1.0,learning_rate=0.5,random_seed=1)
        som1.train_random(data,10)
        som2 = MiniSom(5,5,2,sigma=1.0,learning_rate=0.5,random_seed=1)
        som2.train_random(data,10)
        assert_array_almost_equal(som1.weights,som2.weights) # same state after training

    def test_train_batch(self):
        som = MiniSom(5,5,2,sigma=1.0,learning_rate=0.5,random_seed=1)
        data = array([[4,2],[3,1]])
        q1 = som.quantization_error(data)
        som.train_batch(data,10)
        assert q1 > som.quantization_error(data)

    def test_train_random(self):
        som = MiniSom(5,5,2,sigma=1.0,learning_rate=0.5,random_seed=1)
        data = array([[4,2],[3,1]])
        q1 = som.quantization_error(data)
        som.train_random(data,10)
        assert q1 > som.quantization_error(data)

    def test_random_weights_init(self):
        som = MiniSom(2,2,2,random_seed=1)
        som.random_weights_init(array([[1.0,.0]]))
        for w in som.weights:
            assert_array_equal(w[0],array([1.0,.0]))



