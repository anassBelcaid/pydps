
"""
Special Class to perform a classification on the 
neuroblastoma segmentation problem.

"Learning smooth models of copy number profiles using break points annotations
Hocking et al (2013)
"

The problem is  different to the general framework as the annotation
only contain the existence or absence, not the position of the breakpoint.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydps.datasets
import warnings
from os.path import dirname,abspath
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection



#root folder to access file
root_fol=dirname(abspath(pydps.datasets.__file__))



#Ignore warning
warnings.filterwarnings("ignore")

class Neuroblastoma(object):

    """Dataset to load the annotated segments  from the dataset neuroblastoma"""

    def __init__(self):
        """
        Load the dataset
        """
        #unlike the first version we load the merged dataset
        self.profiles = pd.read_pickle(root_fol+"/neuroblastoma.bz2")

        self.num_patients = self.profiles['profile.id'].max()
    def __len__(self):
        """
        Lenght of the dataset
        """

        return self.num_patients
    
    def patient_sample(self, index, allAnnotations=False):
        """Function to get the annotated chromosomes samples for patient self.index
        :index: index of the patient in the dataset
        :allAnnotations: get all the annotation(even those without ground truth)
        # TODO: explain the return value #
        :returns: 
        """

        assert(index <self.num_patients), "invalid index for patient"

        #getting the chromosomes of the patient
        id_column = self.profiles['profile.id']
        patient = self.profiles[id_column == index]

        #sorting the data by position
        patient.sort_values(by =['position','chromosome'],inplace=True)
        
        #dropping id as is redendant for a single patient
        patient.drop(['profile.id'], axis= 1, inplace= True)

        
        #return dictionnary with all the data
        return self._chromosomeSplit(patient)


    
    def _chromosomeSplit(self, Frame):
        """Split the DataFrame (Frame) by chromosome

        :Frame: Pandas dataframe containing a patient infomration
        :returns: dictionnary (choromosome: array, annotation)

        """

        #dictionnary of each chromosome
        chrom_dict = {}

        for chrom, Serie in Frame.groupby(by = ['chromosome']):
            #getting the log ratio
            measure = Serie.logratio.values
            
            #getting the annotation
            annoation = Serie.annotation.unique()[0]

            #appending the annotation
            chrom_dict[chrom] = (measure, annoation)

        return chrom_dict
    
    def plot_patient_data(self, index,ax,allAnnotations=True,legend=False):
        """plot the annotated data for a patient

        :index: index of the patient
        :returns: plot the data

        """
        data = self.patient_sample(index,allAnnotations)

        print(data.keys())
        #dim of each chromsome
        chroms = ['1','2','3','4','5','6','7','8','9'\
                ,'10','11','12','13','14','15'\
                ,'16','17','18','19','20','21','22','X','Y']
        # print(data.keys())
        dimensions = [len(data[v][0]) for v in chroms if v in data.keys()]

        print("dimensions are:",dimensions)

        #joining the array by chromosome
        signal = np.hstack((data[v][0] for v in chroms))
        indexed_data = np.zeros((len(signal),2))
        indexed_data[:,0], indexed_data[:,1] = np.arange(len(signal)),signal

        # np.savetxt("patient"+str(index)+".csv",\
        #         indexed_data[::3,:],header="pos,log",comments="",\
        #         delimiter=',')
        # ymin,ymax= np.min(signal),np.max(signal)


        
        #plotting the data
        ax.plot(signal,lw=0.8,alpha=0.5,color='C1')
        ax.set_xticks([])
        ax.set_xlabel('Chromosome')


        #rectangles
        normal_collection = []
        break_collection  = []

        x = 0
        #chromosome label dictionnary
        bbox=dict(facecolor='C3', alpha=0.5)

        for (i,chrom) in enumerate(chroms): 
            sig,annot = data[chrom]

            #Creating the rectangle
            rect=Rectangle((x,-2),dimensions[i],5)
            x +=dimensions[i]
            ax.vlines(x,-1,1,lw=2)
            if(legend):
                ax.text(x-dimensions[i]//2,ymax,str(chrom),fontsize=16,bbox=bbox)



            #adding the rectangle
            if(annot==0):
                normal_collection.append(rect)
            elif(annot==np.nan):
                pass
            else:
                break_collection.append(rect)

        #patching the normal collection 
        normal_collection = PatchCollection(normal_collection,facecolor='green'\
                    ,alpha=0.3,label='C5')

        #patching the break 
        break_collection = PatchCollection(break_collection,facecolor='C0'\
                    ,alpha=0.3)

        #adding the collection
        ax.add_collection(break_collection)
        ax.add_collection(normal_collection)


if __name__ == "__main__":

    Data = Neuroblastoma()

    #plotting one patient
    patient_index = 1
    patient = Data.patient_sample(patient_index)

    for chrom in patient:

        print(chrom)
        
        measure, annotation = patient[chrom]
        print("shape is ", measure.shape)
        print("annotation is ", annotation)

    fig,ax = plt.subplots(1,1)
    Data.plot_patient_data(1,ax)
    plt.show()

