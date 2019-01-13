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
        """Load the dataset"""

        #loading the profiles
        self.profiles = pd.read_csv(root_fol+"/neuroblastoma_profiles.csv"\
                ,index_col='index')

        #loading the annotations
        self.annotations = pd.read_csv(root_fol+"/neuroblastoma_annotations.csv"\
                ,index_col='index')

        self.num_patients = self.profiles['profile.id'].max()



    
    
    def patient_sample(self, index):
        """get the the annotated crhomosome samples for a patient

        :index:  Index for the patient
        :returns: TODO

        """
        assert(index<self.num_patients), "index must be inferior to the maxim\
                dataset size"

        #reducing the dataset
        patient = self.profiles[self.profiles['profile.id']==index]

        patient_anno = self.annotations[self.annotations['profile.id']==index]

        #adding the annotation
        patient['annotation']='empty'


        #adding the annoations in annotatins
        chromosome = self.profiles['chromosome']
    
        for (chrom,val) in patient_anno[['chromosome','annotation']].values:
            patient.loc[chromosome==str(chrom),'annotation']= val

        #dropping empty values
        patient=patient[patient.annotation!='empty']
        
        #sorting the dataset
        patient.sort_values(['chromosome','position'])

        #dropping columns
        patient=patient[['chromosome','logratio','annotation']]

        return self._chromosomeSplit(patient.values)

    def _chromosomeSplit(self, array):
        """Split the data numpy array on the chromosome values

        :array:  numpy array containting the annotation values on each
        chromosome
        :returns: dictionnary {chromosome : (logration array, annotation)}
        """
        
        #empty dictinnary
        chrom_dict = {}


        #getting the unique keys 
        chromosomes = np.unique(array[:,0])
        
        #looping on each chromosome
        for chrom in chromosomes:
            #mask on the unique chromosome
            mask = array[:,0]== chrom

            #reduction to a signle chromosome
            red = array[mask]

            # adding the entry
            chrom_dict[chrom]=(red[:,1],red[0,2])
            

        return chrom_dict

    def plot_patient_data(self, index):
        """plot the annotated data for a patient

        :index: index of the patient
        :returns: plot the data

        """
        data = self.patient_sample(index)

        #dim of each chromsome
        dimensions = [len(data[v][0]) for v in data]

        #joining the array by chromosome
        signal = np.hstack((data[v][0] for v in data))
        ymin,ymax= np.min(signal),np.max(signal)


        
        #plotting the data
        fig,ax = plt.subplots(1,1)
        ax.plot(signal)


        #rectangles
        normal_collection = []
        break_collection  = []

        x = 0

        for (i,chrom) in enumerate(data): 
            sig,annot = data[chrom]

            #Creating the rectangle
            rect=Rectangle((x,-2),dimensions[i],3)
            x +=dimensions[i]
            ax.vlines(x,ymin,ymax,lw=4)


            #adding the rectangle
            if(annot=='normal'):
                normal_collection.append(rect)
            else:
                break_collection.append(rect)

        #patching the normal collection 
        normal_collection = PatchCollection(normal_collection,facecolor='green'\
                    ,alpha=0.2)

        #patching the break 
        break_collection = PatchCollection(break_collection,facecolor='red'\
                    ,alpha=0.2)

        #adding the collection
        ax.add_collection(break_collection)
        ax.add_collection(normal_collection)
        plt.show()







if __name__ == "__main__":
    Data = Neuroblastoma()
    Data.plot_patient_data(1)

        

