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


    def __len__(self):

        return self.num_patients
    
    
    def patient_sample(self, index,allAnnotations=False):
        """get the the annotated crhomosome samples for a patient

        :index:  Index for the patient
        :allAnnotations: keep all chromosome even those without annotations
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
        if(not allAnnotations):
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

    def plot_patient_data(self, index,ax,allAnnotations=False,legend=False):
        """plot the annotated data for a patient

        :index: index of the patient
        :returns: plot the data

        """
        data = self.patient_sample(index,allAnnotations)

        #dim of each chromsome
        chroms = ['1','2','3','4','5','6','7','8'\
                ,'9','10','11','12','13','14','15'\
                ,'16','17','18','19','20','21','22','X','Y']
        # print(data.keys())
        dimensions = [len(data[v][0]) for v in chroms]
        print("dimensions are:",dimensions)

        #joining the array by chromosome
        signal = np.hstack((data[v][0] for v in chroms))
        indexed_data = np.zeros((len(signal),2))
        indexed_data[:,0], indexed_data[:,1] = np.arange(len(signal)),signal

        np.savetxt("patient"+str(index)+".csv",\
                indexed_data[::3,:],header="pos,log",comments="",\
                delimiter=',')
        ymin,ymax= np.min(signal),np.max(signal)


        
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
            ax.vlines(x,ymin,ymax,lw=2)
            if(legend):
                ax.text(x-dimensions[i]//2,ymax,str(chrom),fontsize=16,bbox=bbox)



            #adding the rectangle
            if(annot=='normal'):
                normal_collection.append(rect)
            elif(annot=='empty'):
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

        #setting the collections






if __name__ == "__main__":
    Data = Neuroblastoma()
    fig,ax = plt.subplots(1,1)
    Data.plot_patient_data(13,ax)
    plt.show()

        

