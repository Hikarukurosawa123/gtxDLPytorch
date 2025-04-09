#import classes from other files 
from gtxDLClassAWSUtils import Utils
from HelperFunc import Helper
from ModelArchitecture import ModelInit
from DataImport import Operations
from siamese_pytorch import TinyModel

class DL(Utils, Helper, ModelInit, Operations, TinyModel):    
    # Initialization method runs whenever an instance of the class is initiated
    def __init__(self):
        super().__init__()
        self.bucket = '20240909-hikaru'
        print("inside operations: ", self.bucket)
        print("type :", type(self.bucket))
        isCase = 'Default'#input('Choose a case:\n Default\n Default4fx')
        self.case = isCase
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        self.isTesting = False
        self.AWS = False

        self.Model()


if __name__ == "__main__":
    #model = DL()
    #model.Model()
    #model.importData()
    pass

        
    
    
    
    