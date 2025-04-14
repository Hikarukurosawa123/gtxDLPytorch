#import classes from other files 
from gtxDLClassAWSUtils import Utils
from HelperFunc import Helper
from ModelArchitecture import ModelInit
from DataImport import Operations

class DL(Utils, Helper, ModelInit, Operations):    
    # Initialization method runs whenever an instance of the class is initiated
    def __init__(self):
        super().__init__()
        self.bucket = '20240909-hikaru'
        
        isCase = 'Default'#input('Choose a case:\n Default\n Default4fx')
        self.case = isCase
        self.params = self.Defaults(self.case) # Define some default parameters based on the selected case
        self.isTesting = False
        self.AWS = False

        #choose whether to run pytorch or tensorflow 
        self.run_torch = 1


if __name__ == "__main__":

    pass

        
    
    
    
    