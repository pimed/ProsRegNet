import SimpleITK as sitk
import json

class ParserStudyDict:
    def __init__(self,studyDict):
        self.dict = studyDict
  
        self.id                         = None
        self.exvivo_accession           = None
        self.invivo_accession           = None
        self.fixed_filename             = None
        self.fixed_segmentation_filename= None
        self.fixed_landmark1_filename   = None
        self.fixed_landmark2_filename   = None
        self.moving_type                = None
        self.moving_filename            = None
        self.moving_dict                = None

        self.T2_filename                = None
        self.ADC_filename               = None
        self.DWI_filename               = None
        
        self.SetFromDict()
        
    def SetFromDict(self):  
        try:
            self.fixed_filename                 = self.dict['fixed']
        except Exception as e:
            print(e)
            
        try:
            self.fixed_segmentation_filename    = self.dict['fixed-segmentation']
        except Exception as e:
            print(e)

            
        try:
            self.fixed_landmark1_filename        = self.dict['fixed-landmarks1']
        except Exception as e:
            print(e)

        try:
            self.fixed_landmark2_filename        = self.dict['fixed-landmarks2']
        except Exception as e:
            print(e)


        try:
            self.moving_type                    = self.dict['moving-type']

        except Exception as e:
            print(e)

        try:
            self.moving_filename                = self.dict['moving']
        except Exception as e:
            print(e)
            
        try:
            self.id                            = self.dict['id']
        except Exception as e:
            print(e)
        
        try:
            self.invivo_accession              = self.dict['invivo-accession']
        except Exception as e:
            print(e)
        
        try:
            self.exvivo_accession              = self.dict['exvivo-accession']
        except Exception as e:
            print(e)

        try:
            self.T2_filename                   = self.dict['T2w']
        except Exception as e:
            print(e)

        try:
            self.ADC_filename                  = self.dict['ADC']
        except Exception as e:
            print(e)

        try:
            self.DWI_filename                  = self.dict['DWI']
        except Exception as e:
            print(e)
   
    def ReadImage(self, fn):
        im = None
        if fn:
            try:
                im = sitk.ReadImage( fn )
            except Exception as e:
                print(e)
                print("Fixed image cound not be read from", fn)
                im = None
        else:
            print("Fixed filename is not available and fixed image cound't be read")
            im = None
            
        return im
            
    def ReadMovingImage(self):   
        
        if self.moving_type and self.moving_type.lower()=="stack":
            #print(self.moving_filename)
            with open(self.moving_filename) as f:
                self.moving_dict = json.load(f)
        else:
            self.moving_dict = None
        
        return self.moving_dict