import json

class ParserRegistrationJson:
    def __init__(self,filename):
        self.filename   = filename
        self.dict       = None
        self.version    = None
        self.method     = None
        self.studies    = None
        self.ToProcess  = None
        self.study_filenames = None
        self.output_path= None
        self.do_affine  = True
        self.do_deformable = None
        self.fast_execution= None
        self.use_imaging_constraints = None
        self.do_reconstruction = None
                
        self.ReadJson()
        
        
    def ReadJson(self):
        with open(self.filename) as f:
            self.dict = json.load(f)
        try:
            self.version = self.dict['version']
        except Exception as e:
            print(e)
        
        
        try: 
            self.method  = self.dict['method']
            try:
                self.do_affine = self.dict['method']['do_affine']
            except Exception as e:
                print(e)
            try:
                self.do_deformable = self.dict['method']['do_deformable']
            except Exception as e:
                print(e)
            try:
                self.do_reconstruction = self.dict['method']['do_reconstruction']
            except Exception as e:
                print(e)
            try:
                self.fast_execution = self.dict['method']['fast_execution']
            except Exception as e:
                print(e)

            try:
                self.use_imaging_constraints = self.dict['method']['use_imaging_constraints']
            except Exception as e:
                print(e)

        except Exception as e:
            print(e)

            
        try: 
            self.study_filenames = self.dict['studies']
            self.studies = {}
            for s in self.study_filenames:
                fn = self.study_filenames[s]
                try:
                    print("Reading", s, "Study Json",fn)
                    with open(fn) as fs:
                        studyDict = json.load(fs)
                        
                    self.studies[s]=studyDict
                except Exception as ee:
                    
                    print(ee)
        except Exception as e:
            print(e)
            
        try: 
            self.output_path  = self.dict['output_path']
        except Exception as e:
            print(e)
        
        try: 
            self.ToProcess  = self.dict['studies2process']
        except Exception as e:
            print(e)


        
    def PrintJson(self):  
        for d in self.dict:
            print(d,self.dict[d])
        