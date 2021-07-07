import torch.nn as nn

class CNNFeatureExtractionModule(nn.Module):
    def __init__(self,output_size):
        super(CNNFeatureExtractionModule, self).__init__()  

        self.output_size = output_size

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )     
        self.out = nn.Linear(32 * 7 * 7, output_size)    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)       
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output