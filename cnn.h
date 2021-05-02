/*-------------------------------------------------------

Filename: cnn.h

Takes in calcium image (./Calcium/input.txt) and outputs prediction label.

Model based on following PyTorch model:

*******************************************
class Simplenet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 6, 3, 1)
        self.bn = nn.BatchNorm2d(6)
        self.relu = nn.ReLU(inplace=False)
        self.fc1 = nn.Linear(150, 23)
        
    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
*******************************************

---------------------------------------------------------*/
#ifndef CNN_H
#define CNN_H

// Baseline model
#define FP_MODEL

// Compressed model
// #define COMPRESSED_MODEL

#ifdef FP_MODEL
#define MODEL "FP_model"
#endif

#ifdef COMPRESSED_MODEL
#define MODEL "compressed_model"
#endif

#define IN_SIZE 7
#define IN_CHANNELS 1
#define KERNEL_SIZE 3
#define NUM_KERNELS 6
#define NUM_LABELS 23
#define OUT_SIZE IN_SIZE - KERNEL_SIZE + 1
#define DATASET "Calcium"

#define max(x, y) (((x) >= (y)) ? (x) : (y))

#endif