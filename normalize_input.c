/*-------------------------------------------------------

Filename: normalize_input.c

Transforms PIL image input to normalized input for CNN. 

Similar in behavior to TORCHVISION.TRANSFORMS (https://pytorch.org/vision/stable/transforms.html)

First convert PIL Image in the range [0, 255] to the range [0.0, 1.0].

Then, normalize the image with its mean and standard deviation.

---------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include "normalize_input.h"
#include <math.h>
#include "cnn.h"

void normalize_input(const int input[IN_CHANNELS][IN_SIZE][IN_SIZE], float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE])
{
    float mean = 0.9521;
    float sd = 0.0307;
    for (int i = 0; i < IN_SIZE; i++)
    {
        for (int j = 0; j < IN_SIZE; j++)
        {
            normalized_input[0][i][j] = (256.0 - (float)input[0][i][j]) / 255.0;
        }
    }
    for (int i = 0; i < IN_SIZE; i++)
    {
        for (int j = 0; j < IN_SIZE; j++)
        {
            normalized_input[0][i][j] = (normalized_input[0][i][j] - mean) / sd;
        }
    }
    // for (int i = 0; i < IN_SIZE; i++)
    // {
    //     for (int j = 0; j < IN_SIZE; j++)
    //     {
    //         printf("%f\n", normalized_input[0][i][j]);
    //     }
    // }
    // exit(1);
}
