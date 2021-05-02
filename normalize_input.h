/*-------------------------------------------------------

Filename: normalize_input.h

Transforms PIL image input to normalized input for CNN. 

Similar in behavior to TORCHVISION.TRANSFORMS (https://pytorch.org/vision/stable/transforms.html)

First convert PIL Image in the range [0, 255] to the range [0.0, 1.0].

Then, normalize the image with its mean and standard deviation.

---------------------------------------------------------*/

#ifndef NORMALIZEINPUT_H
#define NORMALIZEINPUT_H

#include "cnn.h"
void normalize_input(const int input[IN_CHANNELS][IN_SIZE][IN_SIZE], float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE]);

#endif
