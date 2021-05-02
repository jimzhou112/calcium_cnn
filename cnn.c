/*-------------------------------------------------------

Filename: cnn.c

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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "cnn.h"
#include "normalize_input.h"

// Performs Convolution with 6x5x5 output
void convolution(
    const float input[IN_CHANNELS][IN_SIZE][IN_SIZE],
    const float weight[NUM_KERNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE], const float bias[NUM_KERNELS],
    float output[NUM_KERNELS][OUT_SIZE][OUT_SIZE])
{

    // Bias
    for (int i = 0; i < NUM_KERNELS; ++i)
    {
        for (int h = 0; h < OUT_SIZE; ++h)
        {
            for (int w = 0; w < OUT_SIZE; ++w)
                output[i][h][w] = bias[i];
        }
    }

    // Convolution
    for (int i = 0; i < OUT_SIZE; ++i)
    {
        for (int j = 0; j < OUT_SIZE; ++j)
        {
            for (int h = 0; h < NUM_KERNELS; ++h)
            {
                for (int w = 0; w < IN_CHANNELS; ++w)
                {
                    for (int p = 0; p < KERNEL_SIZE; ++p)
                    {
                        for (int q = 0; q < KERNEL_SIZE; ++q)
                            output[h][i][j] += weight[h][w][p][q] * input[w][i + p][j + q];
                    }
                }
            }
        }
    }
}

// Performs batch normalization, defined as y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
void batchnorm(const float input[NUM_KERNELS][OUT_SIZE][OUT_SIZE], float output[NUM_KERNELS][OUT_SIZE][OUT_SIZE], const float bn_weight[NUM_KERNELS], const float bn_bias[NUM_KERNELS], const float bn_running_mean[NUM_KERNELS], const float bn_running_var[NUM_KERNELS])
{
    // Default value of 1e-5
    float eps = 0.00001;

    for (int i = 0; i < NUM_KERNELS; i++)
    {
        for (int j = 0; j < OUT_SIZE; j++)
        {
            for (int k = 0; k < OUT_SIZE; k++)
            {
                output[i][j][k] = ((input[i][j][k] - bn_running_mean[i]) / sqrtf(eps + bn_running_var[i])) * bn_weight[i] + bn_bias[i];
            }
        }
    }
}

// Applies element-wise rectified linear unit function
void relu(const float input[NUM_KERNELS][OUT_SIZE][OUT_SIZE], float output[NUM_KERNELS][OUT_SIZE][OUT_SIZE])
{
    for (int i = 0; i < NUM_KERNELS; i++)
    {
        for (int j = 0; j < OUT_SIZE; j++)
        {
            for (int k = 0; k < OUT_SIZE; k++)
            {
                output[i][j][k] = max(input[i][j][k], 0);
            }
        }
    }
}

// Applies a linear transformation of the form output = xA^T + b
void fc(const float input[150], const float weight[NUM_LABELS][150], const float bias[NUM_LABELS], float output[NUM_LABELS])
{
    for (int i = 0; i < NUM_LABELS; i++)
    {
        output[i] = bias[i];
    }
    for (int i = 0; i < NUM_LABELS; i++)
    {
        for (int j = 0; j < 150; j++)
        {
            output[i] += weight[i][j] * input[j];
        }
    }
}

int main()
{
    int i, j, k;
    FILE *ifp, *ofp;
    char file_name[100];
    char *line = NULL;
    size_t len = 0;
    const char *s = " ";
    char *token = NULL;

    // Input arrays
    struct timeval t1, t2, tr;
    static int input[IN_CHANNELS][IN_SIZE][IN_SIZE];
    static float normalized_input[IN_CHANNELS][IN_SIZE][IN_SIZE];

    // Convolution arrays
    static float conv_weight[NUM_KERNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE];
    static float conv_bias[NUM_KERNELS];
    static float conv_output[NUM_KERNELS][OUT_SIZE][OUT_SIZE];

    // Batch normalization arrays
    static float bn_weight[NUM_KERNELS];
    static float bn_bias[NUM_KERNELS];
    static float bn_running_mean[NUM_KERNELS];
    static float bn_running_var[NUM_KERNELS];
    static float bn_output[NUM_KERNELS][OUT_SIZE][OUT_SIZE];

    // Relu arrays
    static float relu_output[NUM_KERNELS][OUT_SIZE][OUT_SIZE];

    // Fc arrays
    static float fc_input[150];
    static float fc_weight[NUM_LABELS][150];
    static float fc_bias[NUM_LABELS];

    // Output
    static float output[NUM_LABELS];

    // Load inputs
    sprintf(file_name, "./%s/input.txt", DATASET);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File input.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < IN_SIZE; i++)
    {
        for (j = 0; j < IN_SIZE; j++)
        {
            if (fscanf(ifp, "%u", &input[0][i][j]) != 1)
            {
                printf("File input.txt is not of correct length or format\n");
                return -1;
            }
        }
    }

    // Load conv1weights
    sprintf(file_name, "./%s/conv1weight.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File conv1weight.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_KERNELS; i++)
    {
        for (j = 0; j < KERNEL_SIZE; j++)
        {
            for (k = 0; k < KERNEL_SIZE; k++)
            {
                if (fscanf(ifp, "%f", &conv_weight[i][0][j][k]) != 1)
                {
                    printf("File conv1weight.txt is not of correct length or format\n");

                    return -1;
                }
            }
        }
    }

    // Load conv1bias
    sprintf(file_name, "./%s/conv1bias.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File conv1bias.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_KERNELS; i++)
    {
        if (fscanf(ifp, "%f", &conv_bias[i]) != 1)
        {
            printf("File conv1bias.txt is not of correct length or format\n");

            return -1;
        }
    }

    // Load bn1weight
    sprintf(file_name, "./%s/bn1weight.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File bn1weight.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_KERNELS; i++)
    {
        if (fscanf(ifp, "%f", &bn_weight[i]) != 1)
        {
            printf("File bn1weight.txt is not of correct length or format\n");
            return -1;
        }
    }

    // Load bn1bias
    sprintf(file_name, "./%s/bn1bias.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File bn1bias.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_KERNELS; i++)
    {
        if (fscanf(ifp, "%f", &bn_bias[i]) != 1)
        {
            printf("File bn1bias.txt is not of correct length or format\n");
            return -1;
        }
    }

    // Load bn1running_mean
    sprintf(file_name, "./%s/bn1running_mean.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File bn1running_mean.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_KERNELS; i++)
    {
        if (fscanf(ifp, "%f", &bn_running_mean[i]) != 1)
        {
            printf("File bn1running_mean.txt is not of correct length or format\n");
            return -1;
        }
    }

    // Load bn1running_var
    sprintf(file_name, "./%s/bn1running_var.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File bn1running_var.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_KERNELS; i++)
    {
        if (fscanf(ifp, "%f", &bn_running_var[i]) != 1)
        {
            printf("File bn1running_var.txt is not of correct length or format\n");
            return -1;
        }
    }

    // Load fc1weight
    sprintf(file_name, "./%s/fc1weight.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File fc1weight.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_LABELS; i++)
    {
        for (j = 0; j < 150; j++)
        {
            if (fscanf(ifp, "%f", &fc_weight[i][j]) != 1)
            {
                printf("File fc1weight.txt is not of correct length or format\n");
                return -1;
            }
        }
    }

    // Load fc1bias
    sprintf(file_name, "./%s/fc1bias.txt", MODEL);
    if (!(ifp = fopen(file_name, "r")))
    {
        printf("File fc1bias.txt cannot be opened for read.\n");
        return -1;
    }
    for (i = 0; i < NUM_LABELS; i++)
    {
        if (fscanf(ifp, "%f", &fc_bias[i]) != 1)
        {
            printf("File fc1bias.txt is not of correct length or format\n");
            return -1;
        }
    }

    // Transforms PIL image input to normalized input for CNN
    normalize_input(input, normalized_input);

    // Start timer
    gettimeofday(&t1, NULL);

    // Performs Convolution with 6x5x5 output
    convolution(normalized_input, conv_weight, conv_bias, conv_output);

    // Performs batch normalization, defined as y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    batchnorm(conv_output, bn_output, bn_weight, bn_bias, bn_running_mean, bn_running_var);

    // Applies element-wise rectified linear unit function
    relu(bn_output, relu_output);

    // Reshape to 1D array
    for (i = 0; i < NUM_KERNELS; i++)
    {
        for (j = 0; j < OUT_SIZE; j++)
        {
            for (k = 0; k < OUT_SIZE; k++)
            {
                fc_input[i * 25 + j * 5 + k] = relu_output[i][j][k];
            }
        }
    }

    // Applies a linear transformation of the form output = xA^T + b
    fc(fc_input, fc_weight, fc_bias, output);

    for (i = 0; i < 23; i++)
    {
        printf("%f\n", output[i]);
    }

    // Stop timer
    gettimeofday(&t2, NULL);
    timersub(&t1, &t2, &tr);
    printf("Inference finished. \n%.5f sec elapsed.\n",
           -tr.tv_sec - (double)tr.tv_usec / 1000000.0);

    // Outputs the prediction label
    int index = 0;
    float max = output[index];
    for (i = 0; i < NUM_LABELS; i++)
    {
        if (output[i] > max)
        {
            max = output[i];
            index = i;
        }
    }
    printf("%d is the prediction label.\n", index);
}