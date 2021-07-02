#include "cnn.h"
#include "normalize_input.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

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

    cnn(normalized_input, conv_weight, conv_bias, conv_output, relu_output, fc_input, fc_weight, fc_bias, output);

    // for (i = 0; i < 23; i++)
    // {
    //     printf("%f\n", output[i]);
    // }

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
    printf("%d is the prediction label.\n", index); //3
}