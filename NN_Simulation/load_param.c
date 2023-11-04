#include<stdio.h>
#include<stdlib.h>




int load_Pseudo_Random(double **Sobol_seq)
{
	int j;
	FILE* file;
	file=fopen("Pseudo_Random.txt", "r");
	for (j = 0; j < 256; j++)
		fscanf(file,"%lf",&Sobol_seq[0][j]);
	for (j = 0; j < 256; j++)
		fscanf(file, "%lf", &Sobol_seq[1][j]);

	
	fclose(file);
}

int load_Sobol(double **Sobol_seq)
{
	int j;
	FILE* file;
	file=fopen("Sobol.txt", "r");
	for (j = 0; j < 1024; j++)
		fscanf(file,"%lf",&Sobol_seq[0][j]);
	for (j = 0; j < 1024; j++)
		fscanf(file, "%lf", &Sobol_seq[1][j]);
	for (j = 0; j < 1024; j++)
		fscanf(file, "%lf", &Sobol_seq[2][j]);
	for (j = 0; j < 1024; j++)
		fscanf(file, "%lf", &Sobol_seq[3][j]);
	
	fclose(file);
}

int load_Pseudo_Sobol(double **Sobol_seq)
{
	int j;
	FILE* file;
	file=fopen("Pseudo_Sobol.txt", "r");
	for (j = 0; j < 256; j++)
		fscanf(file,"%lf",&Sobol_seq[0][j]);
	for (j = 0; j < 256; j++)
		fscanf(file, "%lf", &Sobol_seq[1][j]);

	
	fclose(file);
}

int load_Pseudo_Sobol_only_flip(double **Sobol_seq)
{
	int j;
	FILE* file;
	file=fopen("Pseudo_Sobol_only_flip.txt", "r");
	for (j = 0; j < 256; j++)
		fscanf(file,"%lf",&Sobol_seq[0][j]);
	for (j = 0; j < 256; j++)
		fscanf(file, "%lf", &Sobol_seq[1][j]);

	
	fclose(file);
}



int load_param
(
	int in_neuron, //The number of input neurons
	int out_neuron, // The number of output neurons
	double **weights, // Weight array with size out_neuron*in_neuron
	double *bias, // out_neuron biases
	char *filename_weight,
	char *filename_bias
)
{
	FILE* fp;
	int i,j;
	fp = fopen(filename_weight, "r");
	
	for (i = 0; i < out_neuron; i++)
	{
		//printf("i=%d\n",i);
		for (j = 0; j < in_neuron; j++)
		{
			fscanf(fp, "%lf", &weights[i][j]);
			//printf("");
		}
	}
	fclose(fp);
	

	fp = fopen(filename_bias, "r");
	for (i = 0; i < out_neuron; i++)
		fscanf(fp, "%lf", &bias[i]);
	fclose(fp);


	return 0;
}

int load_param_conv
(
	int m, // Kernel size is m*n
	int n,
	int num_in_channel, // Number of input channels
	int num_out_channel, // Number of output channels
	double ****weights, // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	double *bias, // num_out_channel biases
	char *filename_weight,
	char *filename_bias
)
{
	//printf("Begin!\n");
	//printf("%d %d %d %d \n",m,n,num_in_channel,num_out_channel);
	//printf("%s\n",filename_weight);
	FILE* fp;
	int i,j,k,ii,jj,kk;
	//printf("Begin!\n");
	//printf("%s\n",filename_weight);
	fp = fopen(filename_weight, "r");
	//printf("File!\n");
	for(i=0;i<num_out_channel;i++)
	{
		//if(num_in_channel==16 && num_out_channel==32)printf("i=%d\n",i);
		for(j=0;j<num_in_channel;j++)
		{
			//if(num_in_channel==16 && num_out_channel==32)printf("j=%d\n",j);
			for(k=0;k<m;k++)
			{
				for(ii=0;ii<n;ii++)
				{
					/*
					if(num_in_channel==16 && num_out_channel==32)
					{
						printf("(%d,%d,%d,%d)\n",i,j,k,ii);
						printf("%lf\n",weights[i][j][k][ii]);
					}
					*/
					fscanf(fp,"%lf",&weights[i][j][k][ii]);
					//if(strcmp(filename_weight,"param_model_cifar10/conv1_weights.txt")==0)
						//printf("%lf ",weights[i][j][k][ii]);
				}
			}
		}
	}
	/*			
	printf("num_out_channel=%d, num_in_channel=%d:\n",num_out_channel,num_in_channel);
	for(i=0;i<m;i++)
		for(j=0;j<n;j++)
			printf("%lf, ",weights[1][0][i][j]);
	*/
	fclose(fp);
	

	fp = fopen(filename_bias, "r");
	for (i = 0; i < num_out_channel; i++)
		fscanf(fp, "%lf", &bias[i]);
	fclose(fp);


	return 0;
}

int load_param_fusedSqueezeNetChen
(
	int *num_out_channel_conv, int *num_in_channel_conv, // Number of channels for the convolutional layers
	int *kernelH_conv,int *kernelW_conv, // Size of kernels of the convolutional layers
	int num_fire, // Number of fire modules
	int *s11, int *e11, int *e33, // Number of Squeeze 1*1 convolutional kernels, Expand 1*1 convolutional kernels, and Expand 3*3 convolutional kernels for the fire modules
	double *****weights_conv, // 4 convolutional layers which have been combined with BN (see Chen's TCASI)
	double **bias_conv, // 4 convolutional layers
	double ******weight_conv_fire, // convolutional layers for fire modules: 4*4*convolutional kernelH
	double ***bias_conv_fire // convolutional layers for fire modules
)
{
	
	char filename_weight[100],filename_bias[100];
	char filename[100];
	
	printf("Begin to load parameters:\n");
	// conv1
	strcpy(filename_weight,"param_fused_SqueezeNetChen/conv1_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/conv1_bias.txt");
	load_param_conv
	(
	kernelH_conv[0], // Kernel size is m*n
	kernelH_conv[0],
	num_in_channel_conv[0], // Number of input channels
	num_out_channel_conv[0], // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	/*
	printf("After loading the parameters of conv1:\n");
	for(i=0;i<kernelH[0];i++)
		for(j=0;j<kernelW[0];j++)
			printf("%lf, ",weights_conv[0][1][0][i][j]);
	*/
	printf("Parameters of conv1 have been successfully loaded!===========================\n");
	// conv2
	strcpy(filename_weight,"param_fused_SqueezeNetChen/conv2_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/conv2_bias.txt");
	load_param_conv
	(
	kernelH_conv[1], // Kernel size is m*n
	kernelH_conv[1],
	num_in_channel_conv[1], // Number of input channels
	num_out_channel_conv[1], // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of conv2 have been successfully loaded!===========================\n");
	// conv3
	strcpy(filename_weight,"param_fused_SqueezeNetChen/conv3_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/conv3_bias.txt");
	load_param_conv
	(
	kernelH_conv[2], // Kernel size is m*n
	kernelH_conv[2],
	num_in_channel_conv[2], // Number of input channels
	num_out_channel_conv[2], // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of conv3 have been successfully loaded!===========================\n");
	// conv4
	strcpy(filename_weight,"param_fused_SqueezeNetChen/conv4_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/conv4_bias.txt");
	load_param_conv
	(
	kernelH_conv[3], // Kernel size is m*n
	kernelH_conv[3],
	num_in_channel_conv[3], // Number of input channels
	num_out_channel_conv[3], // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of conv4 have been successfully loaded!===========================\n");
	
	// fire1
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire1_s11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire1_s11_bias.txt");
	//printf("Begin!\n");
	//printf("%s\n",filename_weight);
	//printf("s11=%d\n",s11[0]);
	//printf("%s\n",filename_bias);
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	96, // Number of input channels
	s11[0], // Number of output channels
	weight_conv_fire[0][0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[0][0], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire1_e11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire1_e11_bias.txt");
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	s11[0], // Number of input channels
	e11[0], // Number of output channels
	weight_conv_fire[0][1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[0][1], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire1_e33_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire1_e33_bias.txt");
	//printf("%s\n",filename_weight);
	//printf("%d %d\n",s11[0],e33[0]);
	//printf("%lf\n",weight_conv_fire[0][2][0][0][0][0]);
	load_param_conv
	(
	3, // Kernel size is m*n
	3,
	s11[0], // Number of input channels
	e33[0], // Number of output channels
	weight_conv_fire[0][2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[0][2], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of fire1 have been successfully loaded!===========================\n");
	
	// fire2
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire2_s11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire2_s11_bias.txt");
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	64, // Number of input channels
	s11[1], // Number of output channels
	weight_conv_fire[1][0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[1][0], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire2_e11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire2_e11_bias.txt");
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	s11[1], // Number of input channels
	e11[1], // Number of output channels
	weight_conv_fire[1][1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[1][1], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire2_e33_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire2_e33_bias.txt");
	load_param_conv
	(
	3, // Kernel size is m*n
	3,
	s11[1], // Number of input channels
	e33[1], // Number of output channels
	weight_conv_fire[1][2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[1][2], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of fire2 have been successfully loaded!===========================\n");
	
	// fire3
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire3_s11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire3_s11_bias.txt");
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	128, // Number of input channels
	s11[2], // Number of output channels
	weight_conv_fire[2][0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[2][0], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire3_e11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire3_e11_bias.txt");
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	s11[2], // Number of input channels
	e11[2], // Number of output channels
	weight_conv_fire[2][1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[2][1], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire3_e33_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire3_e33_bias.txt");
	load_param_conv
	(
	3, // Kernel size is m*n
	3,
	s11[2], // Number of input channels
	e33[2], // Number of output channels
	weight_conv_fire[2][2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[2][2], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of fire3 have been successfully loaded!===========================\n");
	
	// fire4
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire4_s11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire4_s11_bias.txt");
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	256, // Number of input channels
	s11[3], // Number of output channels
	weight_conv_fire[3][0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[3][0], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire4_e11_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire4_e11_bias.txt");
	load_param_conv
	(
	1, // Kernel size is m*n
	1,
	s11[3], // Number of input channels
	e11[3], // Number of output channels
	weight_conv_fire[3][1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[3][1], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	strcpy(filename_weight,"param_fused_SqueezeNetChen/fire4_e33_weights.txt");
	strcpy(filename_bias,"param_fused_SqueezeNetChen/fire4_e33_bias.txt");
	load_param_conv
	(
	3, // Kernel size is m*n
	3,
	s11[3], // Number of input channels
	e33[3], // Number of output channels
	weight_conv_fire[3][2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv_fire[3][2], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of fire4 have been successfully loaded!===========================\n");
	
	return 0;
}


int load_param_model_cifar10
(
	int *num_out_channel_conv, int *num_in_channel_conv, // Number of channels for the convolutional layers
	int *num_in_neuron,int *num_out_neuron,
	int *kernelH_conv,int *kernelW_conv, // Size of kernels of the convolutional layers
	double *****weights_conv, // 4 convolutional layers
	double **bias_conv, // 4 convolutional layers
	double ***weights_fc, // 2 FC layers
	double **bias_fc // 2 FC layers
)
{
	int i,j;
	char filename_weight[100],filename_bias[100];
	char filename[100];
	
	printf("Begin to load parameters:\n");
	// conv1
	strcpy(filename_weight,"param_model_cifar10/conv1_weights.txt");
	strcpy(filename_bias,"param_model_cifar10/conv1_bias.txt");
	load_param_conv
	(
	kernelH_conv[0], // Kernel size is m*n
	kernelH_conv[0],
	num_in_channel_conv[0], // Number of input channels
	num_out_channel_conv[0], // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	/*
	printf("After loading the parameters of conv1:\n");
	for(i=0;i<kernelH_conv[0];i++)
		for(j=0;j<kernelH_conv[0];j++)
			printf("%lf, ",weights_conv[0][0][0][i][j]);
	*/
	printf("Parameters of conv1 have been successfully loaded!===========================\n");
	// conv2
	strcpy(filename_weight,"param_model_cifar10/conv2_weights.txt");
	strcpy(filename_bias,"param_model_cifar10/conv2_bias.txt");
	load_param_conv
	(
	kernelH_conv[1], // Kernel size is m*n
	kernelH_conv[1],
	num_in_channel_conv[1], // Number of input channels
	num_out_channel_conv[1], // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of conv2 have been successfully loaded!===========================\n");
	// conv3
	strcpy(filename_weight,"param_model_cifar10/conv3_weights.txt");
	strcpy(filename_bias,"param_model_cifar10/conv3_bias.txt");
	load_param_conv
	(
	kernelH_conv[2], // Kernel size is m*n
	kernelH_conv[2],
	num_in_channel_conv[2], // Number of input channels
	num_out_channel_conv[2], // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of conv3 have been successfully loaded!===========================\n");
	// conv4
	strcpy(filename_weight,"param_model_cifar10/conv4_weights.txt");
	strcpy(filename_bias,"param_model_cifar10/conv4_bias.txt");
	load_param_conv
	(
	kernelH_conv[3], // Kernel size is m*n
	kernelH_conv[3],
	num_in_channel_conv[3], // Number of input channels
	num_out_channel_conv[3], // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of conv4 have been successfully loaded!===========================\n");
	
	// fc1 
	strcpy(filename_weight,"param_model_cifar10/fc1_weights.txt");
	strcpy(filename_bias,"param_model_cifar10/fc1_bias.txt");
	load_param(num_in_neuron[0],num_out_neuron[0],weights_fc[0],bias_fc[0],filename_weight,filename_bias);
	printf("Parameters of fc1 have been successfully loaded!===========================\n");
	// fc2
	strcpy(filename_weight,"param_model_cifar10/fc2_weights.txt");
	strcpy(filename_bias,"param_model_cifar10/fc2_bias.txt");
	load_param(num_in_neuron[1],num_out_neuron[1],weights_fc[1],bias_fc[1],filename_weight,filename_bias);
	printf("Parameters of fc2 have been successfully loaded!===========================\n");
	
	return 0;
}