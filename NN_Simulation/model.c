#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<malloc.h>

#include"model.h"
#include"quan.h"

int DNN_3layers
(
	
	double *test_data,
	int num_input, // Number of input pixels
	int num_layer, // Number of layers, excluding the input layer
	int *num_neuron,// Numbers of neurons for different layers, excluding the input layer
	double **bias,
	double ***weights,
	double **out_fc // Output of different layers
)
{ 
	int i,j;
	double max;
	int in_neuron;
	double *input;
	int prediction;
	//printf("weights[0][0][0]=%lf\n",weights[0][0][0]);
	for(i=0;i<num_layer;i++)
	{
		//printf("%d\n",i);
		if(i==0)
		{
			in_neuron=num_input;
			input=test_data;
		}
		else
		{
			in_neuron=num_neuron[i-1];
			input=out_fc[i-1];
		}
		//printf("FC begins\n");
		FC(in_neuron,num_neuron[i],weights[i],bias[i],input,out_fc[i]);
		//printf("FC ends\n");
		if(i==num_layer-1)
		{
			// Make prediction
			max=-99999999;
			for(j=0;j<num_neuron[i];j++)
			{
				
				if(out_fc[i][j]>max)
				{
					max=out_fc[i][j];
					prediction=j;
				}
				//if(i==2)printf("%d: fc3=%lf,max=%lf,pred=%d\n",j,out_fc[i][j],max,prediction);
			}
		}
		else
			ReLU(num_neuron[i],out_fc[i],out_fc[i]);
		if(i==0)
		{
			/*
			for(j=0;j<num_neuron[i];j++)
				printf("%d: %lf\n",j,out_fc[i][j]);
			
			FILE *fp=fopen("fc1_out.txt","w");
			for(j=0;j<num_neuron[i];j++)
				fprintf(fp,"%.6f\n",out_fc[i][j]);
			fclose(fp);
			*/
		}
		if(i==2)
		{
			
			//for(j=0;j<num_neuron[i];j++)
				//printf("%d: %lf\n",j,out_fc[i][j]);
			
		}
	}
	
	return prediction;
}

int LeNet5
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*32); // Padding=2
	for(i=0;i<32;i++)
		data[0][i]=(double *)malloc(sizeof(double)*32);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*6);
	for(i=0;i<6;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*28);
		for(j=0;j<28;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*28);
		}
	}
	
	// padding=2
	for(i=0;i<32;i++)
	{
		for(j=0;j<32;j++)
		{
			if(i<2 || i>29 || j<2 || j>29)
				data[0][i][j]=0;
			else
				data[0][i][j]=test_data[i-2][j-2];
		}
	}
	
	
	Conv // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	32, // Height of input feature map
	32, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	6, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	/*
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
			printf("%lf ",output_conv1[0][i][j]);
		}
		printf("\n");
	}
	*/
	
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*6);
	for(i=0;i<6;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*14);
		for(j=0;j<14;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*14);
		}
	}
		
	max_pooling
	(
	28, // Height of input feature map
	28, // Width of input feature map
	2,
	2,
	14, // Height of output feature map: H_out=H/strideH
	14, // Width of output feature map: W_out=W/strideW
	6, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	// ReLU
	for(i=0;i<6;i++)
	{
		for(j=0;j<14;j++)
		{
			for(k=0;k<14;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}
	
	/*
	for(j=0;j<14;j++)
	{
		for(k=0;k<14;k++)
		{
			printf("%lf ",output_pooling1[0][j][k]);
		}
		printf("\n");
	}
	*/
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*16);
	for(i=0;i<16;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*10);
		for(j=0;j<10;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*10);
		}
	}
	
	
	Conv // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	14, // Height of input feature map
	14, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	6, // Number of input channels
	16, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*16);
	for(i=0;i<16;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*5);
		for(j=0;j<5;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*5);
		}
	}
		
	max_pooling
	(
	10, // Height of input feature map
	10, // Width of input feature map
	2,
	2,
	5, // Height of output feature map: H_out=H/strideH
	5, // Width of output feature map: W_out=W/strideW
	16, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<16;i++)
	{
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}
	
	// FC1
	double input_fc1[400],output_fc1[120];
	ii=0;
	for(i=0;i<16;i++)
	{
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC // Fully connection
	(
	400, //The number of input neurons
	120, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	
	
	//ReLU
	for(i=0;i<120;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[84];
	
	
	FC // Fully connection
	(
	120, //The number of input neurons
	84, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	
	//ReLU
	for(i=0;i<84;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	
	FC // Fully connection
	(
	84, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3
	);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<32;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<6;i++)
	{
		for(j=0;j<28;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<6;i++)
	{
		for(j=0;j<14;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<16;i++)
	{
		for(j=0;j<10;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<16;i++)
	{
		for(j=0;j<5;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}


int LeNet5_Date19
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];

	
	
	Conv // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	/*
	for(i=0;i<24;i++)
	{
		for(j=0;j<24;j++)
		{
			printf("%lf ",output_conv1[1][i][j]);
		}
		printf("\n");
	}
	*/
	
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}
	/*
	printf("conv1:\n");
	for(j=0;j<12;j++)
	{
		for(k=0;k<12;k++)
		{
			printf("%lf ",output_pooling1[1][j][k]);
		}
		printf("\n");
	}
	*/
	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	
	
	Conv // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}
	/*
	for(j=0;j<4;j++)
	{
		for(k=0;k<4;k++)
		{
			printf("%lf ",output_pooling2[0][j][k]);
		}
		printf("\n");
	}
	*/
	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[500];
	
	
	FC // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	
	FC // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3
	);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}



int LeNet5_Date19_sobol
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;



	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];

	
	
	Conv_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	Sobol_seq,
	len
	);
	/*
	for(i=0;i<24;i++)
	{
		for(j=0;j<24;j++)
		{
			printf("%lf ",output_conv1[1][i][j]);
		}
		printf("\n");
	}
	*/
	
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}
	/*
	printf("conv1:\n");
	for(j=0;j<12;j++)
	{
		for(k=0;k<12;k++)
		{
			printf("%lf ",output_pooling1[1][j][k]);
		}
		printf("\n");
	}
	*/
	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	
	
	Conv // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}
	/*
	for(j=0;j<4;j++)
	{
		for(k=0;k<4;k++)
		{
			printf("%lf ",output_pooling2[0][j][k]);
		}
		printf("\n");
	}
	*/
	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[500];
	
	
	FC // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	
	FC // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3
	);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}



int LeNet5_Date19_quan
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];


	// Quantiza the input data
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
			//if(data[0][i][j]!=quan_i_d(data[0][i][j], 0,0.875, 8))
				//printf("data[%d][%d]=%lf, quan=%lf\n",i,j,data[0][i][j],quan_i_d(data[0][i][j], 0,0.875, 8));
			data[0][i][j]=quan_i_d(data[0][i][j], 0,0.99609375, 256);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.9375, 16);
		}
	}
	
	
	Conv_quan // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99609375, 256
	);
	
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
		{
			for(k=0;k<24;k++)
			{
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 0,0.9921875, 128);
			}
		}
	}
	
	
	double max=0;
	//printf("max=%lf,",max);
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);

	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}

	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	
	Conv_quan // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99951171875, 2048
	);
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			for(k=0;k<8;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
			}
		}
	}
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}

	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}	
	
	
	FC_quan // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	0,0.9921875, 128,
	1,0.99951171875, 2048
	);
	

	//quantization
	for(i=0;i<800;i++)
		output_fc1[i]=quan_i_d(output_fc1[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[500];
	
	
	//quantization
	for(i=0;i<500;i++)
	{
		//bias_fc[1][i] = quan_i_d(bias_fc[1][i], 0,0.99609375, 256);
		for(j=0;j<800;j++)
		{
			//weights_fc[1][i][j] = quan_i_d(weights_fc[1][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	0,0.9921875, 128,
	1,0.9990234375, 1024
	);
	
	//quantization
	for(i=0;i<500;i++)
		output_fc2[i]=quan_i_d(output_fc2[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	//quantization
	for(i=0;i<10;i++)
	{
		//bias_fc[2][i] = quan_i_d(bias_fc[2][i], 0,0.99609375, 256);
		for(j=0;j<500;j++)
		{
			//weights_fc[2][i][j] = quan_i_d(weights_fc[2][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3,
	0,0.9921875, 128,
	7,0.9921875, 128
	);
	
	//quantization
	for(i=0;i<10;i++)
	{
		output_fc3[i]=quan_i_d(output_fc3[i], 3,0.96875, 32);
	}
	
	
	// Make prediction
	int prediction;
	prediction=0;
	//double max;
	max = output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}


int LeNet5_Date19_quan_sobol
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];


	// Quantiza the input data
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
			//if(data[0][i][j]!=quan_i_d(data[0][i][j], 0,0.875, 8))
				//printf("data[%d][%d]=%lf, quan=%lf\n",i,j,data[0][i][j],quan_i_d(data[0][i][j], 0,0.875, 8));
			data[0][i][j]=quan_i_d(data[0][i][j], 0,0.99609375, 256);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.9375, 16);
		}
	}
	
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99609375, 256,
    Sobol_seq,
	len
	);
	
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
		{
			for(k=0;k<24;k++)
			{
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 0,0.9921875, 128);
			}
		}
	}
	
	
	double max=0;
	//printf("max=%lf,",max);
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);

	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}

	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	
	Conv_quan // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99951171875, 2048
	);
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			for(k=0;k<8;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
			}
		}
	}
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}

	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}	
	
	
	FC_quan // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	0,0.9921875, 128,
	1,0.99951171875, 2048
	);
	

	//quantization
	for(i=0;i<800;i++)
		output_fc1[i]=quan_i_d(output_fc1[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[500];
	
	
	//quantization
	for(i=0;i<500;i++)
	{
		//bias_fc[1][i] = quan_i_d(bias_fc[1][i], 0,0.99609375, 256);
		for(j=0;j<800;j++)
		{
			//weights_fc[1][i][j] = quan_i_d(weights_fc[1][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	0,0.9921875, 128,
	1,0.9990234375, 1024
	);
	
	//quantization
	for(i=0;i<500;i++)
		output_fc2[i]=quan_i_d(output_fc2[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	//quantization
	for(i=0;i<10;i++)
	{
		//bias_fc[2][i] = quan_i_d(bias_fc[2][i], 0,0.99609375, 256);
		for(j=0;j<500;j++)
		{
			//weights_fc[2][i][j] = quan_i_d(weights_fc[2][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3,
	0,0.9921875, 128,
	7,0.9921875, 128
	);
	
	//quantization
	for(i=0;i<10;i++)
	{
		output_fc3[i]=quan_i_d(output_fc3[i], 3,0.96875, 32);
	}
	
	
	// Make prediction
	int prediction;
	prediction=0;
	//double max;
	max = output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}








int LeNet5_Date19_quan_sobol_all_replace
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];


	// Quantiza the input data
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
			//if(data[0][i][j]!=quan_i_d(data[0][i][j], 0,0.875, 8))
				//printf("data[%d][%d]=%lf, quan=%lf\n",i,j,data[0][i][j],quan_i_d(data[0][i][j], 0,0.875, 8));
			data[0][i][j]=quan_i_d(data[0][i][j], 0,0.99609375, 256);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.9375, 16);
		}
	}
	
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99609375, 256,
    Sobol_seq,
	len
	);
	
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
		{
			for(k=0;k<24;k++)
			{
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 0,0.9921875, 128);
			}
		}
	}
	
	
	double max=0;
	//printf("max=%lf,",max);
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);

	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}

	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99951171875, 2048,
    Sobol_seq,
	len
	);
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			for(k=0;k<8;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
			}
		}
	}
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}

	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k]/2;
				ii++;
			}
		}
	}	
	
	
	FC_quan_sc_sobol // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	0,0.9921875, 128,
	1,0.99951171875, 2048,
    Sobol_seq,
	len
	);
	

	//quantization
	for(i=0;i<800;i++)
		output_fc1[i]=quan_i_d(output_fc1[i]*2, 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=((output_fc1[i]<0)?0:output_fc1[i])/2;
	
	// FC2
	double output_fc2[500];
	
	
	//quantization
	for(i=0;i<500;i++)
	{
		//bias_fc[1][i] = quan_i_d(bias_fc[1][i], 0,0.99609375, 256);
		for(j=0;j<800;j++)
		{
			//weights_fc[1][i][j] = quan_i_d(weights_fc[1][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan_sc_sobol // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	0,0.9921875, 128,
	1,0.9990234375, 1024,
    Sobol_seq,
	len
	);
	
	//quantization
	for(i=0;i<500;i++)
		output_fc2[i]=quan_i_d(output_fc2[i]*2, 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=((output_fc2[i]<0)?0:output_fc2[i])/2;
	
	// FC3
	double output_fc3[10];
	
	//quantization
	for(i=0;i<10;i++)
	{
		//bias_fc[2][i] = quan_i_d(bias_fc[2][i], 0,0.99609375, 256);
		for(j=0;j<500;j++)
		{
			//weights_fc[2][i][j] = quan_i_d(weights_fc[2][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan_sc_sobol // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3,
	0,0.9921875, 128,
	7,0.9921875, 128,
    Sobol_seq,
	len
	);
	
	//quantization
	for(i=0;i<10;i++)
	{
		output_fc3[i]=quan_i_d(output_fc3[i]*2, 3,0.96875, 32);
	}
	
	
	// Make prediction
	int prediction;
	prediction=0;
	//double max;
	max = output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}



int LeNet5_Date19_quan_sobol_conv_replace
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];


	// Quantiza the input data
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
			//if(data[0][i][j]!=quan_i_d(data[0][i][j], 0,0.875, 8))
				//printf("data[%d][%d]=%lf, quan=%lf\n",i,j,data[0][i][j],quan_i_d(data[0][i][j], 0,0.875, 8));
			data[0][i][j]=quan_i_d(data[0][i][j], 0,0.99609375, 256);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.9375, 16);
		}
	}
	
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99609375, 256,
    Sobol_seq,
	len
	);
	
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
		{
			for(k=0;k<24;k++)
			{
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 0,0.9921875, 128);
			}
		}
	}
	
	
	double max=0;
	//printf("max=%lf,",max);
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);

	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}

	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99951171875, 2048,
    Sobol_seq,
	len
	);
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			for(k=0;k<8;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
			}
		}
	}
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}

	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}	
	
	
	FC_quan // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	0,0.9921875, 128,
	1,0.99951171875, 2048
	);
	

	//quantization
	for(i=0;i<800;i++)
		output_fc1[i]=quan_i_d(output_fc1[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[500];
	
	
	//quantization
	for(i=0;i<500;i++)
	{
		//bias_fc[1][i] = quan_i_d(bias_fc[1][i], 0,0.99609375, 256);
		for(j=0;j<800;j++)
		{
			//weights_fc[1][i][j] = quan_i_d(weights_fc[1][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	0,0.9921875, 128,
	1,0.9990234375, 1024
	);
	
	//quantization
	for(i=0;i<500;i++)
		output_fc2[i]=quan_i_d(output_fc2[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	//quantization
	for(i=0;i<10;i++)
	{
		//bias_fc[2][i] = quan_i_d(bias_fc[2][i], 0,0.99609375, 256);
		for(j=0;j<500;j++)
		{
			//weights_fc[2][i][j] = quan_i_d(weights_fc[2][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3,
	0,0.9921875, 128,
	7,0.9921875, 128
	);
	
	//quantization
	for(i=0;i<10;i++)
	{
		output_fc3[i]=quan_i_d(output_fc3[i], 3,0.96875, 32);
	}
	
	
	// Make prediction
	int prediction;
	prediction=0;
	//double max;
	max = output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}




int LeNet5_Date19_quan_sobol_conv_2fc_replace
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];


	// Quantiza the input data
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
			//if(data[0][i][j]!=quan_i_d(data[0][i][j], 0,0.875, 8))
				//printf("data[%d][%d]=%lf, quan=%lf\n",i,j,data[0][i][j],quan_i_d(data[0][i][j], 0,0.875, 8));
			data[0][i][j]=quan_i_d(data[0][i][j], 0,0.99609375, 256);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.9375, 16);
		}
	}
	
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99609375, 256,
    Sobol_seq,
	len
	);
	
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
		{
			for(k=0;k<24;k++)
			{
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 0,0.9921875, 128);
			}
		}
	}
	
	
	double max=0;
	//printf("max=%lf,",max);
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);

	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}

	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99951171875, 2048,
    Sobol_seq,
	len
	);
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			for(k=0;k<8;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
			}
		}
	}
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}

	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}	
	
	
	FC_quan_sc_sobol // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	0,0.9921875, 128,
	1,0.99951171875, 2048,
    Sobol_seq,
	len
	);
	

	//quantization
	for(i=0;i<800;i++)
		output_fc1[i]=quan_i_d(output_fc1[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[500];
	
	
	//quantization
	for(i=0;i<500;i++)
	{
		//bias_fc[1][i] = quan_i_d(bias_fc[1][i], 0,0.99609375, 256);
		for(j=0;j<800;j++)
		{
			//weights_fc[1][i][j] = quan_i_d(weights_fc[1][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan_sc_sobol // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	0,0.9921875, 128,
	1,0.9990234375, 1024,
    Sobol_seq,
	len
	);
	
	//quantization
	for(i=0;i<500;i++)
		output_fc2[i]=quan_i_d(output_fc2[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	//quantization
	for(i=0;i<10;i++)
	{
		//bias_fc[2][i] = quan_i_d(bias_fc[2][i], 0,0.99609375, 256);
		for(j=0;j<500;j++)
		{
			//weights_fc[2][i][j] = quan_i_d(weights_fc[2][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3,
	0,0.9921875, 128,
	7,0.9921875, 128
	);
	
	//quantization
	for(i=0;i<10;i++)
	{
		output_fc3[i]=quan_i_d(output_fc3[i], 3,0.96875, 32);
	}
	
	
	// Make prediction
	int prediction;
	prediction=0;
	//double max;
	max = output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}





int LeNet5_Date19_quan_sobol_conv_1fc_replace
(
	double **test_data, // Input picture 28*28
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	
	// Conv1
	double ***data;
	data=(double ***)malloc(sizeof(double **)*1);
	data[0]=(double **)malloc(sizeof(double *)*28); // No Padding
	for(i=0;i<28;i++)
		data[0][i]=(double *)malloc(sizeof(double)*28);
	
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*24);
		for(j=0;j<24;j++)
		{
			output_conv1[i][j]=(double *)malloc(sizeof(double)*24);
		}
	}
	
	// No padding
	for(i=0;i<28;i++)
		for(j=0;j<28;j++)
			data[0][i][j]=test_data[i][j];


	// Quantiza the input data
	for(i=0;i<28;i++)
	{
		for(j=0;j<28;j++)
		{
			//if(data[0][i][j]!=quan_i_d(data[0][i][j], 0,0.875, 8))
				//printf("data[%d][%d]=%lf, quan=%lf\n",i,j,data[0][i][j],quan_i_d(data[0][i][j], 0,0.875, 8));
			data[0][i][j]=quan_i_d(data[0][i][j], 0,0.99609375, 256);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.9375, 16);
		}
	}
	
	
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	28, // Height of input feature map
	28, // Width of input feature map
	24, // Height of output feature map: H_out=(H-(m-stride))/stride
	24, // Width of output feature map: W_out=(W-(n-stride))/stride
	1, // Number of input channels
	20, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99609375, 256,
    Sobol_seq,
	len
	);
	
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
		{
			for(k=0;k<24;k++)
			{
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 0,0.9921875, 128);
			}
		}
	}
	
	
	double max=0;
	//printf("max=%lf,",max);
	// Pooling
	double ***output_pooling1;
	output_pooling1=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*12);
		}
	}
		
	max_pooling
	(
	24, // Height of input feature map
	24, // Width of input feature map
	2,
	2,
	12, // Height of output feature map: H_out=H/strideH
	12, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv1, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);

	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_pooling1[i][j][k]=(output_pooling1[i][j][k]<0)?0:output_pooling1[i][j][k];
			}
		}
	}

	
	// Conv2
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*8);
		for(j=0;j<8;j++)
		{
			output_conv2[i][j]=(double *)malloc(sizeof(double)*8);
		}
	}
	 
	Conv_quan_SC_Sobol // Convolutional layer
	(
	5, // Kernel size is m*n
	5,
	12, // Height of input feature map
	12, // Width of input feature map
	8, // Height of output feature map: H_out=(H-(m-stride))/stride
	8, // Width of output feature map: W_out=(W-(n-stride))/stride
	20, // Number of input channels
	20, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	0,0.9921875, 128,
	1,0.99951171875, 2048,
    Sobol_seq,
	len
	);
	
	//quantization
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			for(k=0;k<8;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64) ;
			}
		}
	}
	
	// Pooling
	double ***output_pooling2;
	output_pooling2=(double ***)malloc(sizeof(double **)*20);
	for(i=0;i<20;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*4);
		}
	}
		
	max_pooling
	(
	8, // Height of input feature map
	8, // Width of input feature map
	2,
	2,
	4, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	20, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	
	
	// ReLU
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				output_pooling2[i][j][k]=(output_pooling2[i][j][k]<0)?0:output_pooling2[i][j][k];
			}
		}
	}

	// FC1
	double input_fc1[320],output_fc1[800];
	ii=0;
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			for(k=0;k<4;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k]/2;
				ii++;
			}
		}
	}	
	
	
	FC_quan_sc_sobol // Fully connection
	(
	320, //The number of input neurons
	800, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	0,0.9921875, 128,
	1,0.99951171875, 2048,
    Sobol_seq,
	len
	);
	

	//quantization
	for(i=0;i<800;i++)
		output_fc1[i]=quan_i_d(output_fc1[i]*2, 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<800;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	
	// FC2
	double output_fc2[500];
	
	
	//quantization
	for(i=0;i<500;i++)
	{
		//bias_fc[1][i] = quan_i_d(bias_fc[1][i], 0,0.99609375, 256);
		for(j=0;j<800;j++)
		{
			//weights_fc[1][i][j] = quan_i_d(weights_fc[1][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	800, //The number of input neurons
	500, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	0,0.9921875, 128,
	1,0.9990234375, 1024
	);
	
	//quantization
	for(i=0;i<500;i++)
		output_fc2[i]=quan_i_d(output_fc2[i], 1,0.984375, 64);
	
	
	//ReLU
	for(i=0;i<500;i++)
		output_fc2[i]=(output_fc2[i]<0)?0:output_fc2[i];
	
	// FC3
	double output_fc3[10];
	
	//quantization
	for(i=0;i<10;i++)
	{
		//bias_fc[2][i] = quan_i_d(bias_fc[2][i], 0,0.99609375, 256);
		for(j=0;j<500;j++)
		{
			//weights_fc[2][i][j] = quan_i_d(weights_fc[2][i][j], 0,0.99609375, 256);
		}
	}
	
	FC_quan // Fully connection
	(
	500, //The number of input neurons
	10, // The number of output neurons
	weights_fc[2], // Weight array with size out_neuron*in_neuron
	bias_fc[2], // out_neuron biases
	output_fc2,
	output_fc3,
	0,0.9921875, 128,
	7,0.9921875, 128
	);
	
	//quantization
	for(i=0;i<10;i++)
	{
		output_fc3[i]=quan_i_d(output_fc3[i], 3,0.96875, 32);
	}
	
	
	// Make prediction
	int prediction;
	prediction=0;
	//double max;
	max = output_fc3[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc3[i]>max)
		{
			max=output_fc3[i];
			prediction=i;
		}
	}
	
	
	for(i=0;i<28;i++)
		free(data[0][i]);
	free(data[0]);
	free(data);
	
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<24;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<12;j++)
		{
			free(output_pooling1[i][j]);
		}
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<8;j++)
		{
			free(output_conv2[i][j]);
		}
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<20;i++)
	{
		for(j=0;j<4;j++)
		{
			free(output_pooling2[i][j]);
		}
		free(output_pooling2[i]);
	}
	free(output_pooling2);
	
	
	
	return prediction;
}







// 


int SqueezeNetChen
(
	double ***test_data, // Input picture 3*32*32
	double *****weights_conv,
	double **bias_conv,
	double ******weights_fire,
	double ***bias_fire
)
{
	int i,j,k,ii,jj,kk;
	
	
	// First convolutional layer, which has been combined with BN
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*96);
	for(i=0;i<96;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv1[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	Conv // Convolutional layer
	(
	2, // Kernel size is m*n
	2,
	32, // Height of input feature map
	32, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	16, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	96, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	2,
	test_data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	
	for(i=0;i<96;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<16;k++)
			{
				if(output_conv1[i][j][k]<0)
					output_conv1[i][j][k]=0;
			}
		}
	}

	// int stride=2;
	// for(i=0;i<96;i++)// Number of output channels
	// {
	// 	for(j=0;j<16;j++)// Height of output feature map
	// 	{
	// 		for(k=0;k<16;k++)// Width of output feature map
	// 		{
				
	// 			output_conv1[i][j][k]=bias_conv[0][i];
	// 			for(ii=0;ii<3;ii++)// Number of input channels
	// 			{
					
	// 				for(jj=0;jj<2;jj++)
	// 				{
	// 					for(kk=0;kk<2;kk++)
	// 					{
	// 						// output_conv1[i][j][k]+=quan_i_d(weights_conv[0][i][ii][jj][kk]*test_data[ii][j*stride+jj][k*stride+kk],7,0.99609375,256);//90.84%
	// 						//output_conv1[i][j][k]+=quan_i_d(weights_conv[0][i][ii][jj][kk]*test_data[ii][j*stride+jj][k*stride+kk],3,0.99609375,256);
	// 						//output_conv1[i][j][k]+=quan_i_d(weights_conv[0][i][ii][jj][kk]*test_data[ii][j*stride+jj][k*stride+kk],3,0.9921875,128); //90.65%
	// 						output_conv1[i][j][k]+=quan_i_d(weights_conv[0][i][ii][jj][kk]*test_data[ii][j*stride+jj][k*stride+kk],3,0.984375,64); //90.45%
	// 						// output_conv1[i][j][k]=quan_i_d(output_conv1[i][j][k],7,0.99609375,256); //90.84%
	// 						//output_conv1[i][j][k]=quan_i_d(output_conv1[i][j][k],7,0.9921875,128);
	// 						// output_conv1[i][j][k]=quan_i_d(output_conv1[i][j][k],3,0.9921875,128);//89.89%
	// 						output_conv1[i][j][k]=quan_i_d(output_conv1[i][j][k],3,0.96875,32);//89.59%
	// 						//if(i==2&&j==0&&k==4)
	// 							//printf("jj=%d, kk=%d: weight=%lf, test_data=%lf, output=%lf\n",jj,kk,weights_conv[0][i][ii][jj][kk],test_data[ii][j*stride+jj][k*stride+kk],output_conv1[i][j][k]);
							
	// 					}
	// 				}
					
	// 			}

	// 		}
	// 	}
	// }
	
	
	
	// for(i=0;i<96;i++)
	// {
	// 	for(j=0;j<16;j++)
	// 	{
	// 		for(k=0;k<16;k++)
	// 		{
	// 			if(output_conv1[i][j][k]<0)
	// 				output_conv1[i][j][k]=0;
	// 			else
	// 				//output_conv1[i][j][k]=quan_i_d(output_conv1[i][j][k],3,0.984375,64);
	// 				// output_conv1[i][j][k]=quan_i_d(output_conv1[i][j][k],7,0.99609375,256);
	// 				output_conv1[i][j][k]=quan_i_d(output_conv1[i][j][k],3,0.96875,32);
	// 		}
	// 	}
	// }
	
	
	// fire module 1
	double ***output_fire1;
	output_fire1=(double ***)malloc(sizeof(double **)*64);
	for(i=0;i<64;i++)
	{
		output_fire1[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire1[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	96,
	64,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	16, // Number of 1*1 squeeze filters
	32, // Number of 1*1 expand filters
	32, // Number of 3*3 expand filters
	weights_fire[0], // Weights of the three kinds of filters
	bias_fire[0], // Biases of the three kinds of filters
	output_conv1, // Input feature map
	output_fire1 // Output feature map
	);
	
	// fire module 2
	double ***output_fire2;
	output_fire2=(double ***)malloc(sizeof(double **)*128);
	for(i=0;i<128;i++)
	{
		output_fire2[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire2[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	64,
	128,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	32, // Number of 1*1 squeeze filters
	64, // Number of 1*1 expand filters
	64, // Number of 3*3 expand filters
	weights_fire[1], // Weights of the three kinds of filters
	bias_fire[1], // Biases of the three kinds of filters
	output_fire1, // Input feature map
	output_fire2 // Output feature map
	); 
	
	// fire module 3
	double ***output_fire3;
	output_fire3=(double ***)malloc(sizeof(double **)*256);
	for(i=0;i<256;i++)
	{
		output_fire3[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire3[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	128,
	256,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	64, // Number of 1*1 squeeze filters
	128, // Number of 1*1 expand filters
	128, // Number of 3*3 expand filters
	weights_fire[2], // Weights of the three kinds of filters
	bias_fire[2], // Biases of the three kinds of filters
	output_fire2, // Input feature map
	output_fire3 // Output feature map
	);
	
	// fire module 4
	double ***output_fire4;
	output_fire4=(double ***)malloc(sizeof(double **)*360);
	for(i=0;i<360;i++)
	{
		output_fire4[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire4[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	256,
	360,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	96, // Number of 1*1 squeeze filters
	180, // Number of 1*1 expand filters
	180, // Number of 3*3 expand filters
	weights_fire[3], // Weights of the three kinds of filters
	bias_fire[3], // Biases of the three kinds of filters
	output_fire3, // Input feature map
	output_fire4 // Output feature map
	);
	
	// Second convolutional layer, which has been combined with BN
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*180);
	for(i=0;i<180;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv2[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	Conv // Convolutional layer
	(
	1, // Kernel size is m*n
	1,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	16, // Width of output feature map: W_out=(W-(n-stride))/stride
	360, // Number of input channels
	180, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_fire4, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	
	for(i=0;i<180;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<16;k++)
			{
				if(output_conv2[i][j][k]<0)
					output_conv2[i][j][k]=0;
			}
		}
	}
	
	double ***output_conv2_pool;
	output_conv2_pool=(double ***)malloc(sizeof(double **)*180);
	for(i=0;i<180;i++)
	{
		output_conv2_pool[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv2_pool[i][j]=(double *)malloc(sizeof(double)*4);
	}
	
	max_pooling
	(
	16, // Height of input feature map
	16, // Width of input feature map
	1,
	4,
	16, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	180, // Number of channels
	output_conv2, // num_channel*H*W
	output_conv2_pool // num_channel*H_out*W_out
	);
	
	
	// Third convolutional layer, which has been combined with BN
	double ***output_conv3;
	output_conv3=(double ***)malloc(sizeof(double **)*120);
	for(i=0;i<120;i++)
	{
		output_conv3[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv3[i][j]=(double *)malloc(sizeof(double)*4);
	}
	
	Conv // Convolutional layer
	(
	1, // Kernel size is m*n
	1,
	16, // Height of input feature map
	4, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	4, // Width of output feature map: W_out=(W-(n-stride))/stride
	180, // Number of input channels
	120, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_conv2_pool, // num_in_channel*H*W
	output_conv3 // num_out_channel*H_out*W_out
	);
	
	for(i=0;i<120;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<4;k++)
			{
				if(output_conv3[i][j][k]<0)
					output_conv3[i][j][k]=0;
			}
		}
	}
	
	double ***output_conv3_pool;
	output_conv3_pool=(double ***)malloc(sizeof(double **)*120);
	for(i=0;i<120;i++)
	{
		output_conv3_pool[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv3_pool[i][j]=(double *)malloc(sizeof(double)*1);
	}
	
	max_pooling
	(
	16, // Height of input feature map
	4, // Width of input feature map
	1,
	4,
	16, // Height of output feature map: H_out=H/strideH
	1, // Width of output feature map: W_out=W/strideW
	120, // Number of channels
	output_conv3, // num_channel*H*W
	output_conv3_pool // num_channel*H_out*W_out
	);
	
	
	// Fourth convolutional layer, which has been combined with BN
	double ***output_conv4;
	output_conv4=(double ***)malloc(sizeof(double **)*10);
	for(i=0;i<10;i++)
	{
		output_conv4[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv4[i][j]=(double *)malloc(sizeof(double)*1);
	}
	
	Conv // Convolutional layer
	(
	1, // Kernel size is m*n
	1,
	16, // Height of input feature map
	1, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	1, // Width of output feature map: W_out=(W-(n-stride))/stride
	120, // Number of input channels
	10, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3_pool, // num_in_channel*H*W
	output_conv4 // num_out_channel*H_out*W_out
	);
	
	for(i=0;i<10;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<1;k++)
			{
				if(output_conv4[i][j][k]<0)
					output_conv4[i][j][k]=0;
			}
		}
	}
	
	double ***output_conv4_pool;
	output_conv4_pool=(double ***)malloc(sizeof(double **)*10);
	for(i=0;i<10;i++)
	{
		output_conv4_pool[i]=(double **)malloc(sizeof(double *)*1);
		for(j=0;j<1;j++)
			output_conv4_pool[i][j]=(double *)malloc(sizeof(double)*1);
	}
	
	average_pooling
	(
	16, // Height of input feature map
	1, // Width of input feature map
	16,
	1,
	1, // Height of output feature map: H_out=H/strideH
	1, // Width of output feature map: W_out=W/strideW
	10, // Number of channels
	output_conv4, // num_channel*H*W
	output_conv4_pool // num_channel*H_out*W_out
	);
	
	
	
	// Make prediction
	int prediction;
	prediction=0;
	double max;
	max = output_conv4_pool[0][0][0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_conv4_pool[i][0][0]>max)
		{
			max=output_conv4_pool[i][0][0];
			prediction=i;
		}
	}
	
	
	
	
	
	//Free
	for(i=0;i<96;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<64;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire1[i][j]);
		free(output_fire1[i]);
	}
	free(output_fire1);
	
	for(i=0;i<128;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire2[i][j]);
		free(output_fire2[i]);
	}
	free(output_fire2);
	
	for(i=0;i<256;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire3[i][j]);
		free(output_fire3[i]);
	}
	free(output_fire3);
	
	for(i=0;i<360;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire4[i][j]);
		free(output_fire4[i]);
	}
	free(output_fire4);
	
	for(i=0;i<180;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv2[i][j]);
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	
	for(i=0;i<180;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv2_pool[i][j]);
		free(output_conv2_pool[i]);
	}
	free(output_conv2_pool);
	
	for(i=0;i<120;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv3[i][j]);
		free(output_conv3[i]);
	}
	free(output_conv3);
	
	
	for(i=0;i<120;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv3_pool[i][j]);
		free(output_conv3_pool[i]);
	}
	free(output_conv3_pool);
	
	for(i=0;i<10;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv4[i][j]);
		free(output_conv4[i]);
	}
	free(output_conv4);
	
	
	for(i=0;i<10;i++)
	{
		for(j=0;j<1;j++)
			free(output_conv4_pool[i][j]);
		free(output_conv4_pool[i]);
	}
	free(output_conv4_pool);
	
	return prediction;
}



int SqueezeNetChen_SC
(
	double ***test_data, // Input picture 3*32*32
	double *****weights_conv,
	double **bias_conv,
	double ******weights_fire,
	double ***bias_fire,
	double **Sobol_seq, //Sobol number[3][1023]
	int len
)
{
	int i,j,k,ii,jj,kk;
	
	
	// First convolutional layer, which has been combined with BN
	double ***output_conv1;
	output_conv1=(double ***)malloc(sizeof(double **)*96);
	for(i=0;i<96;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv1[i][j]=(double *)malloc(sizeof(double)*16);
	}


	
	Conv_ReLU_quan_SC_Sobol // Convolutional layer
	(
	2, // Kernel size is m*n
	2,
	32, // Height of input feature map
	32, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	16, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	96, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[0], // num_out_channel biases
	2,
	test_data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048,
	Sobol_seq,
	len,
	4
	);
	
	// for(i=0;i<96;i++)
	// {
	// 	for(j=0;j<16;j++)
	// 	{
	// 		for(k=0;k<16;k++)
	// 		{
	// 			if(output_conv1[i][j][k]<0)
	// 				output_conv1[i][j][k]=0;
	// 		}
	// 	}
	// }
	for(i=0;i<96;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<16;k++)
			{
				//output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.9921875, 128);
				// output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.984375, 64);
				// output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 3,0.96875,32);
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 3,0.984375,64);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}


	


	
	// fire module 1
	double ***output_fire1;
	output_fire1=(double ***)malloc(sizeof(double **)*64);
	for(i=0;i<64;i++)
	{
		output_fire1[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire1[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	96,
	64,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	16, // Number of 1*1 squeeze filters
	32, // Number of 1*1 expand filters
	32, // Number of 3*3 expand filters
	weights_fire[0], // Weights of the three kinds of filters
	bias_fire[0], // Biases of the three kinds of filters
	output_conv1, // Input feature map
	output_fire1 // Output feature map
	);
	
	// fire module 2
	double ***output_fire2;
	output_fire2=(double ***)malloc(sizeof(double **)*128);
	for(i=0;i<128;i++)
	{
		output_fire2[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire2[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	64,
	128,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	32, // Number of 1*1 squeeze filters
	64, // Number of 1*1 expand filters
	64, // Number of 3*3 expand filters
	weights_fire[1], // Weights of the three kinds of filters
	bias_fire[1], // Biases of the three kinds of filters
	output_fire1, // Input feature map
	output_fire2 // Output feature map

	); 
	
	// fire module 3
	double ***output_fire3;
	output_fire3=(double ***)malloc(sizeof(double **)*256);
	for(i=0;i<256;i++)
	{
		output_fire3[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire3[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	128,
	256,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	64, // Number of 1*1 squeeze filters
	128, // Number of 1*1 expand filters
	128, // Number of 3*3 expand filters
	weights_fire[2], // Weights of the three kinds of filters
	bias_fire[2], // Biases of the three kinds of filters
	output_fire2, // Input feature map
	output_fire3 // Output feature map
	);
	
	// fire module 4
	double ***output_fire4;
	output_fire4=(double ***)malloc(sizeof(double **)*360);
	for(i=0;i<360;i++)
	{
		output_fire4[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_fire4[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	FireChen
	(
	256,
	360,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map
	16, // Width of output feature map
	96, // Number of 1*1 squeeze filters
	180, // Number of 1*1 expand filters
	180, // Number of 3*3 expand filters
	weights_fire[3], // Weights of the three kinds of filters
	bias_fire[3], // Biases of the three kinds of filters
	output_fire3, // Input feature map
	output_fire4 // Output feature map
	);
	
	// Second convolutional layer, which has been combined with BN
	double ***output_conv2;
	output_conv2=(double ***)malloc(sizeof(double **)*180);
	for(i=0;i<180;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv2[i][j]=(double *)malloc(sizeof(double)*16);
	}
	
	Conv // Convolutional layer
	(
	1, // Kernel size is m*n
	1,
	16, // Height of input feature map
	16, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	16, // Width of output feature map: W_out=(W-(n-stride))/stride
	360, // Number of input channels
	180, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_fire4, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out

	);
	
	for(i=0;i<180;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<16;k++)
			{
				if(output_conv2[i][j][k]<0)
					output_conv2[i][j][k]=0;
			}
		}
	}
	
	double ***output_conv2_pool;
	output_conv2_pool=(double ***)malloc(sizeof(double **)*180);
	for(i=0;i<180;i++)
	{
		output_conv2_pool[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv2_pool[i][j]=(double *)malloc(sizeof(double)*4);
	}
	
	max_pooling
	(
	16, // Height of input feature map
	16, // Width of input feature map
	1,
	4,
	16, // Height of output feature map: H_out=H/strideH
	4, // Width of output feature map: W_out=W/strideW
	180, // Number of channels
	output_conv2, // num_channel*H*W
	output_conv2_pool // num_channel*H_out*W_out
	);
	
	
	// Third convolutional layer, which has been combined with BN
	double ***output_conv3;
	output_conv3=(double ***)malloc(sizeof(double **)*120);
	for(i=0;i<120;i++)
	{
		output_conv3[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv3[i][j]=(double *)malloc(sizeof(double)*4);
	}
	
	Conv // Convolutional layer
	(
	1, // Kernel size is m*n
	1,
	16, // Height of input feature map
	4, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	4, // Width of output feature map: W_out=(W-(n-stride))/stride
	180, // Number of input channels
	120, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_conv2_pool, // num_in_channel*H*W
	output_conv3 // num_out_channel*H_out*W_out

	);
	
	for(i=0;i<120;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<4;k++)
			{
				if(output_conv3[i][j][k]<0)
					output_conv3[i][j][k]=0;
			}
		}
	}
	
	double ***output_conv3_pool;
	output_conv3_pool=(double ***)malloc(sizeof(double **)*120);
	for(i=0;i<120;i++)
	{
		output_conv3_pool[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv3_pool[i][j]=(double *)malloc(sizeof(double)*1);
	}
	
	max_pooling
	(
	16, // Height of input feature map
	4, // Width of input feature map
	1,
	4,
	16, // Height of output feature map: H_out=H/strideH
	1, // Width of output feature map: W_out=W/strideW
	120, // Number of channels
	output_conv3, // num_channel*H*W
	output_conv3_pool // num_channel*H_out*W_out
	);
	
	
	// Fourth convolutional layer, which has been combined with BN
	double ***output_conv4;
	output_conv4=(double ***)malloc(sizeof(double **)*10);
	for(i=0;i<10;i++)
	{
		output_conv4[i]=(double **)malloc(sizeof(double *)*16);
		for(j=0;j<16;j++)
			output_conv4[i][j]=(double *)malloc(sizeof(double)*1);
	}
	
	Conv // Convolutional layer
	(
	1, // Kernel size is m*n
	1,
	16, // Height of input feature map
	1, // Width of input feature map
	16, // Height of output feature map: H_out=(H-(m-stride))/stride
	1, // Width of output feature map: W_out=(W-(n-stride))/stride
	120, // Number of input channels
	10, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3_pool, // num_in_channel*H*W
	output_conv4 // num_out_channel*H_out*W_out

	);
	
	for(i=0;i<10;i++)
	{
		for(j=0;j<16;j++)
		{
			for(k=0;k<1;k++)
			{
				if(output_conv4[i][j][k]<0)
					output_conv4[i][j][k]=0;
			}
		}
	}
	
	double ***output_conv4_pool;
	output_conv4_pool=(double ***)malloc(sizeof(double **)*10);
	for(i=0;i<10;i++)
	{
		output_conv4_pool[i]=(double **)malloc(sizeof(double *)*1);
		for(j=0;j<1;j++)
			output_conv4_pool[i][j]=(double *)malloc(sizeof(double)*1);
	}
	
	average_pooling
	(
	16, // Height of input feature map
	1, // Width of input feature map
	16,
	1,
	1, // Height of output feature map: H_out=H/strideH
	1, // Width of output feature map: W_out=W/strideW
	10, // Number of channels
	output_conv4, // num_channel*H*W
	output_conv4_pool // num_channel*H_out*W_out
	);
	
	
	
	// Make prediction
	int prediction;
	prediction=0;
	double max;
	max = output_conv4_pool[0][0][0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_conv4_pool[i][0][0]>max)
		{
			max=output_conv4_pool[i][0][0];
			prediction=i;
		}
	}
	
	
	
	
	
	//Free
	for(i=0;i<96;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<64;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire1[i][j]);
		free(output_fire1[i]);
	}
	free(output_fire1);
	
	for(i=0;i<128;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire2[i][j]);
		free(output_fire2[i]);
	}
	free(output_fire2);
	
	for(i=0;i<256;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire3[i][j]);
		free(output_fire3[i]);
	}
	free(output_fire3);
	
	for(i=0;i<360;i++)
	{
		for(j=0;j<16;j++)
			free(output_fire4[i][j]);
		free(output_fire4[i]);
	}
	free(output_fire4);
	
	for(i=0;i<180;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv2[i][j]);
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	
	for(i=0;i<180;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv2_pool[i][j]);
		free(output_conv2_pool[i]);
	}
	free(output_conv2_pool);
	
	for(i=0;i<120;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv3[i][j]);
		free(output_conv3[i]);
	}
	free(output_conv3);
	
	
	for(i=0;i<120;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv3_pool[i][j]);
		free(output_conv3_pool[i]);
	}
	free(output_conv3_pool);
	
	for(i=0;i<10;i++)
	{
		for(j=0;j<16;j++)
			free(output_conv4[i][j]);
		free(output_conv4[i]);
	}
	free(output_conv4);
	
	
	for(i=0;i<10;i++)
	{
		for(j=0;j<1;j++)
			free(output_conv4_pool[i][j]);
		free(output_conv4_pool[i]);
	}
	free(output_conv4_pool);
	
	return prediction;
}



int model_cifar10_sc
(
	double ***data, // Input picture 3*32*32
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double ***output_conv1,//32*30*30
	double ***output_conv2,//32*28*28
	double ***output_pooling1,//32*14*14
	double ***output_conv3,//64*12*12
	double ***output_conv4,//32*10*10
	double ***output_pooling2,//64*5*5
	double *input_fc1, //1600
	double *output_fc1, //512
	double *output_fc2 //10
)
{
	int i,j,k,ii,jj,kk;
	//for(i=0;i<32;i++)
		//printf("%lf ",data[0][0][i]);
	// Conv1
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	/*
	for(i=0;i<1;i++)
	{
		for(j=0;j<30;j++)
		{
			printf("%lf ",output_conv1[0][i][j]);
		}
		printf("\n");
	}
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				output_conv1[i][j][k]=(output_conv1[i][j][k]<0)?0:output_conv1[i][j][k];
			}
		}
	}
	*/
	
	
	//printf("Conv1 has been finished!\n");
	// Conv2
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k]=(output_conv2[i][j][k]<0)?0:output_conv2[i][j][k];
			}
		}
	}
	*/
	//printf("Conv2 has been finished!\n");
	
	// Pooling1
	max_pooling
	(
	28, // Height of input feature map
	28, // Width of input feature map
	2,
	2,
	14, // Height of output feature map: H_out=H/strideH
	14, // Width of output feature map: W_out=W/strideW
	32, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	
	
	// Conv3
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3 // num_out_channel*H_out*W_out
	);
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k]=(output_conv3[i][j][k]<0)?0:output_conv3[i][j][k];
			}
		}
	}
	*/
	//printf("Conv3 has been finished!\n");
	
	// Conv4
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4 // num_out_channel*H_out*W_out
	);
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				output_conv4[i][j][k]=(output_conv4[i][j][k]<0)?0:output_conv4[i][j][k];
			}
		}
	}
	*/
	//printf("Conv4 has been finished!\n");
	
	
	// Pooling2
	max_pooling
	(
	10, // Height of input feature map
	10, // Width of input feature map
	2,
	2,
	5, // Height of output feature map: H_out=H/strideH
	5, // Width of output feature map: W_out=W/strideW
	64, // Number of channels
	output_conv4, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<64;i++)
	{	
		printf("i=%d:\n",i);
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
				printf("%lf ",output_pooling2[i][j][k]);
			printf("\n");
		}
	}
	*/
	// FC1
	ii=0;
	for(i=0;i<64;i++)
	{
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC_ReLU // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	
	//for(i=0;i<512;i++)
		//printf("%lf ",output_fc1[i]);
	//printf("%lf",output_fc1[509]);
	//printf("\n");
	/*
	//ReLU
	for(i=0;i<512;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	*/
	
	// FC2
	FC // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	//printf("The final output:\n");
	//for(i=0;i<10;i++)
		//printf("%lf ",output_fc2[i]);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc2[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc2[i]>max)
		{
			max=output_fc2[i];
			prediction=i;
		}
	}
	
	
	
	return prediction;
}


int model_cifar10_sc_quan
(
	double ***data, // Input picture 3*32*32
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double ***output_conv1,//32*30*30
	double ***output_conv2,//32*28*28
	double ***output_pooling1,//32*14*14
	double ***output_conv3,//64*12*12
	double ***output_conv4,//32*10*10
	double ***output_pooling2,//64*5*5
	double *input_fc1, //1600
	double *output_fc1, //512
	double *output_fc2 //10
)
{
	int i,j,k,ii,jj,kk;
	//for(i=0;i<32;i++)
		//printf("%lf ",data[0][0][i]);
	// Conv1
	
	Conv_ReLU_quan // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	*/
	/*
	for(i=0;i<1;i++)
	{
		for(j=0;j<30;j++)
		{
			printf("%lf ",output_conv1[0][i][j]);
		}
		printf("\n");
	}
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				output_conv1[i][j][k]=(output_conv1[i][j][k]<0)?0:output_conv1[i][j][k];
			}
		}
	}
	*/
	
	//Quantize the activations 
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				//output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.9921875, 128);
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.984375, 64);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	//printf("Conv1 has been finished!\n");
	// Conv2
	
	Conv_ReLU_quan // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k]=(output_conv2[i][j][k]<0)?0:output_conv2[i][j][k];
			}
		}
	}
	*/
	//printf("Conv2 has been finished!\n");
	
	//Quantize the activations 
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
				//output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling1
	max_pooling
	(
	28, // Height of input feature map
	28, // Width of input feature map
	2,
	2,
	14, // Height of output feature map: H_out=H/strideH
	14, // Width of output feature map: W_out=W/strideW
	32, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	
	
	// Conv3
	
	Conv_ReLU_quan // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048
	//3,0.990234375, 1024
	//3,0.998046875, 512
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k]=(output_conv3[i][j][k]<0)?0:output_conv3[i][j][k];
			}
		}
	}
	*/
	//printf("Conv3 has been finished!\n");
	
	//Quantize the activations 
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.96875, 32);
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.984375, 64);
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Conv4
	
	Conv_ReLU_quan // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048
	//3,0.990234375, 1024
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				output_conv4[i][j][k]=(output_conv4[i][j][k]<0)?0:output_conv4[i][j][k];
			}
		}
	}
	*/
	//printf("Conv4 has been finished!\n");
	
	//Quantize the activations 
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.875, 8);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9375, 16);
				output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.96875, 32);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.984375, 64);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling2
	max_pooling
	(
	10, // Height of input feature map
	10, // Width of input feature map
	2,
	2,
	5, // Height of output feature map: H_out=H/strideH
	5, // Width of output feature map: W_out=W/strideW
	64, // Number of channels
	output_conv4, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<64;i++)
	{	
		printf("i=%d:\n",i);
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
				printf("%lf ",output_pooling2[i][j][k]);
			printf("\n");
		}
	}
	*/
	// FC1
	ii=0;
	for(i=0;i<64;i++)
	{
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC_ReLU_quan // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	7,0.99951171875, 2048
	//7,0.990234375, 1024
	);
	/*
	FC_ReLU // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	*/
	
	//Quantize the activations 
	for(i=0;i<512;i++)
		//output_fc1[i]=quan_i_d(output_fc1[i], 7,0.984375, 64);
		output_fc1[i]=quan_i_d(output_fc1[i], 7,0.9375, 16);
	
	
	
	//for(i=0;i<512;i++)
		//printf("%lf ",output_fc1[i]);
	//printf("%lf",output_fc1[509]);
	//printf("\n");
	/*
	//ReLU
	for(i=0;i<512;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	*/
	
	// FC2
	
	FC_quan2 // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	//15,0.99951171875, 2048
	15,0.990234375, 1024
	//15,0.998046875, 512
	);
	/*
	FC // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	*/
	//printf("The final output:\n");
	//for(i=0;i<10;i++)
		//printf("%lf ",output_fc2[i]);
	
	
	//Quantize the activations 
	for(i=0;i<10;i++)
		//output_fc2[i]=quan_i_d(output_fc2[i], 15,0.9921875, 128);
		output_fc2[i]=quan_i_d(output_fc2[i], 15,0.990234375, 1024);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc2[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc2[i]>max)
		{
			max=output_fc2[i];
			prediction=i;
		}
	}
	
	
	
	return prediction;
}


int model_cifar10_sc_quan_sobol_all_replace
(
	double ***data, // Input picture 3*32*32
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double ***output_conv1,//32*30*30
	double ***output_conv2,//32*28*28
	double ***output_pooling1,//32*14*14
	double ***output_conv3,//64*12*12
	double ***output_conv4,//32*10*10
	double ***output_pooling2,//64*5*5
	double *input_fc1, //1600
	double *output_fc1, //512
	double *output_fc2, //10
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	//for(i=0;i<32;i++)
		//printf("%lf ",data[0][0][i]);
	// Conv1
	
	Conv_ReLU_quan_SC_Sobol // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048,
	Sobol_seq,
	len,
	4
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	*/
	/*
	for(i=0;i<1;i++)
	{
		for(j=0;j<30;j++)
		{
			printf("%lf ",output_conv1[0][i][j]);
		}
		printf("\n");
	}
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				output_conv1[i][j][k]=(output_conv1[i][j][k]<0)?0:output_conv1[i][j][k];
			}
		}
	}
	*/
	
	//Quantize the activations
	//!!!!scaled by 2!!!!!!!! 
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				//output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.9921875, 128);
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.984375, 64);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	//printf("Conv1 has been finished!\n");
	// Conv2
	
	Conv_ReLU_quan_SC_Sobol // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048,
	Sobol_seq,
	len,
	2
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k]=(output_conv2[i][j][k]<0)?0:output_conv2[i][j][k];
			}
		}
	}
	*/
	//printf("Conv2 has been finished!\n");
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 2!!!!!!!! 
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
				//output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling1
	max_pooling
	(
	28, // Height of input feature map
	28, // Width of input feature map
	2,
	2,
	14, // Height of output feature map: H_out=H/strideH
	14, // Width of output feature map: W_out=W/strideW
	32, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	
	
	// Conv3
	
	Conv_ReLU_quan_SC_Sobol // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048,
	Sobol_seq,
	len,
	2
	//3,0.990234375, 1024
	//3,0.998046875, 512
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k]=(output_conv3[i][j][k]<0)?0:output_conv3[i][j][k];
			}
		}
	}
	*/
	//printf("Conv3 has been finished!\n");
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 4!!!!!!!!  
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.96875, 32) ;
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.984375, 64);
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Conv4
	
	Conv_ReLU_quan_SC_Sobol // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048,
	Sobol_seq,
	len,
	4
	//3,0.990234375, 1024
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				output_conv4[i][j][k]=(output_conv4[i][j][k]<0)?0:output_conv4[i][j][k];
			}
		}
	}
	*/
	//printf("Conv4 has been finished!\n");
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 4!!!!!!!! 
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.875, 8);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9375, 16);
				output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.96875, 32) ;
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.984375, 64);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling2
	max_pooling
	(
	10, // Height of input feature map
	10, // Width of input feature map
	2,
	2,
	5, // Height of output feature map: H_out=H/strideH
	5, // Width of output feature map: W_out=W/strideW
	64, // Number of channels
	output_conv4, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<64;i++)
	{	
		printf("i=%d:\n",i);
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
				printf("%lf ",output_pooling2[i][j][k]);
			printf("\n");
		}
	}
	*/
	// FC1
	ii=0;
	for(i=0;i<64;i++)
	{
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC_ReLU_quan_SC_Sobol // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	7,0.99951171875, 2048,
	Sobol_seq,
	len,
	4
	//7,0.990234375, 1024
	);
	/*
	FC_ReLU // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	*/
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 128!!!!!!!! 
	for(i=0;i<512;i++)
		//output_fc1[i]=quan_i_d(output_fc1[i], 7,0.984375, 64);
		output_fc1[i]=quan_i_d(output_fc1[i], 7,0.9375, 16);
	
	
	
	//for(i=0;i<512;i++)
		//printf("%lf ",output_fc1[i]);
	//printf("%lf",output_fc1[509]);
	//printf("\n");
	/*
	//ReLU
	for(i=0;i<512;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	*/
	
	// FC2
	
	FC_quan2_SC_Sobol // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	//15,0.99951171875, 2048
	15,0.990234375, 1024,
	Sobol_seq,
	len,
	8
	//15,0.998046875, 512
	);
	/*
	FC // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	*/
	//printf("The final output:\n");
	//for(i=0;i<10;i++)
		//printf("%lf ",output_fc2[i]);
	
	
	//Quantize the activations 
	for(i=0;i<10;i++)
		//output_fc2[i]=quan_i_d(output_fc2[i], 15,0.9921875, 128);
		output_fc2[i]=quan_i_d(output_fc2[i] , 15,0.990234375, 1024);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc2[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc2[i]>max)
		{
			max=output_fc2[i];
			prediction=i;
		}
	}
	
	
	
	return prediction;
}

int model_cifar10_sc_quan_sobol_all_replace_clip
(
	double ***data, // Input picture 3*32*32
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double ***output_conv1,//32*30*30
	double ***output_conv2,//32*28*28
	double ***output_pooling1,//32*14*14
	double ***output_conv3,//64*12*12
	double ***output_conv4,//32*10*10
	double ***output_pooling2,//64*5*5
	double *input_fc1, //1600
	double *output_fc1, //512
	double *output_fc2, //10
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	//for(i=0;i<32;i++)
		//printf("%lf ",data[0][0][i]);
	// Conv1
	
	Conv_ReLU_quan_SC_Sobol_Clip // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048,
	Sobol_seq,
	len,
	4
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	*/
	/*
	for(i=0;i<1;i++)
	{
		for(j=0;j<30;j++)
		{
			printf("%lf ",output_conv1[0][i][j]);
		}
		printf("\n");
	}
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				output_conv1[i][j][k]=(output_conv1[i][j][k]<0)?0:output_conv1[i][j][k];
			}
		}
	}
	*/
	
	//Quantize the activations
	//!!!!scaled by 2!!!!!!!! 
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				//output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.9921875, 128);
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 0,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	//printf("Conv1 has been finished!\n");
	// Conv2
	
	Conv_ReLU_quan_SC_Sobol_Clip // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048,
	Sobol_seq,
	len,
	1
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k]=(output_conv2[i][j][k]<0)?0:output_conv2[i][j][k];
			}
		}
	}
	*/
	//printf("Conv2 has been finished!\n");
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 2!!!!!!!! 
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 0,0.9921875, 128);
				//output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling1
	max_pooling
	(
	28, // Height of input feature map
	28, // Width of input feature map
	2,
	2,
	14, // Height of output feature map: H_out=H/strideH
	14, // Width of output feature map: W_out=W/strideW
	32, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	
	
	// Conv3
	
	Conv_ReLU_quan_SC_Sobol_Clip // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048,
	Sobol_seq,
	len,
	1
	//3,0.990234375, 1024
	//3,0.998046875, 512
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k]=(output_conv3[i][j][k]<0)?0:output_conv3[i][j][k];
			}
		}
	}
	*/
	//printf("Conv3 has been finished!\n");
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 4!!!!!!!!  
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 0,0.9921875, 128) ;
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.984375, 64);
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Conv4
	
	Conv_ReLU_quan_SC_Sobol_Clip // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048,
	Sobol_seq,
	len,
	1
	//3,0.990234375, 1024
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				output_conv4[i][j][k]=(output_conv4[i][j][k]<0)?0:output_conv4[i][j][k];
			}
		}
	}
	*/
	//printf("Conv4 has been finished!\n");
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 4!!!!!!!! 
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.875, 8);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9375, 16);
				output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 0,0.9921875, 128) ;
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.984375, 64);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling2
	max_pooling
	(
	10, // Height of input feature map
	10, // Width of input feature map
	2,
	2,
	5, // Height of output feature map: H_out=H/strideH
	5, // Width of output feature map: W_out=W/strideW
	64, // Number of channels
	output_conv4, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<64;i++)
	{	
		printf("i=%d:\n",i);
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
				printf("%lf ",output_pooling2[i][j][k]);
			printf("\n");
		}
	}
	*/
	// FC1
	ii=0;
	for(i=0;i<64;i++)
	{
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC_ReLU_quan_SC_Sobol_Clip // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	7,0.99951171875, 2048,
	Sobol_seq,
	len,
	1
	//7,0.990234375, 1024
	);
	/*
	FC_ReLU // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	*/
	
	//Quantize the activations 
	//!!!!left shift and result is scaled by 128!!!!!!!! 
	for(i=0;i<512;i++)
		//output_fc1[i]=quan_i_d(output_fc1[i], 7,0.984375, 64);
		output_fc1[i]=quan_i_d(output_fc1[i], 0,0.9921875, 128);
	
	
	
	//for(i=0;i<512;i++)
		//printf("%lf ",output_fc1[i]);
	//printf("%lf",output_fc1[509]);
	//printf("\n");
	/*
	//ReLU
	for(i=0;i<512;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	*/
	
	// FC2
	
	FC_quan2_SC_Sobol_Clip // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	//15,0.99951171875, 2048
	15,0.990234375, 1024,
	Sobol_seq,
	len,
	1
	//15,0.998046875, 512
	);
	/*
	FC // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	*/
	//printf("The final output:\n");
	//for(i=0;i<10;i++)
		//printf("%lf ",output_fc2[i]);
	
	
	//Quantize the activations 
	// for(i=0;i<10;i++)
	// 	//output_fc2[i]=quan_i_d(output_fc2[i], 15,0.9921875, 128);
	// 	output_fc2[i]=quan_i_d(output_fc2[i] , 15,0.990234375, 1024);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc2[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc2[i]>max)
		{
			max=output_fc2[i];
			prediction=i;
		}
	}
	
	
	
	return prediction;
}


int model_cifar10_sc_quan_sobol_layer1_replace
(
	double ***data, // Input picture 3*32*32
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc,
	double ***output_conv1,//32*30*30
	double ***output_conv2,//32*28*28
	double ***output_pooling1,//32*14*14
	double ***output_conv3,//64*12*12
	double ***output_conv4,//32*10*10
	double ***output_pooling2,//64*5*5
	double *input_fc1, //1600
	double *output_fc1, //512
	double *output_fc2, //10
	double **Sobol_seq,
	int len
)
{
	int i,j,k,ii,jj,kk;
	//for(i=0;i<32;i++)
		//printf("%lf ",data[0][0][i]);
	// Conv1
	
	Conv_ReLU_quan_SC_Sobol // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048,
	Sobol_seq,
	len,
	4
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	32, // Height of input feature map
	32, // Width of input feature map
	30, // Height of output feature map: H_out=(H-(m-stride))/stride
	30, // Width of output feature map: W_out=(W-(n-stride))/stride
	3, // Number of input channels
	32, // Number of output channels
	weights_conv[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[0], // num_out_channel biases
	1,
	data, // num_in_channel*H*W
	output_conv1 // num_out_channel*H_out*W_out
	);
	*/
	/*
	for(i=0;i<1;i++)
	{
		for(j=0;j<30;j++)
		{
			printf("%lf ",output_conv1[0][i][j]);
		}
		printf("\n");
	}
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				output_conv1[i][j][k]=(output_conv1[i][j][k]<0)?0:output_conv1[i][j][k];
			}
		}
	}
	*/
	
	//Quantize the activations
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
		{
			for(k=0;k<30;k++)
			{
				//output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.9921875, 128);
				output_conv1[i][j][k] = quan_i_d(output_conv1[i][j][k], 1,0.984375, 64);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	//printf("Conv1 has been finished!\n");
	// Conv2
	
	Conv_ReLU_quan // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2, // num_out_channel*H_out*W_out
	1,0.99951171875, 2048
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	30, // Height of input feature map
	30, // Width of input feature map
	28, // Height of output feature map: H_out=(H-(m-stride))/stride
	28, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	32, // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel*m*n
	bias_conv[1], // num_out_channel biases
	1,
	output_conv1, // num_in_channel*H*W
	output_conv2 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k]=(output_conv2[i][j][k]<0)?0:output_conv2[i][j][k];
			}
		}
	}
	*/
	//printf("Conv2 has been finished!\n");
	
	//Quantize the activations 
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
		{
			for(k=0;k<28;k++)
			{
				output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.984375, 64);
				//output_conv2[i][j][k] = quan_i_d(output_conv2[i][j][k], 1,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling1
	max_pooling
	(
	28, // Height of input feature map
	28, // Width of input feature map
	2,
	2,
	14, // Height of output feature map: H_out=H/strideH
	14, // Width of output feature map: W_out=W/strideW
	32, // Number of channels
	output_conv2, // num_channel*H*W
	output_pooling1 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<14;i++)
	{
		for(j=0;j<14;j++)
		{
			printf("%lf ",output_pooling1[0][i][j]);
		}
		printf("\n");
	}
	*/
	
	
	// Conv3
	
	Conv_ReLU_quan // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048
	//3,0.990234375, 1024
	//3,0.998046875, 512
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	14, // Height of input feature map
	14, // Width of input feature map
	12, // Height of output feature map: H_out=(H-(m-stride))/stride
	12, // Width of output feature map: W_out=(W-(n-stride))/stride
	32, // Number of input channels
	64, // Number of output channels
	weights_conv[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[2], // num_out_channel biases
	1,
	output_pooling1, // num_in_channel*H*W
	output_conv3 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k]=(output_conv3[i][j][k]<0)?0:output_conv3[i][j][k];
			}
		}
	}
	*/
	//printf("Conv3 has been finished!\n");
	
	//Quantize the activations 
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
		{
			for(k=0;k<12;k++)
			{
				output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.96875, 32);
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.984375, 64);
				//output_conv3[i][j][k] = quan_i_d(output_conv3[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Conv4
	
	Conv_ReLU_quan // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4, // num_out_channel*H_out*W_out
	3,0.99951171875, 2048
	//3,0.990234375, 1024
	);
	
	/*
	Conv_ReLU // Convolutional layer
	(
	3, // Kernel size is m*n
	3,
	12, // Height of input feature map
	12, // Width of input feature map
	10, // Height of output feature map: H_out=(H-(m-stride))/stride
	10, // Width of output feature map: W_out=(W-(n-stride))/stride
	64, // Number of input channels
	64, // Number of output channels
	weights_conv[3], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[3], // num_out_channel biases
	1,
	output_conv3, // num_in_channel*H*W
	output_conv4 // num_out_channel*H_out*W_out
	);
	*/
	/*
	//ReLU
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				output_conv4[i][j][k]=(output_conv4[i][j][k]<0)?0:output_conv4[i][j][k];
			}
		}
	}
	*/
	//printf("Conv4 has been finished!\n");
	
	//Quantize the activations 
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
		{
			for(k=0;k<10;k++)
			{
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.875, 8);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9375, 16);
				output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.96875, 32);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.984375, 64);
				//output_conv4[i][j][k] = quan_i_d(output_conv4[i][j][k], 3,0.9921875, 128);
				//if(output_conv1[i][j][k]!=0 && i==0  && j==16 && k==14)
					//printf("output_conv1[%d][%d][%d]=%lf\n",i,j,k,output_conv1[i][j][k]);
			}
		}
	}
	
	
	// Pooling2
	max_pooling
	(
	10, // Height of input feature map
	10, // Width of input feature map
	2,
	2,
	5, // Height of output feature map: H_out=H/strideH
	5, // Width of output feature map: W_out=W/strideW
	64, // Number of channels
	output_conv4, // num_channel*H*W
	output_pooling2 // num_channel*H_out*W_out
	);
	/*
	for(i=0;i<64;i++)
	{	
		printf("i=%d:\n",i);
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
				printf("%lf ",output_pooling2[i][j][k]);
			printf("\n");
		}
	}
	*/
	// FC1
	ii=0;
	for(i=0;i<64;i++)
	{
		for(j=0;j<5;j++)
		{
			for(k=0;k<5;k++)
			{
				input_fc1[ii]=output_pooling2[i][j][k];
				ii++;
			}
		}
	}
	
	FC_ReLU_quan // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1,
	7,0.99951171875, 2048
	//7,0.990234375, 1024
	);
	/*
	FC_ReLU // Fully connection
	(
	1600, //The number of input neurons
	512, // The number of output neurons
	weights_fc[0], // Weight array with size out_neuron*in_neuron
	bias_fc[0], // out_neuron biases
	input_fc1,
	output_fc1
	);
	*/
	
	//Quantize the activations 
	for(i=0;i<512;i++)
		//output_fc1[i]=quan_i_d(output_fc1[i], 7,0.984375, 64);
		output_fc1[i]=quan_i_d(output_fc1[i], 7,0.9375, 16);
	
	
	
	//for(i=0;i<512;i++)
		//printf("%lf ",output_fc1[i]);
	//printf("%lf",output_fc1[509]);
	//printf("\n");
	/*
	//ReLU
	for(i=0;i<512;i++)
		output_fc1[i]=(output_fc1[i]<0)?0:output_fc1[i];
	*/
	
	// FC2
	
	FC_quan2 // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2,
	//15,0.99951171875, 2048
	15,0.990234375, 1024
	//15,0.998046875, 512
	);
	/*
	FC // Fully connection
	(
	512, //The number of input neurons
	10, // The number of output neurons
	weights_fc[1], // Weight array with size out_neuron*in_neuron
	bias_fc[1], // out_neuron biases
	output_fc1,
	output_fc2
	);
	*/
	//printf("The final output:\n");
	//for(i=0;i<10;i++)
		//printf("%lf ",output_fc2[i]);
	
	
	//Quantize the activations 
	for(i=0;i<10;i++)
		//output_fc2[i]=quan_i_d(output_fc2[i], 15,0.9921875, 128);
		output_fc2[i]=quan_i_d(output_fc2[i], 15,0.990234375, 1024);
	
	// Make prediction
	int prediction;
	prediction=0;
	double max=output_fc2[0];
	for(i=1;i<10;i++)
	{
		//printf("max=%lf,output_fc3[%d]=%lf\n",max,i,output_fc3[i]);
		if(output_fc2[i]>max)
		{
			max=output_fc2[i];
			prediction=i;
		}
	}
	
	
	
	return prediction;
}




// int FireChen
// (
// 	int num_in_channel,
// 	int num_out_channel,
// 	int H_in, // Height of input feature map
// 	int W_in, // Width of input feature map
// 	int H_out, // Height of output feature map
// 	int W_out, // Width of output feature map
// 	int s11, // Number of 1*1 squeeze filters
// 	int e11, // Number of 1*1 expand filters
// 	int e33, // Number of 3*3 expand filters
// 	double *****weight, // Weights of the three kinds of filters
// 	double **bias, // Biases of the three kinds of filters
// 	double ***input, // Input feature map
// 	double ***output // Output feature map
// )
// {
// 	int i,j,k,ii,jj,kk;
	
	
// 	// Squeeze convolutional kernals, which have been combined with BN
// 	double ***output_s11;
// 	output_s11=(double ***)malloc(sizeof(double **)*s11);
// 	for(i=0;i<s11;i++)
// 	{
// 		output_s11[i]=(double **)malloc(sizeof(double *)*H_in);
// 		for(j=0;j<H_in;j++)
// 			output_s11[i][j]=(double *)malloc(sizeof(double )*W_in);
// 	}
	
// 	Conv // Convolutional layer
// 	(
// 	1, // Kernel size is m*n
// 	1,
// 	H_in, // Height of input feature map
// 	H_out, // Width of input feature map
// 	H_in, // Height of output feature map
// 	H_out, // Width of output feature map
// 	num_in_channel, // Number of input channels
// 	s11, // Number of output channels
// 	weight[0], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
// 	bias[0], // num_out_channel biases
// 	1,
// 	input, // num_in_channel*H*W
// 	output_s11 // num_out_channel*H_out*W_out
// 	);
	
// 	for(i=0;i<s11;i++)
// 	{
// 		for(j=0;j<H_in;j++)
// 		{
// 			for(k=0;k<W_in;k++)
// 			{
// 				if(output_s11[i][j][k]<0)
// 					output_s11[i][j][k]=0;
// 			}
// 		}
// 	}
	
	
	
// 	// Expand 1*1 convolutional kernals, which have been combined with BN
// 	double ***output_e11;
// 	output_e11=(double ***)malloc(sizeof(double **)*e11);
// 	for(i=0;i<e11;i++)
// 	{
// 		output_e11[i]=(double **)malloc(sizeof(double *)*H_in);
// 		for(j=0;j<H_in;j++)
// 			output_e11[i][j]=(double *)malloc(sizeof(double )*W_in);
// 	}
	
// 	Conv // Convolutional layer
// 	(
// 	1, // Kernel size is m*n
// 	1,
// 	H_in, // Height of input feature map
// 	H_out, // Width of input feature map
// 	H_in, // Height of output feature map
// 	H_out, // Width of output feature map
// 	s11, // Number of input channels
// 	e11, // Number of output channels
// 	weight[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
// 	bias[1], // num_out_channel biases
// 	1,
// 	output_s11, // num_in_channel*H*W
// 	output_e11 // num_out_channel*H_out*W_out
// 	);
	
// 	for(i=0;i<e11;i++)
// 	{
// 		for(j=0;j<H_in;j++)
// 		{
// 			for(k=0;k<W_in;k++)
// 			{
// 				if(output_e11[i][j][k]<0)
// 					output_e11[i][j][k]=0;
// 			}
// 		}
// 	}
	
	
	
// 	//Padding
// 	double ***output_s11_pad; // The input for 3*3 expand convolutional kernals is padded
// 	output_s11_pad=(double ***)malloc(sizeof(double **)*s11);
// 	for(i=0;i<s11;i++)
// 	{
// 		output_s11_pad[i]=(double **)malloc(sizeof(double *)*(H_in+2));
// 		for(j=0;j<H_in+2;j++)
// 		{
// 			output_s11_pad[i][j]=(double *)malloc(sizeof(double )*(W_in+2));
// 			if(j==0 || j==H_in+1)
// 			{
// 				for(k=0;k<W_in+2;k++)
// 					output_s11_pad[i][j][k]=0;
// 			}
// 			else
// 			{
// 				for(k=0;k<W_in+2;k++)
// 				{
// 					if(k==0 || k==W_in+1)
// 						output_s11_pad[i][j][k]=0;
// 					else
// 						output_s11_pad[i][j][k]=output_s11[i][j-1][k-1];
// 				}
// 			}
// 		}
// 	}
	
	
// 	// Expand 3*3 convolutional kernals, which have been combined with BN
// 	double ***output_e33;
// 	output_e33=(double ***)malloc(sizeof(double **)*e33);
// 	for(i=0;i<e33;i++)
// 	{
// 		output_e33[i]=(double **)malloc(sizeof(double *)*H_in);
// 		for(j=0;j<H_in;j++)
// 			output_e33[i][j]=(double *)malloc(sizeof(double )*W_in);
// 	}
	
// 	Conv // Convolutional layer
// 	(
// 	3, // Kernel size is m*n
// 	3,
// 	H_in+2, // Height of input feature map
// 	H_out+2, // Width of input feature map
// 	H_in, // Height of output feature map
// 	H_out, // Width of output feature map
// 	s11, // Number of input channels
// 	e33, // Number of output channels
// 	weight[2], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
// 	bias[2], // num_out_channel biases
// 	1,
// 	output_s11_pad, // num_in_channel*H*W
// 	output_e33 // num_out_channel*H_out*W_out
// 	);
	
// 	for(i=0;i<e33;i++)
// 	{
// 		for(j=0;j<H_in;j++)
// 		{
// 			for(k=0;k<W_in;k++)
// 			{
// 				if(output_e33[i][j][k]<0)
// 					output_e33[i][j][k]=0;
// 			}
// 		}
// 	}
	
	
// 	//Concatenate
// 	for(i=0;i<num_out_channel;i++)
// 	{
// 		if(i<e11)
// 		{
// 			for(j=0;j<H_in;j++)
// 				for(k=0;k<W_in;k++)
// 					output[i][j][k]=output_e11[i][j][k];
// 		}
// 		else
// 		{
// 			for(j=0;j<H_in;j++)
// 				for(k=0;k<W_in;k++)
// 					output[i][j][k]=output_e33[i-e11][j][k];
// 		}
// 	}
	
// 	for(i=0;i<s11;i++)
// 	{
// 		for(j=0;j<H_in;j++)
// 			free(output_s11[i][j]);
// 		free(output_s11[i]);
// 	}
// 	free(output_s11);
	
// 	for(i=0;i<e11;i++)
// 	{
// 		for(j=0;j<H_in;j++)
// 			free(output_e11[i][j]);
// 		free(output_e11[i]);
// 	}
// 	free(output_e11);
	
// 	for(i=0;i<e33;i++)
// 	{
// 		for(j=0;j<H_in;j++)
// 			free(output_e33[i][j]);
// 		free(output_e33[i]);
// 	}
// 	free(output_e33);
	
// 	for(i=0;i<s11;i++)
// 	{
// 		for(j=0;j<H_in+2;j++)
// 			free(output_s11_pad[i][j]);
// 		free(output_s11_pad[i]);
// 	}
// 	free(output_s11_pad);
	
// 	return 0;
// }







