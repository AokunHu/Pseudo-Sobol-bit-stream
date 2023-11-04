#include<math.h>
#include<string.h>
#include<time.h>
#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>

// #include"MWC.h"
#include"load_dataset.h"
#include"load_param.h"
#include"quan.h"
#include"layers.h"
#include"model.h"
// #include"generate_RTL.h"





int run_core
(
	int start_idx,
	int end_idx,
	double ****input_data,
	int *label,
	double *****weights_conv, // 4 convolutional layers
	double **bias_conv,// 4 convolutional layers
    double ***output_final
)
{
	int i, j, k,prediction,err;

	register double ***output_conv1;//64*56*56
	output_conv1=(double ***)malloc(sizeof(double **)*64);
	for(i=0;i<64;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*56);
		for(j=0;j<56;j++)
			output_conv1[i][j]=(double *)malloc(sizeof(double)*56);
	}
	register double ***output_conv2;//32*28*28
	output_conv2=(double ***)malloc(sizeof(double **)*64);
	for(i=0;i<64;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*56);
		for(j=0;j<56;j++)
			output_conv2[i][j]=(double *)malloc(sizeof(double)*56);
	}
	
	
   


	err=0;
	for (i = start_idx; i <= end_idx; i++)
	{

		prediction=model_soc(input_data[i],weights_conv,bias_conv,output_conv1,output_conv2);
		
	}
    
   
    for(i=0;i<64;i++)
	{
		for(j=0;j<56;j++){
            for(k=0;k<56;j++){
                output_final[i][j][k]=output_conv1[i][j][k]+output_conv2[i+1][j+1][k]+input_data[0][i][j][k];
            }
        }
			
	}


	//free
	for(i=0;i<64;i++)
	{
		for(j=0;j<56;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<64;i++)
	{
		for(j=0;j<56;j++)
			free(output_conv2[i][j]);
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	
	return 0;
}

int main(void)
{
	time_t start_time,end_time;
	start_time=time(NULL);
	int i, j,k,l,ii,jj,kk,layer,cnt;
	int num_pic=1; // Number of pictures
	int inputH=58;
	int inputW=58;
	
	int num_conv=2; // Number of convolutional layers
	int num_fc=0; // Number of fully connection layers
	
	int *num_out_channel_conv,*num_in_channel_conv; // Number of channels for the convolutional layers
	int *kernelH_conv,*kernelW_conv; // Size of kernels of the convolutional layers
	int *num_in_neuron,*num_out_neuron;
	
	num_out_channel_conv=(int *)malloc(sizeof(int)*num_conv);
	num_out_channel_conv[0]=64;
	num_out_channel_conv[1]=64;

	num_in_channel_conv=(int *)malloc(sizeof(int)*num_conv);
	num_in_channel_conv[0]=64;
	num_in_channel_conv[1]=64;

	kernelH_conv=(int *)malloc(sizeof(int)*num_conv);
	kernelH_conv[0]=3;
	kernelH_conv[1]=1;

	kernelW_conv=(int *)malloc(sizeof(int)*num_conv);
	kernelW_conv[0]=3;
	kernelW_conv[1]=1;

	
		
	
	
	// Allocate space for test_data, weights, bias
	double ****input_data; // array of size 10000*3*32*32
	double *****weights_conv; // 4 convolutional layers which have been combined with BN (see Chen's TCASI)
	double **bias_conv;// 4 convolutional layers
	double ***weights_fc;
	double **bias_fc;
    double ***output_final;
	
	input_data=(double ****)malloc(sizeof(double ***)*num_pic);
	for(i=0;i<num_pic;i++)
	{
		input_data[i]=(double ***)malloc(sizeof(double **)*64);
		for(j=0;j<64;j++)
		{
			input_data[i][j]=(double **)malloc(sizeof(double *)*inputH);
			for(k=0;k<inputH;k++)
			{
				input_data[i][j][k]=(double *)malloc(sizeof(double )*inputW);
			}
		}
	}

	
	weights_conv=(double *****)malloc(sizeof(double ****)*num_conv);
	for(i=0;i<num_conv;i++)
	{
		weights_conv[i]=(double ****)malloc(sizeof(double ***)*num_out_channel_conv[i]);
		for(j=0;j<num_out_channel_conv[i];j++)
		{
			weights_conv[i][j]=(double ***)malloc(sizeof(double **)*num_in_channel_conv[i]);
			for(k=0;k<num_in_channel_conv[i];k++)
			{
				weights_conv[i][j][k]=(double **)malloc(sizeof(double *)*kernelH_conv[i]);
				for(ii=0;ii<kernelH_conv[i];ii++)
					weights_conv[i][j][k][ii]=(double *)malloc(sizeof(double)*kernelW_conv[i]);
			}
		}
	}
	
	bias_conv=(double **)malloc(sizeof(double *)*num_conv);
	for(i=0;i<num_conv;i++)
		bias_conv[i]=(double *)malloc(sizeof(double)*num_out_channel_conv[i]);
	
	
	output_final=(double ***)malloc(sizeof(double **)*64);
	for(j=0;j<64;j++)
	{
		output_final[j]=(double **)malloc(sizeof(double *)*56);
		for(k=0;k<56;k++)
		{
			output_final[j][k]=(double *)malloc(sizeof(double )*56);
		}
	}

	// Load data and parameters
	load_test_data(input_data);
	//for(i=0;i<32;i++)
		//printf("%lf ",input_data[0][0][0][i]);
	load_param_soc(num_out_channel_conv,num_in_channel_conv,num_in_neuron,num_out_neuron,kernelH_conv,kernelW_conv,weights_conv,bias_conv);
	
	
	/********************* Quantization *******************************/
	int integer;
	double decimal;
	int decimal_shift;
	
	//8-bit quantization for both weights and biases of all convolutional layers
	integer=0;
	decimal=0.9921875;
	//decimal=0.99609375;
	//decimal=0.998046875;
	decimal_shift=128;
	//decimal_shift=256;
	//decimal_shift=512;
	for(i=0;i<num_conv;i++)
	{
		for(j=0;j<num_out_channel_conv[i];j++)
		{
			for(k=0;k<num_in_channel_conv[i];k++)
				for(ii=0;ii<kernelH_conv[i];ii++)
					for(jj=0;jj<kernelW_conv[i];jj++)
						weights_conv[i][j][k][ii][jj]=quan_i_d(weights_conv[i][j][k][ii][jj], integer,decimal, decimal_shift);
			
			//bias_conv[i][j]=quan_i_d(bias_conv[i][j], integer,decimal, decimal_shift);
			bias_conv[i][j]=quan_i_d(bias_conv[i][j], integer,decimal=0.99609375, 256);
		}
	}
	
	
	
	
	// Quantize the input data
	integer=3;
	//decimal=0.9921875;
	//decimal_shift=128;
	//decimal=0.984375;
	//decimal_shift=64;
	decimal=0.96875;
	decimal_shift=32;
	for(i=0;i<1;i++)
	{
		for(j=0;j<64;j++)
		{
			for(k=0;k<58;k++)
			{
				for(ii=0;ii<58;ii++)
			//if(data[0][i][j]!=quan_i_d(data[0][i][j], 0,0.875, 8))
				//printf("data[%d][%d]=%lf, quan=%lf\n",i,j,data[0][i][j],quan_i_d(data[0][i][j], 0,0.875, 8));
			//if(i==16 && j==14)printf("%lf ",data[0][i][j]);
					// input_data[i][j][k][ii]=quan_i_d(input_data[i][j][k][ii], integer,decimal, decimal_shift);
					input_data[i][j][k][ii]=quan_i_d(input_data[i][j][k][ii], 0,0.99609375, 256);
			//if(i==16 && j==14)printf("%lf ",data[0][i][j]);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.9375, 16);
			//data[0][i][j]=quan_i_d(data[0][i][j], 0,0.875, 8);
			}
		}
		//printf("\n");
	}
	printf("All input data has been quantized!\n");
	
	/********************* Quantization *******************************/
	
	
	
	
	// For parrellizing the simulations
	int core_num=1,core;
	int *err_core;
	err_core=(int*)malloc(sizeof(int)*core_num);

	for(core=0;core<core_num;core++)
		err_core[core]=0;
	
	printf("Inference:\n");	
	
	#pragma omp parallel for
	for(core = 0; core < core_num; core++)
		err_core[core] = run_core(core*num_pic/core_num,(core+1)*num_pic/core_num-1,input_data,weights_conv,bias_conv,output_final);
	
	//err_core[0] = run_core(0,0,input_data,label,weights_conv,bias_conv,weights_fc,bias_fc);
	
 
	FILE *fp;
	fp = fopen("./huaokun/soc/data/output.txt", "w");
	if(fp==NULL)printf("fopen failed!\n");
	for (i = 0; i < 64; i++)
	{
		//if(i%1000==0)printf("loading========================================\n");
		for (j = 0; j < 56; j++)
		{
			for(k = 0; k < 56; k++){
                fprintf(fp, "%lf", &output_final[i][j][k]);
            }
		}
	}
	fclose(fp);
	printf("successfully write!\n");


	
	

	// Free space for test_data, weights, bias
	
	
	for(i=0;i<num_pic;i++)
	{
		for(j=0;j<64;j++)
		{
			for(k=0;k<inputH;k++)
			{
				free(input_data[i][j][k]);
			}
			free(input_data[i][j]);
		}
		free(input_data[i]);
	}
	free(input_data);

	
	for(i=0;i<num_conv;i++)
	{
		for(j=0;j<num_out_channel_conv[i];j++)
		{
			for(k=0;k<num_in_channel_conv[i];k++)
			{
				for(ii=0;ii<kernelH_conv[i];ii++)
					free(weights_conv[i][j][k][ii]);
				free(weights_conv[i][j][k]);
			}
			free(weights_conv[i][j]);
		}
		free(weights_conv[i]);
	}
	free(weights_conv);
	
	for(i=0;i<num_conv;i++)
		free(bias_conv[i]);
	free(bias_conv);
	
  
	for(j=0;j<64;j++)
	{
		for(k=0;k<56;k++)
		{
			free(output_final[j][k]);
		}
		free(output_final[j]);
	}
	free(output_final);

	
	
	free(err_core);
	
	free(num_out_channel_conv);
	free(num_in_channel_conv);
	free(num_out_neuron);
	free(num_in_neuron);
	free(kernelH_conv);
	free(kernelW_conv);

	end_time=time(NULL);
	
	printf("The whole processing concumes %lfs\n",difftime(end_time,start_time));
	return 0;
}



int load_test_data
(
	double **test_data, // array of size 10000*784, each entry of which is in [0,1]
)
{
	printf("Begin to load\n");
	int i,j;
	FILE *fp;
	fp = fopen("./huaokun/soc/data/test_data.txt", "r");
	if(fp==NULL)printf("fopen failed!\n");
	for (i = 0; i < 1; i++)
	{
		//if(i%1000==0)printf("loading========================================\n");
		for (j = 0; j < 784; j++)
		{
			//printf("%d\n",j);
			fscanf(fp, "%lf", &test_data[i][j]);
		}
	}
	fclose(fp);
	printf("successfully loaded!\n");
	
	return 0;
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

int load_param_soc
(
	int *num_out_channel_conv, int *num_in_channel_conv, // Number of channels for the convolutional layers
	int *num_in_neuron,int *num_out_neuron,
	int *kernelH_conv,int *kernelW_conv, // Size of kernels of the convolutional layers
	double *****weights_conv, // 4 convolutional layers
	double **bias_conv, // 4 convolutional layers
)
{
	int i,j;
	char filename_weight[100],filename_bias[100];
	char filename[100];
	
	printf("Begin to load parameters:\n");
	// conv1
	strcpy(filename_weight,"./huaokun/soc/data/output/conv1_weights.txt");
	strcpy(filename_bias,"./huaokun/soc/data/output/conv1_bias.txt");
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
	strcpy(filename_weight,"./huaokun/soc/data/output/conv2_weights.txt");
	strcpy(filename_bias,"./huaokun/soc/data/output/conv2_bias.txt");
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
	
	
	return 0;
}