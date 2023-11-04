#include<math.h>
#include<string.h>
#include<time.h>
#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>


#include"load_dataset.h"
#include"load_param.h"
#include"layers.h"
#include"model.h"





int run_core
(
	int start_idx,
	int end_idx,
	double ***input_data,
	int *label,
	double *****weights_conv,
	double **bias_conv,
	double ***weights_fc,
	double **bias_fc
)
{
	int i, j, k,prediction,err;
    int len=32;
	double **Sobol_seq; //Sobol_seq[4][1024];
	Sobol_seq=(double **)malloc(sizeof(double *)*4);
	for(i=0;i<4;i++)
		Sobol_seq[i]=(double *)malloc(sizeof(double)*1024);

	load_Pseudo_Sobol(Sobol_seq);
    // load_Sobol(Sobol_seq);
	// load_Pseudo_Random(Sobol_seq);
    // load_Pseudo_Sobol_only_flip(Sobol_seq);
	printf("load Sobol successfully");
	err=0;
	for (i = start_idx; i <= end_idx; i++)
	{
		
		// prediction=LeNet5_Date19_quan_sobol(input_data[i],weights_conv,bias_conv,weights_fc,bias_fc,Sobol_seq,len);
		prediction=LeNet5_Date19_quan_sobol_all_replace(input_data[i],weights_conv,bias_conv,weights_fc,bias_fc,Sobol_seq,len);
		// prediction=LeNet5_Date19_quan_sobol_conv_replace(input_data[i],weights_conv,bias_conv,weights_fc,bias_fc,Sobol_seq,len);
		// prediction=LeNet5_Date19_quan_sobol_conv_1fc_replace(input_data[i],weights_conv,bias_conv,weights_fc,bias_fc,Sobol_seq,len);
		//printf("%d: pred=%d,label=%d\n",i,prediction,label[i]);
		if (prediction != label[i])
		{
			err++;
			//printf("%d\n",i);
			//printf("%d: pred=%d,label=%d\n",i,prediction,label[i]);
			//FILE *fp=fopen("pred.txt","a");
			//fprintf(fp,"%d\n",i);
			//fclose(fp);
		}
	}

	
	
	return err;
}

int main(void)
{
	time_t start_time,end_time;
	start_time=time(NULL);
	int i, j,k,ii,jj,kk,layer,cnt;
	int num_pic=10000; // Number of pictures
	int num_input=784; // Number of input pixels
	int inputH=28;
	int inputW=28;
	int num_conv=2; // Number of convolutional layers
	int num_fc=3; // Number of fully connection layers
	int *num_out_channel,*num_in_channel,*num_out_neuron,*num_in_neuron;
	int *kernelH,*kernelW;
	
	num_out_channel=(int *)malloc(sizeof(int)*num_conv);
	num_out_channel[0]=20;
	num_out_channel[1]=20;
	
	num_in_channel=(int *)malloc(sizeof(int)*num_conv);
	num_in_channel[0]=1;
	num_in_channel[1]=20;
	
	kernelH=(int *)malloc(sizeof(int)*num_conv);
	kernelH[0]=5;
	kernelH[1]=5;
	
	kernelW=(int *)malloc(sizeof(int)*num_conv);
	kernelW[0]=5;
	kernelW[1]=5;
	
	num_in_neuron=(int *)malloc(sizeof(int)*num_fc);
	num_in_neuron[0]=320;
	num_in_neuron[1]=800;
	num_in_neuron[2]=500;
	
	num_out_neuron=(int *)malloc(sizeof(int)*num_fc);
	num_out_neuron[0]=800;
	num_out_neuron[1]=500;
	num_out_neuron[2]=10;
	
	
	// Allocate space for test_data, weights, bias
	double ***input_data; // Input picture num_pic*28*28
	double **load_data;
	int *label;
	double *****weights_conv;
	double **bias_conv;
	double ***weights_fc;
	double **bias_fc;
	
	input_data=(double ***)malloc(sizeof(double **)*num_pic);
	for(i=0;i<num_pic;i++)
	{
		input_data[i]=(double **)malloc(sizeof(double *)*inputH);
		for(j=0;j<inputH;j++)
			input_data[i][j]=(double *)malloc(sizeof(double)*inputW);
	}
	
	load_data=(double**)malloc(sizeof(double*)*num_pic);
	for(i=0;i<num_pic;i++)
		load_data[i]=(double *)malloc(sizeof(double)*num_input);
	
	label=(int*)malloc(sizeof(int)*num_pic);
	
	
	weights_conv=(double *****)malloc(sizeof(double ****)*num_conv);
	for(i=0;i<num_conv;i++)
	{
		weights_conv[i]=(double ****)malloc(sizeof(double ***)*num_out_channel[i]);
		for(j=0;j<num_out_channel[i];j++)
		{
			weights_conv[i][j]=(double ***)malloc(sizeof(double **)*num_in_channel[i]);
			for(k=0;k<num_in_channel[i];k++)
			{
				weights_conv[i][j][k]=(double **)malloc(sizeof(double *)*kernelH[i]);
				for(ii=0;ii<kernelH[i];ii++)
					weights_conv[i][j][k][ii]=(double *)malloc(sizeof(double)*kernelW[i]);
			}
		}
	}
	
	
	bias_conv=(double **)malloc(sizeof(double *)*num_conv);
	for(i=0;i<num_conv;i++)
		bias_conv[i]=(double *)malloc(sizeof(double)*num_out_channel[i]);
	
	weights_fc=(double ***)malloc(sizeof(double **)*num_fc);
	for(i=0;i<num_fc;i++)
	{
		weights_fc[i]=(double **)malloc(sizeof(double *)*num_out_neuron[i]);
		for(j=0;j<num_out_neuron[i];j++)
		{
			weights_fc[i][j]=(double *)malloc(sizeof(double)*num_in_neuron[i]);
		}
	}
	
	bias_fc=(double **)malloc(sizeof(double *)*num_fc);
	for(i=0;i<num_fc;i++)
		bias_fc[i]=(double *)malloc(sizeof(double)*num_out_neuron[i]);
	
	
	

	// Load data and parameters
	char filename_weight[100],filename_bias[100];
	char filename[100];
	load_MNIST_test_dataset(load_data,label);
	
	for(i=0;i<num_pic;i++)
	{
		cnt=0;
		//printf("i=%d\n",i);
		for(j=0;j<inputH;j++)
		{
			for(k=0;k<inputW;k++)
			{
				input_data[i][j][k]=load_data[i][cnt];
				cnt++;
			}
		}
	}
	
	
	
	printf("Begin to load parameters:\n");
	// conv1
	strcpy(filename_weight,"param_LeNet5_Date19/conv1_weights.txt");
	strcpy(filename_bias,"param_LeNet5_Date19/conv1_bias.txt");
	load_param_conv
	(
	kernelH[0], // Kernel size is m*n
	kernelW[0],
	num_in_channel[0], // Number of input channels
	num_out_channel[0], // Number of output channels
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
	strcpy(filename_weight,"param_LeNet5_Date19/conv2_weights.txt");
	strcpy(filename_bias,"param_LeNet5_Date19/conv2_bias.txt");
	load_param_conv
	(
	kernelH[1], // Kernel size is m*n
	kernelW[1],
	num_in_channel[1], // Number of input channels
	num_out_channel[1], // Number of output channels
	weights_conv[1], // The are num_out_channel filters, each of which contains num_in_channel kernels. The size of weight is num_out_channel*num_in_channel_m*n
	bias_conv[1], // num_out_channel biases
	filename_weight,
	filename_bias
	);
	printf("Parameters of conv2 have been successfully loaded!===========================\n");
	// fc1_bias
	strcpy(filename_weight,"param_LeNet5_Date19/fc1_weights.txt");
	strcpy(filename_bias,"param_LeNet5_Date19/fc1_bias.txt");
	load_param(num_in_neuron[0],num_out_neuron[0],weights_fc[0],bias_fc[0],filename_weight,filename_bias);
	printf("Parameters of fc1 have been successfully loaded!===========================\n");
	// fc2_bias
	strcpy(filename_weight,"param_LeNet5_Date19/fc2_weights.txt");
	strcpy(filename_bias,"param_LeNet5_Date19/fc2_bias.txt");
	load_param(num_in_neuron[1],num_out_neuron[1],weights_fc[1],bias_fc[1],filename_weight,filename_bias);
	printf("Parameters of fc2 have been successfully loaded!===========================\n");
	// fc3_bias
	strcpy(filename_weight,"param_LeNet5_Date19/fc3_weights.txt");
	strcpy(filename_bias,"param_LeNet5_Date19/fc3_bias.txt");
	load_param(num_in_neuron[2],num_out_neuron[2],weights_fc[2],bias_fc[2],filename_weight,filename_bias);
	printf("Parameters of fc3 have been successfully loaded!===========================\n");

	
	// For parrellizing the simulations
	int err_img;
	int core_num=40,core;
	int *err_core;
	err_core=(int*)malloc(sizeof(int)*core_num);

	for(core=0;core<core_num;core++)
		err_core[core]=0;
	    

	
	printf("Inference:\n");	
	
	#pragma omp parallel for
	for(core = 0; core < core_num; core++)
		err_core[core] = run_core(core*num_pic/core_num,(core+1)*num_pic/core_num-1,input_data,label,weights_conv,bias_conv,weights_fc,bias_fc);
	
	
	//err_core[0] = run_core(1,1,input_data,label,weights_conv,bias_conv,weights_fc,bias_fc);
	
	
	err_img=0;
	for(core=0;core<core_num;core++)
		err_img+=err_core[core];
	printf("err_img=%d,Accuracy: %lf %%\n",err_img, (double)(num_pic - err_img) / 100);

	
	

	// Free space for test_data, weights, bias
	
	for(i=0;i<num_pic;i++)
	{
		for(j=0;j<inputH;j++)
			free(input_data[i][j]);
		free(input_data[i]);
	}
	free(input_data);
	
	for(i=0;i<num_pic;i++)
		free(load_data[i]);
	free(load_data);
	
	free(label);
	
	for(i=0;i<num_conv;i++)
	{
		for(j=0;j<num_out_channel[i];j++)
		{
			for(k=0;k<num_in_channel[i];k++)
			{
				for(ii=0;ii<kernelH[i];ii++)
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
	
	for(i=0;i<num_fc;i++)
	{
		for(j=0;j<num_out_neuron[i];j++)
		{
			free(weights_fc[i][j]);
		}
		free(weights_fc[i]);
	}
	free(weights_fc);
	
	free(num_out_channel);
	free(num_in_channel);
	free(num_out_neuron);
	free(num_in_neuron);
	free(kernelH);
	free(kernelW);
	
	
	end_time=time(NULL);
	
	printf("The whole processing concumes %lfs\n",difftime(end_time,start_time));
	return 0;
}