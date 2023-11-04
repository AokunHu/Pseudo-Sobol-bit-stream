#include<math.h>
#include<string.h>
#include<time.h>
#include<stdio.h>
#include<malloc.h>
#include<stdlib.h>

// #include"MWC.h"
#include"load_dataset.h"
#include"load_param.h"
#include"layers.h"
//include"quan.h"
#include"model.h"
// #include"generate_RTL.h"





int run_core
(
	int start_idx,
	int end_idx,
	double ****input_data,
	int *label,
	double *****weights_conv, // 4 convolutional layers which have been combined with BN (see Chen's TCASI)
	double **bias_conv,// 4 convolutional layers
	double ******weight_conv_fire, // convolutional layers for fire modules: 4*4*convolutional kernelH
	double ***bias_conv_fire // convolutional layers for fire modules
)
{
	int i, j, k,prediction,err;

	
	
	err=0;
	for (i = start_idx; i <= end_idx; i++)
	{
		
		
		prediction=SqueezeNetChen(input_data[i],weights_conv,bias_conv,weight_conv_fire,bias_conv_fire);
		//if(start_idx==0)printf("%d: pred=%d,label=%d\n",i,prediction,label[i]);
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
	int i, j,k,l,ii,jj,kk,layer,cnt;
	int num_pic=10000; // Number of pictures
	int inputH=32;
	int inputW=32;
	
	int num_conv=4; // Number of convolutional layers, which have been combined with BN
	int *num_out_channel_conv,*num_in_channel_conv; // Number of channels for the convolutional layers
	int *kernelH_conv,*kernelW_conv; // Size of kernels of the convolutional layers
	num_out_channel_conv=(int *)malloc(sizeof(int)*num_conv);
	num_out_channel_conv[0]=96;
	num_out_channel_conv[1]=180;
	num_out_channel_conv[2]=120;
	num_out_channel_conv[3]=10;
	num_in_channel_conv=(int *)malloc(sizeof(int)*num_conv);
	num_in_channel_conv[0]=3;
	num_in_channel_conv[1]=360;
	num_in_channel_conv[2]=180;
	num_in_channel_conv[3]=120;
	kernelH_conv=(int *)malloc(sizeof(int)*num_conv);
	kernelH_conv[0]=2;
	kernelH_conv[1]=1;
	kernelH_conv[2]=1;
	kernelH_conv[3]=1;
	kernelW_conv=(int *)malloc(sizeof(int)*num_conv);
	kernelW_conv[0]=2;
	kernelW_conv[1]=1;
	kernelW_conv[2]=1;
	kernelW_conv[3]=1;
	
	int num_fire=4; // Number of fire modules
	int *s11,*e11,*e33; // Number of Squeeze 1*1 convolutional kernels, Expand 1*1 convolutional kernels, and Expand 3*3 convolutional kernels for the fire modules
	s11=(int *)malloc(sizeof(int)*num_fire);
	e11=(int *)malloc(sizeof(int)*num_fire);
	e33=(int *)malloc(sizeof(int)*num_fire);
	s11[0]=16;
	e11[0]=32;
	e33[0]=32;
	s11[1]=32;
	e11[1]=64;
	e33[1]=64;
	s11[2]=64;
	e11[2]=128;
	e33[2]=128;
	s11[3]=96;
	e11[3]=180;
	e33[3]=180;
		
	
	
	// Allocate space for test_data, weights, bias
	double ****input_data; // array of size 10000*3*32*32
	int *label;
	double *****weights_conv; // 4 convolutional layers which have been combined with BN (see Chen's TCASI)
	double **bias_conv;// 4 convolutional layers
	double ******weight_conv_fire; // convolutional layers for fire modules: 4*4*convolutional kernelH
	double ***bias_conv_fire; // convolutional layers for fire modules
	
	
	input_data=(double ****)malloc(sizeof(double ***)*num_pic);
	for(i=0;i<num_pic;i++)
	{
		input_data[i]=(double ***)malloc(sizeof(double **)*3);
		for(j=0;j<3;j++)
		{
			input_data[i][j]=(double **)malloc(sizeof(double *)*inputH);
			for(k=0;k<inputH;k++)
			{
				input_data[i][j][k]=(double *)malloc(sizeof(double )*inputW);
			}
		}
	}
	label=(int*)malloc(sizeof(int)*num_pic);
	
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
	
	int num_input_channel_s11;
	weight_conv_fire = (double ******)malloc(sizeof(double *****)*num_fire);
	bias_conv_fire = (double ***)malloc(sizeof(double **)*num_fire);
	for(i=0;i<num_fire;i++)
	{
		if(i==0)
			num_input_channel_s11=96;
		else if(i==1)
			num_input_channel_s11=64;
		else if(i==2)
			num_input_channel_s11=128;
		else if(i==3)
			num_input_channel_s11=256;
		
		weight_conv_fire[i] = (double *****)malloc(sizeof(double ****)*3);
		bias_conv_fire[i] = (double **)malloc(sizeof(double *)*3);
		// weight_conv_fire[0] for s11, weight_conv_fire[1] for e11, weight_conv_fire[2] for e33
		weight_conv_fire[i][0] = (double ****)malloc(sizeof(double ***)*s11[i]);
		bias_conv_fire[i][0] = (double *)malloc(sizeof(double)*s11[i]);
		for(j=0;j<s11[i];j++)
		{
			weight_conv_fire[i][0][j] = (double ***)malloc(sizeof(double **)*num_input_channel_s11);
			for(k=0;k<num_input_channel_s11;k++)
			{
				weight_conv_fire[i][0][j][k]=(double **)malloc(sizeof(double *)*1);
				weight_conv_fire[i][0][j][k][0]=(double *)malloc(sizeof(double)*1);
			}
			
		}
		
		weight_conv_fire[i][1] = (double ****)malloc(sizeof(double ***)*e11[i]);
		bias_conv_fire[i][1] = (double *)malloc(sizeof(double)*e11[i]);
		for(j=0;j<e11[i];j++)
		{
			weight_conv_fire[i][1][j] = (double ***)malloc(sizeof(double **)*s11[i]);
			for(k=0;k<s11[i];k++)
			{
				weight_conv_fire[i][1][j][k] = (double **)malloc(sizeof(double *)*1);
				weight_conv_fire[i][1][j][k][0]=(double *)malloc(sizeof(double)*1);
			}
		}
		
		weight_conv_fire[i][2] = (double ****)malloc(sizeof(double ***)*e33[i]);
		bias_conv_fire[i][2] = (double *)malloc(sizeof(double)*e33[i]);
		for(j=0;j<e33[i];j++)
		{
			weight_conv_fire[i][2][j] = (double ***)malloc(sizeof(double **)*s11[i]);
			for(k=0;k<s11[i];k++)
			{
				weight_conv_fire[i][2][j][k] = (double **)malloc(sizeof(double *)*3);
				for(l=0;l<3;l++)
					weight_conv_fire[i][2][j][k][l]=(double *)malloc(sizeof(double)*3);
			}
		}
	}
	
	

	// Load data and parameters
	load_Cifar10_test_dataset(input_data,label);
	load_param_fusedSqueezeNetChen
	(
		num_out_channel_conv,num_in_channel_conv, // Number of channels for the convolutional layers
		kernelH_conv,kernelW_conv, // Size of kernels of the convolutional layers
		num_fire, // Number of fire modules
		s11,e11,e33, // Number of Squeeze 1*1 convolutional kernels, Expand 1*1 convolutional kernels, and Expand 3*3 convolutional kernels for the fire modules
		weights_conv, // 4 convolutional layers which have been combined with BN (see Chen's TCASI)
		bias_conv, // 4 convolutional layers
		weight_conv_fire, // convolutional layers for fire modules: 4*4*convolutional kernelH
		bias_conv_fire // convolutional layers for fire modules
	);
	
	
	
	// For parrellizing the simulations
	int err_img;
	int core_num=50,core;
	int *err_core;
	err_core=(int*)malloc(sizeof(int)*core_num);

	for(core=0;core<core_num;core++)
		err_core[core]=0;
	    

	
	printf("Inference:\n");	
	
	#pragma omp parallel for
	for(core = 0; core < core_num; core++)
		err_core[core] = run_core(core*num_pic/core_num,(core+1)*num_pic/core_num-1,input_data,label,weights_conv,bias_conv,weight_conv_fire,bias_conv_fire);
	
	
	//err_core[0] = run_core(3869,3869,input_data,label,weights_conv,bias_conv,weights_fc,bias_fc);
	
	
	err_img=0;
	for(core=0;core<core_num;core++)
		err_img+=err_core[core];
	printf("err_img=%d,Accuracy: %lf %%\n",err_img, (double)(num_pic - err_img) / 100);

	
	

	// Free space for test_data, weights, bias
	
	
	for(i=0;i<num_pic;i++)
	{
		for(j=0;j<3;j++)
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
	free(label);
	
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
	
	for(i=0;i<num_fire;i++)
	{
		if(i==0)
			num_input_channel_s11=96;
		else if(i==1)
			num_input_channel_s11=64;
		else if(i==2)
			num_input_channel_s11=128;
		else if(i==3)
			num_input_channel_s11=256;
		
		for(j=0;j<s11[i];j++)
		{
			for(k=0;k<num_input_channel_s11;k++)
			{
				free(weight_conv_fire[i][0][j][k][0]);
				free(weight_conv_fire[i][0][j][k]);
			}
			free(weight_conv_fire[i][0][j]);
		}
		free(weight_conv_fire[i][0]);
		
		for(j=0;j<e11[i];j++)
		{
			for(k=0;k<s11[i];k++)
			{
				free(weight_conv_fire[i][1][j][k][0]);
				free(weight_conv_fire[i][1][j][k]);
			}
			free(weight_conv_fire[i][1][j]);
		}
		free(weight_conv_fire[i][1]);
		
		for(j=0;j<e33[i];j++)
		{
			for(k=0;k<s11[i];k++)
			{
				for(l=0;l<3;l++)
					free(weight_conv_fire[i][2][j][k][l]);
				free(weight_conv_fire[i][2][j][k]);
			}
			free(weight_conv_fire[i][2][j]);
		}
		free(weight_conv_fire[i][2]);
		
		free(weight_conv_fire[i]);
	}
	free(weight_conv_fire);
	
	free(err_core);
	
	free(num_out_channel_conv);
	free(num_in_channel_conv);
	free(kernelH_conv);
	free(kernelW_conv);
	free(s11);
	free(e11);
	free(e33);
	
	end_time=time(NULL);
	
	printf("The whole processing concumes %lfs\n",difftime(end_time,start_time));
	return 0;
}