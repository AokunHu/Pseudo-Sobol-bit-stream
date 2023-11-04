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
	double ***weights_fc, // 2 fully-connected layers
	double **bias_fc, // 2 fully-connected layers
	double **Sobol_seq,
	int len
)
{
	int i, j, k,prediction,err;

	register double ***output_conv1;//32*30*30
	output_conv1=(double ***)malloc(sizeof(double **)*32);
	for(i=0;i<32;i++)
	{
		output_conv1[i]=(double **)malloc(sizeof(double *)*30);
		for(j=0;j<30;j++)
			output_conv1[i][j]=(double *)malloc(sizeof(double)*30);
	}
	register double ***output_conv2;//32*28*28
	output_conv2=(double ***)malloc(sizeof(double **)*32);
	for(i=0;i<32;i++)
	{
		output_conv2[i]=(double **)malloc(sizeof(double *)*28);
		for(j=0;j<28;j++)
			output_conv2[i][j]=(double *)malloc(sizeof(double)*28);
	}
	
	register double ***output_pooling1;//32*14*14
	output_pooling1=(double ***)malloc(sizeof(double **)*32);
	for(i=0;i<32;i++)
	{
		output_pooling1[i]=(double **)malloc(sizeof(double *)*14);
		for(j=0;j<14;j++)
		{
			output_pooling1[i][j]=(double *)malloc(sizeof(double)*14);
		}
	}
	
	register double ***output_conv3;//64*12*12
	output_conv3=(double ***)malloc(sizeof(double **)*64);
	for(i=0;i<64;i++)
	{
		output_conv3[i]=(double **)malloc(sizeof(double *)*12);
		for(j=0;j<12;j++)
			output_conv3[i][j]=(double *)malloc(sizeof(double)*12);
	}
	register double ***output_conv4;//64*10*10
	output_conv4=(double ***)malloc(sizeof(double **)*64);
	for(i=0;i<64;i++)
	{
		output_conv4[i]=(double **)malloc(sizeof(double *)*10);
		for(j=0;j<10;j++)
			output_conv4[i][j]=(double *)malloc(sizeof(double)*10);
	}
	
	register double ***output_pooling2;//64*5*5
	output_pooling2=(double ***)malloc(sizeof(double **)*64);
	for(i=0;i<64;i++)
	{
		output_pooling2[i]=(double **)malloc(sizeof(double *)*5);
		for(j=0;j<5;j++)
		{
			output_pooling2[i][j]=(double *)malloc(sizeof(double)*5);
		}
	}
	
	double input_fc1[1600],output_fc1[512];
	double output_fc2[10];
	

   


	err=0;
	for (i = start_idx; i <= end_idx; i++)
	{
		//printf("i=%d\n",i);
		//!!!!!!modify the line-349!!!!!!!!!!
		prediction=model_cifar10_sc_quan_sobol_all_replace_clip(input_data[i],weights_conv,bias_conv,weights_fc,bias_fc,output_conv1,output_conv2,output_pooling1,output_conv3,output_conv4,output_pooling2,input_fc1,output_fc1,output_fc2,Sobol_seq,len);
		
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
		printf("start_idx: %d ,   end_index=%d , index= %d finish, errorrate=%lf\n",start_idx,end_idx,i,(double)err/(i-start_idx+1));
	}
	//free
	for(i=0;i<32;i++)
	{
		for(j=0;j<30;j++)
			free(output_conv1[i][j]);
		free(output_conv1[i]);
	}
	free(output_conv1);
	
	for(i=0;i<32;i++)
	{
		for(j=0;j<28;j++)
			free(output_conv2[i][j]);
		free(output_conv2[i]);
	}
	free(output_conv2);
	
	for(i=0;i<32;i++)
	{
		for(j=0;j<14;j++)
			free(output_pooling1[i][j]);
		free(output_pooling1[i]);
	}
	free(output_pooling1);
	
	for(i=0;i<64;i++)
	{
		for(j=0;j<12;j++)
			free(output_conv3[i][j]);
		free(output_conv3[i]);
	}
	free(output_conv3);
	
	for(i=0;i<64;i++)
	{
		for(j=0;j<10;j++)
			free(output_conv4[i][j]);
		free(output_conv4[i]);
	}
	free(output_conv4);
	
	for(i=0;i<64;i++)
	{
		for(j=0;j<5;j++)
			free(output_pooling2[i][j]);
		free(output_pooling2[i]);
	}
	free(output_pooling2);

	// for(i=0;i<4;i++)
	// 	free(Sobol_seq[i]);
	// free(Sobol_seq);
	
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
	
	int num_conv=4; // Number of convolutional layers
	int num_fc=2; // Number of fully connection layers
	
	int *num_out_channel_conv,*num_in_channel_conv; // Number of channels for the convolutional layers
	int *kernelH_conv,*kernelW_conv; // Size of kernels of the convolutional layers
	int *num_in_neuron,*num_out_neuron;
	
	num_out_channel_conv=(int *)malloc(sizeof(int)*num_conv);
	num_out_channel_conv[0]=32;
	num_out_channel_conv[1]=32;
	num_out_channel_conv[2]=64;
	num_out_channel_conv[3]=64;
	num_in_channel_conv=(int *)malloc(sizeof(int)*num_conv);
	num_in_channel_conv[0]=3;
	num_in_channel_conv[1]=32;
	num_in_channel_conv[2]=32;
	num_in_channel_conv[3]=64;
	kernelH_conv=(int *)malloc(sizeof(int)*num_conv);
	kernelH_conv[0]=3;
	kernelH_conv[1]=3;
	kernelH_conv[2]=3;
	kernelH_conv[3]=3;
	kernelW_conv=(int *)malloc(sizeof(int)*num_conv);
	kernelW_conv[0]=3;
	kernelW_conv[1]=3;
	kernelW_conv[2]=3;
	kernelW_conv[3]=3;
	
	num_in_neuron=(int *)malloc(sizeof(int)*num_fc);
	num_in_neuron[0]=1600;
	num_in_neuron[1]=512;
	
	num_out_neuron=(int *)malloc(sizeof(int)*num_fc);
	num_out_neuron[0]=512;
	num_out_neuron[1]=10;
		
	
	
	// Allocate space for test_data, weights, bias
	double ****input_data; // array of size 10000*3*32*32
	int *label;
	double *****weights_conv; // 4 convolutional layers which have been combined with BN (see Chen's TCASI)
	double **bias_conv;// 4 convolutional layers
	double ***weights_fc;
	double **bias_fc;
	
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
	load_Cifar10_test_dataset(input_data,label);
	//for(i=0;i<32;i++)
		//printf("%lf ",input_data[0][0][0][i]);
	load_param_model_cifar10(num_out_channel_conv,num_in_channel_conv,num_in_neuron,num_out_neuron,kernelH_conv,kernelW_conv,weights_conv,bias_conv,weights_fc,bias_fc);
	
	
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
	
	//8-bit quantization for both weights and biases of all fully-connected layers
	for(i=0;i<num_fc;i++)
	{
		for(j=0;j<num_out_neuron[i];j++)
		{
			for(k=0;k<num_in_neuron[i];k++)
				weights_fc[i][j][k]=quan_i_d(weights_fc[i][j][k], integer,decimal, decimal_shift);
			if(i==1 && j==1)printf("%lf\n",bias_fc[i][j]);
			//bias_fc[i][j]=quan_i_d(bias_fc[i][j], integer,decimal, decimal_shift);
			bias_fc[i][j]=quan_i_d(bias_fc[i][j], integer,decimal=0.99609375, 256);
			if(i==1 && j==1)printf("%lf\n",bias_fc[i][j]);
		}
	}
	printf("All weights and biases have been quantizaed!\n");
	
	
	// Quantize the input data
	integer=3;
	//decimal=0.9921875;
	//decimal_shift=128;
	//decimal=0.984375;
	//decimal_shift=64;
	decimal=0.96875;
	decimal_shift=32;
	for(i=0;i<10000;i++)
	{
		for(j=0;j<3;j++)
		{
			for(k=0;k<32;k++)
			{
				for(ii=0;ii<32;ii++)
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
	int err_img;
	int core_num=20,core;
	int *err_core;
	err_core=(int*)malloc(sizeof(int)*core_num);

	for(core=0;core<core_num;core++)
		err_core[core]=0;
	    
    int len=256;
	// double **Sobol_seq; //Sobol_seq[4][1024];
	// Sobol_seq=(double **)malloc(sizeof(double *)*4);
	// for(i=0;i<4;i++)
	// 	Sobol_seq[i]=(double *)malloc(sizeof(double)*1024);


    double ***Sobol_seq;
    Sobol_seq=(double ***)malloc(sizeof(double **)*core_num);
	for(i=0;i<core_num;i++)
	{
		Sobol_seq[i]=(double **)malloc(sizeof(double *)*4);
		for(j=0;j<4;j++)
		{
			Sobol_seq[i][j]=(double *)malloc(sizeof(double)*1024);
		}
	}



	// load_Pseudo_Sobol(Sobol_seq);
	for(i=0;i<core_num;i++)
	{
		// load_Sobol(Sobol_seq[i]);
		// load_Pseudo_Sobol_only_flip(Sobol_seq[i]);
		load_Pseudo_Sobol(Sobol_seq[i]);
	}
    // load_Pseudo_Sobol_only_flip(Sobol_seq);
	printf("load Sobol successfully\n"); 
	
	printf("Inference:\n");	
	
	#pragma omp parallel for
	for(core = 0; core < core_num; core++)
		err_core[core] = run_core(core*num_pic/core_num,(core+1)*num_pic/core_num-1,input_data,label,weights_conv,bias_conv,weights_fc,bias_fc,Sobol_seq[core],len);
	
	//err_core[0] = run_core(0,0,input_data,label,weights_conv,bias_conv,weights_fc,bias_fc);
	
	// for(i=0;i<4;i++)
	// 	free(Sobol_seq[i]);
	// free(Sobol_seq);
    for(i=0;i<core_num;i++)
	{
		for(j=0;j<4;j++)
			free(Sobol_seq[i][j]);
		free(Sobol_seq[i]);
	}
	free(Sobol_seq);



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
	
	
	for(i=0;i<num_fc;i++)
	{
		for(j=0;j<num_out_neuron[i];j++)
			free(weights_fc[i][j]);
		free(weights_fc[i]);
	}
	free(weights_fc);
	
	for(i=0;i<num_fc;i++)
		free(bias_fc[i]);
	free(bias_fc);
	
	
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