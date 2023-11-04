/* In this file, some loading functions are given, for specific dataset, such as MNIST. */
#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<math.h>

int load_MNIST_test_dataset
(
	double **test_data, // array of size 10000*784, each entry of which is in [0,1]
	int *test_label // 10000 labels, each of which is an integer among [0,9]
)
{
	printf("Begin to loading MNIST test dataset\n");
	int i,j;
	FILE *fp;
	fp = fopen("dataset/MNIST_test_dataset.txt", "r");
	if(fp==NULL)printf("fopen failed!\n");
	for (i = 0; i < 10000; i++)
	{
		//if(i%1000==0)printf("loading========================================\n");
		for (j = 0; j < 784; j++)
		{
			//printf("%d\n",j);
			fscanf(fp, "%lf", &test_data[i][j]);
			if(test_data[i][j]>1)printf("(%d,%d)\n",i,j);
		}
		fscanf(fp, "%d", &test_label[i]);
	}
	fclose(fp);
	printf("Dataset has been successfully loaded!\n");
	
	return 0;
}

int load_Cifar10_test_dataset
(
	double ****test_data, // array of size 10000*3*32*32, each entry of which is in [0,1]
	int *test_label // 10000 labels, each of which is an integer among [0,9]
)
{
	printf("Begin to load Cifar10 test dataset\n");
	int i,j,k,l;
	FILE *fp;
	fp = fopen("dataset/Cifar10_test_dataset.txt", "r");
	if(fp==NULL)printf("fopen failed!\n");
	for (i = 0; i < 10000; i++)
	{
		//if(i%1000==0)printf("loading========================================\n");
		for(j=0;j<3;j++)
		{
			for(k=0;k<32;k++)
			{
				for(l=0;l<32;l++)
				{
					fscanf(fp,"%lf",&test_data[i][j][k][l]);
					//if(test_data[i][j][k][l]>1)printf("(%d,%d,%d,%d)\n",i,j,k,l);
				}
			}
		}
		fscanf(fp, "%d", &test_label[i]);
	}
	fclose(fp);
	printf("Dataset has been successfully loaded!\n");
	
	return 0;
}
