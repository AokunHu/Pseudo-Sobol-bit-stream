#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"quan.h"

double quan_i_d(double value, int integer, double decimal, int decimal_shift)
{
	double quan_value;
	
	if(value>(integer+decimal))
	{
		quan_value=integer+decimal;
		return quan_value;
	}
	else if(value<-(integer+decimal))
	{
		quan_value=-(integer+decimal);
		return quan_value;
	}
	
	int sign;
	sign=(value<0)?-1:1;
	quan_value = floor(fabs(value) * decimal_shift) / decimal_shift*sign;
	//printf("quan_value=%lf\n",quan_value);
	return quan_value;	
}

double quan_debug(int layer,int i, int j, double value, int integer, double decimal, int decimal_shift)
{
	double quan_value;
	
	if(value>(integer+decimal))
	{
		quan_value=integer+decimal;
		return quan_value;
	}
	else if(value<-(integer+decimal))
	{
		quan_value=-(integer+decimal);
		return quan_value;
	}
	
	int sign;
	sign=(value<0)?-1:1;
	quan_value = floor(fabs(value) * decimal_shift) / decimal_shift*sign;
	if(layer==0&&i==0)printf("%d: %lf,%d,%lf,%d,%lf\n",j,value,integer,decimal,decimal_shift,quan_value);
	return quan_value;	
}