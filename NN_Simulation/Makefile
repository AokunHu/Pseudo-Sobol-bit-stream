##all: inference_LeNet5 inference_LeNet5_Date19 inference_LeNet5_Date19_Sobol
all: inference_LeNet5_Date19 inference_LeNet5_Date19_Sobol inference_model_cifar10_quan inference_fusedSqueezeNetChen inference_fusedSqueezeNetChen_SC
##inference_LeNet5: inference_LeNet5.o load_dataset.o layers.o load_param.o model.o
##	gcc -fopenmp -o inference_LeNet5 inference_LeNet5.o load_dataset.o layers.o load_param.o model.o -L /usr/lib/x86_64-linux-gnu -lm
##inference_LeNet5.o: inference_LeNet5.c
##	gcc -fopenmp -c inference_LeNet5.c
inference_LeNet5_Date19: inference_LeNet5_Date19.o load_dataset.o layers.o load_param.o model.o quan.o
	gcc -fopenmp -o inference_LeNet5_Date19 inference_LeNet5_Date19.o load_dataset.o layers.o load_param.o model.o quan.o -L /usr/lib/x86_64-linux-gnu -lm
inference_LeNet5_Date19.o: inference_LeNet5_Date19.c
	gcc -fopenmp -c inference_LeNet5_Date19.c

# inference_model_cifar10_quan: inference_model_cifar10_quan.o load_dataset.o layers.o load_param.o model.o  quan.o 
# 	gcc -fopenmp -o inference_model_cifar10_quan inference_model_cifar10_quan.o load_dataset.o layers.o load_param.o model.o  quan.o -L /usr/lib/x86_64-linux-gnu -lm
# inference_model_cifar10_quan.o: inference_model_cifar10_quan.c
# 	gcc -fopenmp -c inference_model_cifar10_quan.c

inference_LeNet5_Date19_Sobol: inference_LeNet5_Date19_Sobol.o load_dataset.o layers.o load_param.o model.o quan.o
	gcc -fopenmp -o inference_LeNet5_Date19_Sobol inference_LeNet5_Date19_Sobol.o load_dataset.o layers.o load_param.o model.o quan.o -L /usr/lib/x86_64-linux-gnu -lm
inference_LeNet5_Date19_Sobol.o: inference_LeNet5_Date19_Sobol.c
	gcc -fopenmp -c inference_LeNet5_Date19_Sobol.c	

inference_fusedSqueezeNetChen: 	inference_fusedSqueezeNetChen.o load_dataset.o layers.o load_param.o model.o quan.o
	gcc -fopenmp -o inference_fusedSqueezeNetChen inference_fusedSqueezeNetChen.o load_dataset.o layers.o load_param.o model.o  quan.o -L /usr/lib/x86_64-linux-gnu -lm
inference_fusedSqueezeNetChen.o: 	inference_fusedSqueezeNetChen.c
	gcc -fopenmp -c inference_fusedSqueezeNetChen.c

inference_fusedSqueezeNetChen_SC: 	inference_fusedSqueezeNetChen_SC.o load_dataset.o layers.o load_param.o model.o quan.o
	gcc -fopenmp -o inference_fusedSqueezeNetChen_SC inference_fusedSqueezeNetChen_SC.o load_dataset.o layers.o load_param.o model.o  quan.o -L /usr/lib/x86_64-linux-gnu -lm
inference_fusedSqueezeNetChen_SC.o: 	inference_fusedSqueezeNetChen_SC.c
	gcc -fopenmp -c inference_fusedSqueezeNetChen_SC.c

quan.o: quan.c
	gcc -fopenmp -c quan.c
load_dataset.o: load_dataset.c
	gcc -fopenmp -c load_dataset.c
layers.o: layers.c
	gcc -fopenmp -c layers.c
load_param.o: load_param.c
	gcc -fopenmp -c load_param.c
model.o: model.c
	gcc -fopenmp -c model.c




