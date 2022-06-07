__kernel void printText () {  
int N = get_group_id (0);               
int M = get_local_id (0);                 
int K = get_global_id (0);                
printf("I am from %d block, %d thread (global index: %d)", N, M, K);	
}
