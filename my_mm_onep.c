/******************************************************************************
 * FILE: my_mm.c
 * DESCRIPTION:  
 *   MPI Matrix Multiply - C Version
 *   Code based on https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_mm.c
 * ORIGINAL AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
 *   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
 * LAST REVISED: 04/13/05
 * By Adam Pollack, 10/31/17, adapted for AMS 530 hw3 assignment
 ******************************************************************************/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int main (int argc, char *argv[])
{
  int	numtasks,              /* number of tasks in partition */
taskid,                /* a task identifier */
numworkers,            /* number of worker tasks */
source,                /* task id of message source */
dest,                  /* task id of message destination */
mtype,                 /* message type */
rows,                  /* rows of matrix A sent to each worker */
NRA, NCA, NCB,         /* number rows a, number columns a, number columns b so can be easily adapted for non square multiplication */
averow, extra, offset, /* used to determine rows sent to each worker */
i, j, k, l, m, rc;           /* misc */
int n[3] = {1400,2800,5600};
double starttime, endtime; /*Calculating runtime for each multiplication of NxN */

MPI_Status status;


MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);


for(l = 0; l < 3; l++){
  m = n[l]; /*square multiplication, can be adapted for non square multiplication easily */
  NRA = m;
  NCA = m;
  NCB = m;
  double	a[NRA][NCA],           /* matrix A to be multiplied */
  b[NCA][NCB],           /* matrix B to be multiplied */
  c[NRA][NCB];           /* result matrix C */
  
/**************************** master task ************************************/

if (taskid == MASTER)
{
  printf("my_mm has started with %d tasks.\n",numtasks);
  printf("Initializing arrays...\n");
  srand((unsigned)time(NULL));
  for (i=0; i<NRA; i++)
    for (j=0; j<NCA; j++){
      a[i][j]= ((float)rand()/(float)(RAND_MAX))*2 -1; /* Random float generation between -1 and 1 */
    }
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++){
      b[i][j]= ((float)rand()/(float)(RAND_MAX))*2 -1; /* Random float generation between -1 and 1 */
    }
      
      
    starttime = MPI_Wtime(); /*Start timing for multiplication and getting final matrix */
  
  /*Perform calculation on single processor with correct algorithm to check correctness */
  for(i = 0; i < m; i++){
    for(j = 0; j < m; j++){
      c[i][j] = 0;
      for(k = 0; k < m; k++){
        c[i][j] = c[i][j] + a[i][k]*b[k][j];
      }
    }
  }
  endtime = MPI_Wtime();
  printf("%d by %d matrix multiplication on %d processors took %6.2f seconds\n", m, m, numtasks, endtime-starttime);
}
}
printf("\n All done.");
MPI_Finalize();
}

