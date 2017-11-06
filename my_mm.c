/******************************************************************************
 * FILE: my_mm.c
 * DESCRIPTION:  
 *   MPI Matrix Multiply - C Version
 *   Code based on https://computing.llnl.gov/tutorials/mpi/samples/C/mpi_mm.c
 * ORIGINAL AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
 *   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
 * By Adam Pollack, 11/6/17, adapted for AMS 530 hw3 assignment
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
i, j, k, l, m;           /* misc */
int n[3] = {1400,2800,5600};
double starttime, endtime; /*Calculating runtime for each multiplication of NxN */
MPI_Status status;

MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

numworkers = numtasks-1;

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
    for (j=0; j<NCA; j++)
      a[i][j]= ((float)rand()/(float)(RAND_MAX))*2 -1; /* generate random number between 1 and -1 */
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++)
      b[i][j]= ((float)rand()/(float)(RAND_MAX))*2 -1; /* generate random number between 1 and -1 */
  
  printf("a[1][3]: %f, b[100][100]: %f as example entries.", a[1][3], b[100][100]);
  

  /* Send matrix data to the worker tasks */
  /* Broadcasting work each worker has to do */
  averow = NRA/numworkers;
  extra = NRA%numworkers;
  offset = 0;
  mtype = FROM_MASTER;
  for (dest=1; dest<=numworkers; dest++)
  {
    rows = (dest <= extra) ? averow+1 : averow;   	
    MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
    MPI_Send(&a[offset][0], rows*NCA, MPI_DOUBLE, dest, mtype,
             MPI_COMM_WORLD);
    MPI_Send(&b, NCA*NCB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
    offset = offset + rows;
  }
  
  /* Receive results from worker tasks */
  mtype = FROM_WORKER;
  for (i=1; i<=numworkers; i++)
  {
    source = i;
    MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
    MPI_Recv(&c[offset][0], rows*NCB, MPI_DOUBLE, source, mtype, 
             MPI_COMM_WORLD, &status);
  }
  
}


/**************************** worker task ************************************/
if (taskid > MASTER)
{
  mtype = FROM_MASTER;
  MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&a, rows*NCA, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
  MPI_Recv(&b, NCA*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
  
  starttime = MPI_Wtime(); /*Start timing for multiplication and getting final matrix */
  for (k=0; k<NCB; k++)
    for (i=0; i<rows; i++)
    {
      c[i][k] = 0.0;
      for (j=0; j<NCA; j++)
        c[i][k] = c[i][k] + a[i][j] * b[j][k];
    }
    endtime = MPI_Wtime();
  
  printf("%d by %d matrix multiplication on %d processors took %6.2f seconds\n", m, m, numtasks, endtime-starttime);
    mtype = FROM_WORKER;
  MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&rows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
  MPI_Send(&c, rows*NCB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
  
}

}

MPI_Finalize();
}

