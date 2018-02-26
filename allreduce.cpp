/***************************************************************************
 *   Name Andrea Garritano
 ***************************************************************************/

/***************************************************************************
 *   Copyright (C) 2012, 2015 Jan Fostier (jan.fostier@intec.ugent.be)     *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include "mpi.h"
#include <cstdlib>
#include <iostream>
#include <math.h>
#include <sstream>

using namespace std;

/**
 * Wrapper function around MPI_Allreduce (leave this unchanged)
 * @param sendbuf Send buffer containing count doubles (input)
 * @param recvbuf Pre-allocated receive buffer (output)
 * @param count Number of elements in the send and receive buffers
 */
void allreduce(double *sendbuf, double *recvbuf, int count)
{
	MPI_Allreduce(sendbuf, recvbuf, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

/**
 * Wrapper function around MPI_Allreduce (implement reduce-scatter / allgather algorithm)
 * @param sendbuf Send buffer containing count doubles (input)
 * @param recvbuf Pre-allocated receive buffer (output)
 * @param count Number of elements in the send and receive buffers
 */
void allreduceRSAG(double *sendbuf, double *recvbuf, int count)
{
	// Note: assume MPI_DOUBLE, MPI_SUM and MPI_COMM_WORLD parameters
	// These may be hard-coded in the implementation. You may assume
	// that the number of processes is a power of two and you may assume
	// that "count" is a multiple of P.

	// Implement the allreduce algorithm using only point-to-point
	// communications, using the reduce-scatter / allgather algorithm
	// (see exercises slides for communication scheme)

	// Do not change the function signature as your implementation
	// will be automatically validated using a test framework.

	// Do not overwrite the sendbuf contents. Allocate a temp buffer
	// if necessary.

	int thisProc, nProc;
    MPI_Comm_rank( MPI_COMM_WORLD, &thisProc );
    MPI_Comm_size( MPI_COMM_WORLD, &nProc );
    MPI_Status status;

    int dest;
    int width;
    double *myData = new double[count];
    for (int i=0; i<count; i++)
        myData[i]=sendbuf[i];

    double *temp_sendbuf = new double[count];
    for (int i=0; i<count; i++)
        temp_sendbuf[i]=sendbuf[i];

    int nPhases = log2(nProc);

    for (int p = 0; p<nPhases; p++){
        width = (count/pow(2, p+1));

        if (thisProc%((int)pow(2, p+1))<(int)pow(2, p+1)/2){
            dest = thisProc + pow(2,p);
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i+width]; //second half of the message
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &status);

            for (size_t i = 0; i < width; i++)
                myData[i] += recvbuf[i]; //sum the replay
        }   
        else{
            dest = thisProc - pow(2,p);

            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &status);

            
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i]; //first half of the message

            for (size_t i = 0; i < width; i++)
                myData[i+width] += recvbuf[i]; //sum the replay
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

            //Copy the second half on the first half
            for (size_t i = 0; i < width; i++)
                myData[i]=myData[i+width];
        }
    }
    for (int p = nPhases-1; p>=0; p--){
        width = (count/pow(2, p+1));
        if (thisProc%((int)pow(2, p+1))<(int)pow(2, p+1)/2){
            dest = thisProc + pow(2,p);
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i];
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD, &status);

            for (size_t i = 0; i < width; i++)
                myData[i+width] = recvbuf[i];
        }   
        else{
            dest = thisProc - pow(2,p);

            MPI_Recv(recvbuf, width, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD, &status);
            
            for (int i=0; i<width; i++)
                temp_sendbuf[i]=myData[i];

            
            for (size_t i = 0; i < width; i++)
                myData[i+width] = myData[i];

            for (size_t i = 0; i < width; i++)
                myData[i] = recvbuf[i];
            MPI_Send(temp_sendbuf, width, MPI_DOUBLE, dest, 1, MPI_COMM_WORLD);

        }
    }

    for (size_t i = 0; i < count; i++)
        recvbuf[i] = myData[i];

    delete [] myData;
    delete [] temp_sendbuf;

}

/**
 * Program entry
 */
int main(int argc, char* argv[])
{
	int thisProc, nProc;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &thisProc);
	MPI_Comm_size(MPI_COMM_WORLD, &nProc);

	// initialize sendbuf
	double *sendbuf = new double[nProc];

	for (size_t i = 0; i < nProc; i++)
        sendbuf[i] = (i+1)*(thisProc+1);

    // initialize recvbuf
    double *recvbuf = new double[nProc];
   
	allreduceRSAG(sendbuf, recvbuf, nProc);

    /*
    double *recvbufGround = new double[nProc];
    allreduce(sendbuf, recvbufGround, nProc);
    allreduce(sendbuf, recvbuf, nProc);

    bool test = true;
    for (int i=0; i<nProc; i++){
        if (recvbuf[i]!=recvbufGround[i])
            test=false;
    }
    if (test)
        cout<<"Test passed"<<endl;
    else
        cout<<"Test failed"<<endl;
    */

	// optionally: write test code
	// (this is not required as we will only test the 
	// allreduceRSAG implementation itself.

	MPI_Finalize();
	exit(EXIT_SUCCESS);
}
