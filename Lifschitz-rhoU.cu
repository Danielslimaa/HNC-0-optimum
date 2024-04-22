#include <iostream>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <omp.h>
#include <stdio.h>
#include <cstdlib>
#include "main-loop.cuh"

int main(int argc, char* argv[]){

dt = (double)std::stod(argv[1]);
rho = (double)std::stod(argv[2]);
U = (double)std::stod(argv[3]);
double dU = (double)std::stod(argv[4]);

int p = 2;

  N = (1 << (11 + p));
  if (N >= 1024)
  {
    ThreadsPerBlock_N = 1024;
  }
  else
  {
    ThreadsPerBlock_N = N;
  }
  Blocks_N = (int)ceil((double)N / 1024.0);
  // For the LPGE8_1:
  //dr = 30.0 / (double)N;
  //dk = 10.0 / (double)N;
  // For the LPGE8_2:
  dr = 28.0 / (double)N;
  dk = 10.0 / (double)N;
  time_t start_total, end_total;
  
  double initial_precision = 2.00e-5;
  precision = initial_precision;
  //0.0001,0.0005,0.005,0.010,0.0125,
  //double valores[] = {0.05,0.06,0.07,0.08,0.09};
  //int tamanho = (int)(sizeof(valores) / sizeof(valores[0]));
  int coluna_U;
  int max_linha_U = 1;
  int max_coluna_U = 1;//tamanho;
  total = max_linha_U * max_coluna_U;
  aux_total = 1;

  //printf("V.0.2.4 - State:\n U = %f, N = %ld, dr = %f, dk = %f, dt = %f, rho = %1.3f, and precision = %1.0e. \n", U, N, dr, dk, dt, rho, precision);
  time(&start_total);
  for (coluna_U = 0; coluna_U < max_coluna_U; coluna_U++)
  { 
    //U = valores[coluna_U];    
    printf("State:\n U = %.12f, rho = %1.4f\n", U, rho);
    calculate_one_scenario();
    aux_total += 1;
    U = U + dU;
  }

  std::cout << "\nTotal computational time-lapse: " << time(&end_total) - start_total << " s." << std::endl;
  return 0;
}