#pragma once
#include <iostream>
#include <sstream>
#include <iterator>
#include <algorithm>
#include <math.h>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>
#include <fstream>
#include <string>
#include <bits/stdc++.h>
#include "kernels.cuh"

void calculate_one_scenario()
{
	dim3 dim_grid(2 * (N / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE);
	dim3 dim_block(BLOCK_SIZE);

	double *energy_pp = new double[M];
	double *diff_energy = new double[M];
	int *reference = new int[1];
	double *time_reference = new double[1];
	double *derivada = new double[1];

	double *g, *S, *first_term, *second_term, *third_term, *V_ph_k, *potential;
	double *energia_p, *sum_diff_g, *sum_diff_S, *old_S;
	int *checker;

	cudaMallocManaged(&g, N * sizeof(double));
	cudaMallocManaged(&S, N * sizeof(double));
	cudaMallocManaged(&first_term, N * sizeof(double));
	cudaMallocManaged(&second_term, N * sizeof(double));
	cudaMallocManaged(&third_term, N * sizeof(double));
	cudaMallocManaged(&V_ph_k, N * sizeof(double));
	cudaMallocManaged(&potential, N * sizeof(double));
	cudaMallocManaged(&energia_p, 1 * sizeof(double));
	cudaMallocManaged(&sum_diff_g, 1 * sizeof(double));
	cudaMallocManaged(&sum_diff_S, 1 * sizeof(double));
	cudaMallocManaged(&old_S, N * sizeof(double));
	cudaMallocManaged(&checker, 1 * sizeof(int));

	for (int i = 0; i < N; i++)
	{
		g[i] = 1.0;
		S[i] = 1.0;
	} // g(r) = S(k) = 1 is the initial condition

	compute_potential(potential);

	double energy, aux_energy, real_time;
	bool trigger = true;
	int k = 0;
	char loop_buffer[5000];
	bool correcao = false;
  	int plug = 1;
	//-------------------------------Cálculo de g(r) e S(k) --------------------------------------

	while (trigger)
	{
		// First term
		calculate_first_term<<<Blocks_N, ThreadsPerBlock_N>>>(g, first_term, potential);
		cudaDeviceSynchronize();
		// Second term
		calculate_second_term<<<Blocks_N, ThreadsPerBlock_N>>>(g, second_term);
		cudaDeviceSynchronize();
		// Third term
		calculate_third_term2<<<dim_grid, dim_block>>>(g, S, third_term, N, N);
		cudaDeviceSynchronize();
		// d_V_ph_k
		calculate_V_ph_k2<<<dim_grid, dim_block>>>(first_term, second_term, third_term, V_ph_k, N, N);
		cudaDeviceSynchronize();
		// old_S(k) = S(k)
		equal_old_S_to_S<<<Blocks_N, ThreadsPerBlock_N>>>(old_S, S);
		cudaDeviceSynchronize();
		// Update S(k) = new_S(k)
		update_S2<<<dim_grid, dim_block>>>(V_ph_k, S, checker);
		cudaDeviceSynchronize();

		//int N_menor;
		int N_pico;
		//double menor_S;
    double S_0;

		if (correcao)
		{
			for (int i = 0; i < N; i++)
			{
				if ((pow((double)i * dk, 2) + 4.0 * V_ph_k[i] < 0) && (pow((double)(i + 1) * dk, 2) + 4.0 * V_ph_k[i + 1] > 0))
				{
					N_pico = i + 1;
					S_0 = S[N_pico];
					break;
				}
			}


			for (int i = N_pico; i >= 0; i--)
			{
				S[i] = (S_0 / N_pico) * i;
			}
		
		}
		sum_diff_S[0] = 0.0;
		compute_diff_S2<<<dim_grid, dim_block>>>(S, old_S, sum_diff_S);
		cudaDeviceSynchronize();

		// Mixing the 'new g(r)' with the 'old g(r)' in a exponential
		sum_diff_g[0] = 0.0;
		g_from_S3<<<dim_grid, dim_block>>>(g, S, sum_diff_g);
		cudaDeviceSynchronize();

		real_time = ((double)k) * dt;
		// Compute the energy
		energy_CUDA_wrapper(g, S, potential, energia_p);

		energy = *energia_p;

		energy_pp[k] = energy;
		diff_energy[k] = std::abs(energy - aux_energy);
		derivada[0] = diff_energy[k] / dt;
		sprintf(loop_buffer, "\r\r\r Time = %.6f -----------------> e = %.6f +- %1.3e. Diff_g = %1.2e. Diff_S = %1.2e", real_time, energy, diff_energy[k], sum_diff_g[0], sum_diff_S[0]);
		std::cout << loop_buffer << std::flush;
		if (k * dt > 10 && (std::isnan(energy) == 1 || sum_diff_S[0] > 1.0))
		{
			if (sum_diff_S[0] > 1.0)
				{std::cout << "\nFRONTEIRA ATRAVESSADA POR sum_diff_S[0] > 1.0: sum_diff_S[0] = " << sum_diff_S[0] << std::endl;}
			if (std::isnan(energy) == 1)
				{std::cout << "\nFRONTEIRA ATRAVESSADA POR std::isnan(energy) == 1: energy = " << energy << std::endl;
				printf("/nOnde há instabilidade:/n");

				}

      //plug = 0;

			//cudaDeviceReset(); 
      exit(0);//break;
		}
		// STOP CRITERIUM
		bool criterium1 = (k * dt > 11 && sum_diff_S[0] < 1e-2);
		bool criterium2 = (k > 100 && sum_diff_g[0] < 1e-6 && sum_diff_S[0] < 1e-6);
		bool criterium3 = (k > 100 && derivada[0] < precision);
		bool criterium4 = (k > 100 && (derivada[0] < precision || (sum_diff_g[0] < 1e-6 && sum_diff_S[0] < 1e-6)));
    bool criterium5 = (k > 100 && sum_diff_g[0] < 1e-5 && sum_diff_S[0] < 1e-5);
        if (criterium5)
        {
            if (checker[0] == 0)
            {
			  printf("\nSTOP CRITERIUM FULFILLED. No testing.\n");
              //S_from_g2<<<dim_grid, dim_block>>>(g, S);
              //cudaDeviceSynchronize();
			  printf("Checking for instabilities: \n");
			 for(int i = 0; i < N; i++)
				{
					if(pow(double(i) * dk, 2) + 4.0 * V_ph_k[i] < 0)
					{
						printf("i = %d, double(i) * dk = %.6f, pow(double(i) * dk, 2) + 4.0 * d_V_ph_k[i] = %.20f \n", i, double(i) * dk, pow(double(i) * dk, 2) + 4.0 * V_ph_k[i]);
					}
				}
              trigger = 0;
			  break; // Break while loop
            }
			checker[0] = 1;
            printf("\nSTOP CRITERIUM FULFILLED. Testing. checker = %d, iteration = %d, N_pico = %.d\n", checker[0], k, N_pico);
            //trigger = 0;
            //break; // Break while loop          
            
        }
		aux_energy = energy;
		k++;
	}
	
	//-------------------------------Cálculo da fração BEC e OBDM ---------------------------------//
	
	double *f, *N_r, *N_wd_r, *N_wd_k, *g_wd, *S_wd, *new_N_wd_k, *old_g_wd, *old_S_wd, *N_ww_r;
	double *R_w, *R_d, *n_0p, *nq, *OBDM, *auto_energia;
	cudaMallocManaged(&f, N * sizeof(double));
	cudaMallocManaged(&N_r, N * sizeof(double));
	cudaMallocManaged(&g_wd, N * sizeof(double));
	cudaMallocManaged(&old_g_wd, N * sizeof(double));
	cudaMallocManaged(&S_wd, N * sizeof(double));
	cudaMallocManaged(&old_S_wd, N * sizeof(double));
	cudaMallocManaged(&N_wd_k, N * sizeof(double));
	cudaMallocManaged(&new_N_wd_k, N * sizeof(double));
	cudaMallocManaged(&N_wd_r, N * sizeof(double));
	cudaMallocManaged(&N_ww_r, N * sizeof(double));
	cudaMallocManaged(&N_wd_r, N * sizeof(double));
	cudaMallocManaged(&R_w, 1 * sizeof(double));
	cudaMallocManaged(&R_d, 1 * sizeof(double));
	cudaMallocManaged(&n_0p, 1 * sizeof(double));
	cudaMallocManaged(&nq, N * sizeof(double));
	cudaMallocManaged(&OBDM, N * sizeof(double));
	cudaMallocManaged(&auto_energia, sizeof(double));

	compute_N_r<<<dim_grid, dim_block>>>(S, N_r);
	cudaDeviceSynchronize();
	compute_f<<<dim_grid, dim_block>>>(N_r, g, f);
	cudaDeviceSynchronize();

	// Dados iniciais para os campos
	for (int i = 0; i < N; i++)
	{
		N_wd_k[i] = 0.0;
	}

	double *sum_diff_g_wd = new double[1];
	double *sum_diff_S_wd = new double[1];
	double *sum_diff_N_wd = new double[1];
	char buffer_n0[200];
	long long int contador = 0;
	double old_n = 0.0;
	double new_n = 0.0;
	double ratio = 1;
	
	if (plug == 1)
	{
		while (ratio > 1e-9)
		{
			compute_N_wd_r<<<dim_grid, dim_block>>>(N_wd_r, N_wd_k);
			cudaDeviceSynchronize();
			compute_g_wd<<<Blocks_N, ThreadsPerBlock_N>>>(g_wd, f, N_wd_r);
			cudaDeviceSynchronize();
			S_from_g2<<<dim_grid, dim_block>>>(g_wd, S_wd);
			cudaDeviceSynchronize();
			compute_N_wd_k<<<dim_grid, dim_block>>>(new_N_wd_k, S_wd, S);
			cudaDeviceSynchronize();
			update_field<<<dim_grid, dim_block>>>(N_wd_k, new_N_wd_k);
			cudaDeviceSynchronize();
			sum_diff_g_wd[0] = 0.0;
			sum_diff_S_wd[0] = 0.0;
			for (int i = 0; i < N; i++)
			{
				sum_diff_g_wd[0] += fabs(g_wd[i] - old_g_wd[i]);
				sum_diff_S_wd[0] += fabs(S_wd[i] - old_S_wd[i]);
			}
			for (int i = 0; i < N; i++)
			{
				old_g_wd[i] = g_wd[i];
				old_S_wd[i] = S_wd[i];
			}
			compute_Trap_wrapper(g_wd, N_wd_r, R_w);
			compute_Trap_wrapper(g, N_r, R_d);

			old_n = n_0p[0];
			n_0p[0] = exp(2.0 * (R_w[0]) - (R_d[0]));
			new_n = n_0p[0];
			sprintf(buffer_n0, "i = %Ld: n0 = %.8f --- diff_g_wd = %.6f ---- diff_S_wd = %.6f ---- ratio = %.10f", contador, n_0p[0], sum_diff_g_wd[0], sum_diff_S_wd[0], ratio);
			std::cout << "\r" << buffer_n0 << std::flush;

			contador += 1;
			R_w[0] = 0.0;
			R_d[0] = 0.0;
			if (contador > 1000)
			{
				ratio = fabs(new_n - old_n) / new_n;
			}
		}
	}
	compute_N_ww_r<<<dim_grid, dim_block>>>(N_wd_k, S_wd, N_ww_r); // N_ww_r which will be used bellow
	cudaDeviceSynchronize();
	compute_OBDM<<<dim_grid, dim_block>>>(OBDM, N_ww_r, n_0p); // OBDM
	cudaDeviceSynchronize();
	compute_nq<<<dim_grid, dim_block>>>(nq, OBDM, n_0p); // DENSITY OF EXCITED STATES n(q)
	cudaDeviceSynchronize();

	double *entropia;
	cudaMallocManaged(&entropia, sizeof(double));
	entropia_CUDA_wrapper(g, N, entropia); // Compute the entropy
	double *cumulante = new double[N];	   // Accumulated entropy
	for (int i = 0; i < N; i++)
	{
		entropia_CUDA_wrapper(g, i, entropia);
		cumulante[i] = fabs(*entropia);
	}
	double integral = 0.0;
	for (int i = 0; i < N; i++)
	{
		integral += i * potential[i] * dr * dr;
	}
	integral *= M_PI * rho;
	auto_energia[0] = 1.0 - integral / energy_pp[k]; // Compute the self-energy

	//-------------------------------------PRINTING THE RESULTS---------------------------------------------------//

	char buffer[1000];
	char g_buffer[1000];
	char S_buffer[1000];
	char file_buffer[1000];
	char file_buffer_n0[1000];
	char file_buffer_entropia_rho[1000];
	char file_buffer_entropia_U[1000];
	char file_buffer_Roton[1000];
	char BF_buffer[1000];
	char file_buffer_cumulante[1000];
	char file_buffer_omega[1000];
	char file_buffer_OBDM[1000];
	char file_buffer_nq[1000];
	char file_buffer_g0[1000];
	char file_buffer_W[1000];

	sprintf(g_buffer, "Dados-HNC-Lifschitz_LPG8_2/g-functions/g-rho-%3.8f-U-%1.20f.dat", rho, U);
	sprintf(S_buffer, "Dados-HNC-Lifschitz_LPG8_2/S-functions/S-rho-%3.8f-U-%1.20f.dat", rho, U);
	sprintf(file_buffer, "Dados-HNC-Lifschitz_LPG8_2/tabelas_de_auto_energia/auto-energia-sigma-%1.4f-U-%1.20f.dat", sigma, U);
	sprintf(BF_buffer, "Dados-HNC-Lifschitz_LPG8_2/BF-spectra/BF-spectra-rho-%3.8f-U-%1.20f.dat", rho, U);
	sprintf(file_buffer_Roton, "Dados-HNC-Lifschitz_LPG8_2/Rotons/Rotons.dat");
	sprintf(file_buffer_n0, "Dados-HNC-Lifschitz_LPG8_2/tabelas_de_n0/n0-rho-%3.8f-sigma-%1.4f.dat", rho, sigma);
	sprintf(file_buffer_entropia_rho, "Dados-HNC-Lifschitz_LPG8_2/tabelas_de_entropia/entropia-rho-%3.8f-sigma-%1.4f.dat", rho, sigma);
	sprintf(file_buffer_entropia_U, "Dados-HNC-Lifschitz_LPG8_2/tabelas_de_entropia/entropia-U-%1.20f-sigma-%1.4f.dat", U, sigma);
	sprintf(file_buffer_cumulante, "Dados-HNC-Lifschitz_LPG8_2/cumulantes/c-U-%1.20f-rho-%3.8f-sigma-%1.4f.dat", U, rho, sigma);
	sprintf(file_buffer_omega, "Dados-HNC-Lifschitz_LPG8_2/Weff-functions/Weff-rho-%3.8f-U-%1.20f.dat", rho, U);
	sprintf(file_buffer_OBDM, "Dados-HNC-Lifschitz_LPG8_2/OBDM-functions/OBDM-rho-%3.8f-U-%1.20f.dat", rho, U);
	sprintf(file_buffer_nq, "Dados-HNC-Lifschitz_LPG8_2/nq-functions/nq-rho-%3.8f-U-%1.20f.dat", rho, U);
	sprintf(file_buffer_g0, "Dados-HNC-Lifschitz_LPG8_2/g0-functions/g0-rho-%3.8f.dat", rho);
	sprintf(file_buffer_W, "Dados-HNC-Lifschitz_LPG8_2/W-functions/W-U-%1.20f-rho-%3.8f-sigma-%1.4f.dat", U, rho, sigma);

	std::ofstream myfile_s1; // PRINT ENTROPY BY rho
	myfile_s1.open(file_buffer_entropia_rho, std::ios::app);
	myfile_s1 << U << "," << *entropia << "\n";
	myfile_s1.close();

	std::ofstream myfile_s2; // PRINT ENTROPY BY U
	myfile_s2.open(file_buffer_entropia_U, std::ios::app);
	myfile_s2 << TildeRc << "," << *entropia << "\n";
	myfile_s2.close();

	std::ofstream myfile_n0; // PRINT THE BEC FRACTION n0
	myfile_n0.open(file_buffer_n0, std::ios::app);
	myfile_n0 << U << "," << n_0p[0] << "\n";
	myfile_n0.close();

	std::ofstream myfile_cumulante; // PRINT THE ACCUMULATED ENTROPY
	myfile_cumulante.open(file_buffer_cumulante, std::ios::out);
	for (int i = 0; i < N; i++)
	{
		myfile_cumulante << cumulante[i] << std::endl;
	}
	myfile_cumulante.close();

	std::ofstream myfile_g;
	myfile_g.open(g_buffer, std::ios::out);
	for (int i = 0; i < N; i++)
	{
		myfile_g << (double)i * dr << "," << g[i] << std::endl;
	}
	myfile_g.close();

	std::ofstream myfile_S;
	myfile_S.open(S_buffer, std::ios::out);
	for (int i = 0; i < N; i++)
	{
		myfile_S << (double)i * dk << "," << S[i] << std::endl;
	}
	myfile_S.close();

	std::ofstream myfile_BF; // PRINT BIJL-FEYNMAN ESPECTRUM
	myfile_BF.open(BF_buffer, std::ios::out);
	for (int i = 0; i < N; i++)
	{
		myfile_BF << (double)i * dk << "," << pow(dk * (double)i, 2) / (S[i]) << std::endl;
	}
	myfile_BF.close();

	std::ofstream myfile_omega; // PRINT W_eff(r)
	myfile_omega.open(file_buffer_omega, std::ios::out);
	for (int i = 0; i < N; i++)
	{
		myfile_omega << (double)i * dr << "," << potential[i] + (third_term[i] / (g[i] - 1.0)) << std::endl;
	}
	myfile_omega.close();

	std::ofstream myfile_OBDM; // PRINT OBDM(K)
	myfile_OBDM.open(file_buffer_OBDM, std::ios::out);
	for (int i = 0; i < N; i++)
	{
		myfile_OBDM << (double)i * dr << "," << OBDM[i] << std::endl;
	}
	myfile_OBDM.close();

	std::ofstream myfile_nq; // PRINT n(k)
	myfile_nq.open(file_buffer_nq, std::ios::out);
	for (int i = 0; i < N; i++)
	{
		myfile_nq << (double)i * dk << "," << nq[i] << std::endl;
	}
	myfile_nq.close();

	std::ofstream myfile_energia;
	myfile_energia.open(file_buffer, std::ios::app);
	myfile_energia << TildeRc << "," << *auto_energia << "\n";
	myfile_energia.close();

	// Print the results
	sprintf(buffer, "U = %3.16f, rho = %3.8f. e = %.2f +- %.1e. \u03B5 = %.3f. n0 = %1.8f. s = %3.5f",
			U, rho, energy_pp[k], diff_energy[k],
			*auto_energia, *n_0p, *entropia);

	std::cout << "\r\r\r\r\r\r(" << aux_total << "/" << total << ") --- " << buffer << " # = " << k << " iterations."
			  << " #i_n0 = " << contador << " iterations."
			  << std::endl;

	//-----------------------------------------------DONE--------------------------------------------------//
	cudaDeviceReset();
	delete[] cumulante;
	delete[] sum_diff_g_wd;
	delete[] sum_diff_S_wd;
	delete[] sum_diff_N_wd;
	delete[] energy_pp;
	delete[] diff_energy;
	delete[] reference;
	delete[] time_reference;
}

