#include <hls_stream.h>
#include <hw/svd.hpp>

#define K 500
#define N 500
#define M 10

const unsigned int c_M = M;

static void load_input_A(float* A, hls::stream<float>& A_stream) {
	load_input_A_loop: for (int i = 0; i < K*N; i++)
		A_stream << A[i];
}

static void load_input_y(float* y, hls::stream<float>& y_stream) {
	load_input_y_loop: for (int i = 0; i < K; i++)
		y_stream << y[i];
}

static void store_result_x(float* x, hls::stream<float>& x_stream) {
	store_result_x_loop: for (int i = 0; i < N + 5*N*N; i++)
		x[i] = x_stream.read();
}

static void traspuesta_KporN(float A[K][N], float R[N][K]) {
	traspuesta_KporN_outer_loop: for(int i = 0; i < N; i++)
		traspuesta_KpoN_inner_loop: for(int j = 0; j < K; j++)
			R[i][j] = A[j][i];
}

static void traspuesta_NporN(float U[N][N], float R[N][N]) {
	traspuesta_NporN_outer_loop: for(int i = 0; i < N; i++)
		traspuesta_NporN_inner_loop: for(int j = 0; j < N; j++)
			R[i][j] = U[j][i];
}

static void producto_vector_NporK(float A[N][K], float y[K], float R[N]) {
	producto_vector_NporK_outer_loop: for(int i = 0; i < N; i++) {
		float Ri = 0;
		producto_vector_NporK_inner_loop: for(int j = 0; j < K; j++) {
			Ri += A[i][j] * y[j];
		}
		R[i] = Ri;
	}
}

static void producto_NporKporN(float A[N][K], float B[K][N], float R[N][N]) {
	// #pragma HLS array_reshape variable=A complete dim=2
	// #pragma HLS array_reshape variable=B complete dim=1
	producto_NporKporN_outer_loop: for (int i = 0; i < N; i++) {
		producto_NporKporN_inner_loop: for (int j = 0; j < N; j++) {
			// #pragma HLS pipeline II=1
			float Rij = 0;
			producto_NporKporN_innermost_loop: for (int k = 0; k < K; k++)
				Rij += A[i][k] * B[k][j];
			R[i][j] = Rij;
		}
	}
}

static void producto_NporNporK(float A[N][N], float B[N][K], float R[N][K]) {
	// #pragma HLS array_reshape variable=A complete dim=2
	// #pragma HLS array_reshape variable=B complete dim=1
	producto_NporNporK_outer_loop: for (int i = 0; i < N; i++) {
		producto_NporNporK_inner_loop: for (int j = 0; j < K; j++) {
			// #pragma HLS pipeline II=1
			float Rij = 0;
			producto_NporNporK_innermost_loop: for (int k = 0; k < N; k++)
				Rij += A[i][k] * B[k][j];
			R[i][j] = Rij;
		}
	}
}

static void producto_NporNporN(float A[N][N], float B[N][N], float R[N][N]) {
	#pragma HLS array_reshape variable=A complete dim=2
	#pragma HLS array_reshape variable=B complete dim=1
	producto_NporNporN_outer_loop: for (int i = 0; i < N; i++) {
		producto_NporNporN_inner_loop: for (int j = 0; j < N; j++) {
			#pragma HLS pipeline II=1
			float Rij = 0;
			producto_NporNporN_innermost_loop: for (int k = 0; k < N; k++)
				Rij += A[i][k] * B[k][j];
			R[i][j] = Rij;
		}
	}
}

static void reescalado(float A[K][N], float y[K]) {
	float norm_f = 0;
	norma_frobenius_outer_loop: for(int i = 0; i < K; i++) {
		norma_frobenius_inner_loop: for(int j = 0; j < N; j++) {
			float Aij = A[i][j];
			norm_f += Aij * Aij;
		}
	}
	norm_f = hls::sqrt(norm_f);
	norm_f = norm_f * (25 + N) / N;
	reajuste_A_outer_loop: for(int i = 0; i < K; i++)
		reajuste_A_inner_loop: for(int j = 0; j < N; j++)
			A[i][j] = A[i][j] / norm_f;
	reajuste_y_loop: for(int i = 0; i < K; i++)
		y[i] = y[i] / norm_f;
}

static void calcular_IB1(float U[N][N], float Ut[N][N], float S_0[N][N], float mu, float C[N], float IB[N][N], float IB1[N][N]) {
	// #pragma HLS expression_balance
	float S[N][N] = {{}}, SUt[N][N] = {{}}, Cspc[N][N] = {{}};
	float suma_por_filas[N] = {}, suma_por_columnas[N] = {};
	float suma_total_IB = 0;
	calculo_S_loop: for(int i = 0; i < N; i++)
		S[i][i] = 1 / (S_0[i][i] + mu);
	producto_NporNporN(S, Ut, SUt);
	producto_NporNporN(U, SUt, IB);
	sumas_outer_loop: for(int i = 0; i < N; i++) {
		sumas_inner_loop: for(int j = 0; j < N; j++) {
			suma_por_filas[i] += IB[i][j];
			suma_por_columnas[i] += IB[j][i];
		}
	}
	suma_total_loop: for(int i = 0; i < N; i++)
		suma_total_IB += suma_por_filas[i];
	vector_C_loop: for(int i = 0; i < N; i++)
		C[i] = (1/suma_total_IB) * suma_por_filas[i];
	Cspc_outer_loop: for(int i = 0; i < N; i++)
		Cspc_inner_loop: for(int j = 0; j < N; j++)
			Cspc[i][j] = C[i] * suma_por_columnas[j];
	IB1_outer_loop: for(int i = 0; i < N; i++)
		IB1_inner_loop: for(int j = 0; j < N; j++)
			IB1[i][j] = IB[i][j] - Cspc[i][j];
}

static void compute_SUNSAL(hls::stream<float>& A_stream, hls::stream<float>& y_stream, hls::stream<float>& x_stream) {

	float A[K][N];
	float y[K];
	float x_sol[N];

	// LECTURA

	lectura_A_outer_loop: for(int i = 0; i < K; i++)
		lectura_A_inner_loop: for(int j = 0; j < N; j++)
			A[i][j] = A_stream.read();

	lectura_y_loop: for(int i = 0; i < K; i++)
		y[i] = y_stream.read();

	// REESCALADO

	reescalado(A,y);

	// CALCULO

			float mu = 0.01;
			bool mu_changed = false;
			float const tol = hls::sqrt(N) * 0.0001;
			float res_p = 1000;
			float res_d = 1000;

			// INICIALIZACION

			float At[N][K] = {{}}, IBAt[N][K] = {{}}, AtA[N][N] = {{}};

			traspuesta_KporN(A, At);

			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					x_stream << A[i][j];

			float At_aux[N*K], A_aux[K*N], AtA_aux[N*N];

			producto_NporKporN(At, A, AtA);

			float U[N][N], S_0[N][N], Utr[N][N], IB[N][N], IB1[N][N];
			float C[N] = {}, Aty[N] = {};

			producto_vector_NporK(At, y, Aty);


			float v[N][N] = {{}};
			xf::solver::svdTop<N, N, xf::solver::svdTraits<N, N, float, float>,float, float>(AtA, S_0, U, v);

			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					x_stream << U[i][j];

			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					x_stream << S_0[i][j];

			traspuesta_NporN(U,Utr);

			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					x_stream << Utr[i][j];

			calcular_IB1(U, Utr, S_0, mu, C, IB, IB1);

			for(int i = 0; i < N; i++)
				for(int j = 0; j < N; j++)
					x_stream << IB[i][j];

			// x

			producto_NporNporK(IB, At, IBAt);
			producto_vector_NporK(IBAt, y, x_sol);

			// u, d

			float u[N], d[N] = {}, W[N] = {}, u_0[N] = {};
			inicializar_x_loop: for(int i = 0; i < N; i++)
				u[i] = x_sol[i];

			// ITERACIONES

			int iters = 1;

			main_loop: for (int iters = 1; iters < M; iters++) {
				#pragma HLS loop_tripcount min = c_M max = c_M
				if (!((res_d > tol) || (res_d < 0 - tol) || (res_p > tol) || (res_p < 0 - tol)))
					break;

				if ((iters % 10) == 1) {
				   residuo_u_loop: for(int i = 0; i < N; i++)
					   u_0[i] = u[i];
				}

				// u = max(x - d, 0)

				actualizar_u_loop: for (int i = 0; i < N; i++) {
					float r = x_sol[i] - d[i];
					if (r > 0)
						u[i] = r;
					else
						u[i] = 0;
				}

				// x

				w_loop: for(int i = 0; i < N; i++)
					W[i] = Aty[i] + mu * (u[i] + d[i]);

				// !!!!!!!!!!

				actualizar_x_outer_loop: for(int i = 0; i < N; i++) {
					x_sol[i] = C[i];
					actualizar_x_inner_loop: for(int j = 0; j < N; j++)
						x_sol[i] += IB1[i][j] * W[j];
				}

				// d = d - (x - u)

				actualizar_d_loop: for (int i = 0; i < N; i++)
					d[i] = d[i] - (x_sol[i] - u[i]);

				// control de los residuos (res_p y res_d)

				if ((iters % 10) == 1) {

					// residuos: {
						// #pragma HLS loop_merge

						// residuo primal (x - z)

						// !!!!!!!!!!

						res_p = 0;
						res_p_loop: for(int i = 0; i < N; i++) {
							res_p += (x_sol[i] - u[i]) * (x_sol[i] - u[i]);
						}
						res_p = hls::sqrt(res_p);

						// !!!!!!!!!!

						// residuo dual (u - u_0)

						res_d = 0;
						res_d_loop: for(int i = 0; i < N; i++)
							res_d += (u[i] - u_0[i]) * (u[i] - u_0[i]);
						res_d = mu * hls::sqrt(res_d);

					// }

					// actualizar mu

					if (res_p > 10 * res_d) {
						mu = mu * 2;
						d_res_p_loop: for(int i = 0; i < N; i++)
							d[i] = d[i] / 2;
						mu_changed = true;
					}
					else if (res_d > 10 * res_p) {
						mu = mu / 2;
						d_res_d_loop: for(int i = 0; i < N; i++)
							d[i] = d[i] * 2;
						mu_changed = true;
					}

					if (mu_changed) {
						calcular_IB1(U, Utr, S_0, mu, C, IB, IB1);
						mu_changed = false;
					}

				}

			}

	// ESCRITURA

	escritura_x_loop: for(int i = 0; i < N; i++)
		x_stream << x_sol[i];

}

extern "C" {
	void krnl_SUNSAL(float* A, float* y, float* x) {
		#pragma HLS INTERFACE m_axi port = A bundle = gmem0
		#pragma HLS INTERFACE m_axi port = y bundle = gmem1
		#pragma HLS INTERFACE m_axi port = x bundle = gmem2

		static hls::stream<float> A_stream("input_A");
		static hls::stream<float> y_stream("input_y");
		static hls::stream<float> x_stream("output_x");

		#pragma HLS dataflow
	    load_input_A(A, A_stream);
	    load_input_y(y, y_stream);
	    compute_SUNSAL(A_stream, y_stream, x_stream);
	    store_result_x(x, x_stream);
	}
}
