#include <emmintrin.h>
#include <stdio.h>
#include <xmmintrin.h>


void swap(float *value_1, float *value_2) {
	float temp = *value_1;
	*value_1 = *value_2;
	*value_2 = temp;
}

void sum_mat_2x2(float* A, float* B, float* C) {
	__m128 a = _mm_load_ps(A);
	__m128 b = _mm_load_ps(B);	

	__m128 sum = _mm_add_ps(a, b);

	_mm_store_ps(C, sum);
}

void mul_mat_2x2(float* A, float* B, float* C) {

	__m128 a = _mm_load_ps(A);
	__m128 b = _mm_load_ps(B);

	swap(&b[1], &b[2]);

	__m128 a11 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(0, 0, 0, 0)); // [a11 a11 a11 a11]
	__m128 a12 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)); // [a12 a12 a12 a12]
	__m128 a21 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 2, 2)); // [a21 a21 a21 a21]
	__m128 a22 = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 3, 3, 3)); // [a22 a22 a22 a22]

	__m128 b11_b21 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 0, 2, 0)); // [b21 b11 b21 b11]
	__m128 b12_b22 = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 1, 3, 1)); // [b22 b12 b22 b12]

	__m128 c11_c12 = _mm_add_ps(_mm_mul_ps(a11, b11_b21), _mm_mul_ps(a12, b12_b22));
	__m128 c21_c22 = _mm_add_ps(_mm_mul_ps(a21, b11_b21), _mm_mul_ps(a22, b12_b22));

	__m128 c = _mm_shuffle_ps(c11_c12, c21_c22, _MM_SHUFFLE(1, 0, 1, 0));
	
	_mm_store_ps(C, c);
}

int main() {
	float* A = _mm_malloc(4 * sizeof(float), 16);

	A[0] = 1.0f;
	A[1] = 2.0f;
	A[2] = 3.0f;
	A[3] = 4.0f;

	float* B = _mm_malloc(4 * sizeof(float), 16);

	B[0] = 5.0f;
	B[1] = 6.0f;
	B[2] = 7.0f;
	B[3] = 8.0f;	

	float* C = _mm_malloc(4 * sizeof(float), 16);

	mul_mat_2x2(A, B, C);

	printf("%.1f %.1f %.1f %.1f\n", C[0], C[1], C[2], C[3]);	

	_mm_free(A);
	_mm_free(B);
	_mm_free(C);

	return 0;
}

