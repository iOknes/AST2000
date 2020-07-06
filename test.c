#include <stdio.h>
#include <math.h>

#define DIM 3

double vec_squared(double vec[], size_t size){
    double ret;
    for (int i = 0; i < size; i++){
        ret += pow(vec[i], 2);
    }
    return ret;
}

double * acc(double r[DIM]){
    static double a[DIM];
    for (int i = 0; i < DIM; i++) {
        a[i] = 2 * (1 - pow(r[0], 2) - pow(r[1], 2)) * r[i];
    }
    return a;
}

int main(){
    double dt = 0.001, T = 15;
    int N = T / dt;
    int pN = 2;

    double r0[DIM] = {1.5, -1, -1.5};
    double v0[DIM] = {0, 0.1, 0};

    double r[N][pN][DIM];

    double *a;
    a = acc(r0);

    printf("(%f, %f, %f)\n", a[0], a[1], a[2]);
}
