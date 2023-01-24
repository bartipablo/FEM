#include <iostream>
#include <cmath>
#include <functional>
#include "Eigen/Dense"
#include <vector>
#include <string>
//#include "matplotlibcpp.h"

#define N 10
#define BETA  1.0
#define GAMMA  1.0
#define U1 exp(-1)

using namespace std;
using namespace Eigen;
//namespace plt = matplotlibcpp;

function<double(double)> multiplyTwoFunctions(function<double(double)> f1, function<double(double)> f2) {
    return [f1, f2](double x) { return f1(x) * f2(x); };
}

function<double(double)> multiplyThreeFunctions(function<double(double)> f1, function<double(double)> f2, function<double(double)> f3) {
    return [f1, f2, f3](double x) { return f1(x) * f2(x) * f3(x); };
}

double function_a(double x) {
    return 0.0;
}

double function_b(double x) {
    return 1.0;
}

double function_c(double x) {
    return 2.0;
}

double function_f(double x) {
    return exp(-x);
}

function<double(double)> e(int i, int n) {
    return [i, n](double x) {
        if (1.0 - abs(n * (x - i / n)) >= 0) {
            return 1.0 - abs(n * (x - i / n));
        } else {
            return 0.0;
        }
    };
}

function<double(double)> e_der(int i, int n) {
    return [i, n](double x) {
        if (((i - 1) / n <= x) && (x < i / n)) {
            return (double) n;
        } else if (((i + 1) / n > x) && (x >= i / n)){
            return (double) -n;
        } else {
            return 0.0;
        }
    };
}


//Gauss–Legendre quadrature integration, two points
double integration(function<double(double)> f, double a, double b) {
    double x1 = 1 / sqrt(3);
    double x2 = - 1 / sqrt(3);
    double w1 = 1, w2 = 1;
    return (b-a)/2 *( w1*f((b-a)/2*x1 + (a+b)/2) + w2*f((b-a)/2*x2 + (a+b)/2) );
}


double B(function<double(double)> u_der, function<double(double)> v_der, function<double(double)> u, function<double(double)> v, double start, double end) {
    function<double(double)> f1 = multiplyThreeFunctions(function_a, u_der, v_der);
    function<double(double)> f2 = multiplyThreeFunctions(function_b, u_der, v);
    function<double(double)> f3 = multiplyThreeFunctions(function_c, u, v);
    return -u(0)*v(0)*BETA - integration(f1, start, end) + integration(f2, start, end) + integration(f3, start, end);
}


double L(function<double(double)> v, double start, double end) {
    function<double(double)> f = multiplyTwoFunctions(v, function_f);
    return integration(f, start, end) - GAMMA*v(0);
}


Eigen::Matrix<float, N, N> initialize_matrix() {
    Matrix<float, N, N> matrix;
    matrix.setZero();
    double start, end;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (abs(i - j) > 0) {
                matrix(i, j) = 0.0;
                continue;
            }

            if (abs(i - j) == 1) {
                start = max(0.0, (double) min(i, j)/N);
                end = min(1.0, (double) max(i, j)/N);
            } else {
                start = max(0.0, (double) (i - 1)/N);
                end = min(1.0, (double) (i + 1)/N);
            }
            matrix(i, j) = B(e_der(j, N), e_der(i, N), e(j, N), e(i, N), start, end);
        }
    }
    return matrix;
}


Eigen::Matrix<float, N, 1> initialize_vector() {
    Matrix<float, N, 1> vector;
    for (int i = 0; i < N; i++) {
        double start = max(0.0, (double) i/N - 1.0/N);
        double end = min(1.0, (double) i/N + 1.0/N);
        vector(i, 0) = L(e(i, N), start, end);
    }
    return vector;
}

Eigen::Matrix<float, N, 1> solve_linear_matrix_equation(Eigen::Matrix<float, N, N> matrix, Eigen::Matrix<float, N, 1> vector) {
    Eigen::ColPivHouseholderQR<Eigen::Matrix<float, N, N>> dec(matrix);
    Eigen::Matrix<float, N, 1> solution = dec.solve(vector);
    return solution;
}

vector<double> initialize_x_values() {
    vector<double> result;
    for (int i = 0; i <= 1000; i++) {
        result.push_back((double) i / 1000);
    }
    return result;
}

vector<double> initialize_y_values(Eigen::Matrix<float, N, 1> matrix) {
    vector<double> result;
    for (int x = 0; x <= 1000; x++) {
        double sum = 0;
        for (int i = 0; i < N; i++) {
            sum += matrix(i, 0) * e(i, N) ((double) x / 1000);
        }
        sum += x * U1 / 1000;
        result.push_back(sum);
    }
    return result;
}

void print_calculated_value(vector<double> x, vector<double> y) {
    cout << "Equation solved. Calculated values (x, y): " << endl;
    for (int i = 0; i <= 1000; i++) {
        cout << "(" + to_string(x[i]) + ", " + to_string(y[i]) + ") ";
    }
}

void draw_plot(vector<double> x, vector<double> y) {

}


int main() {
    Eigen::Matrix<float, N, N> matrix = initialize_matrix();
    Eigen::Matrix<float, N, 1> vectorLELE = initialize_vector();
    Eigen::Matrix<float, N, 1> solution = solve_linear_matrix_equation(matrix, vectorLELE);

    vector<double> x_values = initialize_x_values();
    vector<double> y_values = initialize_y_values(solution);

    print_calculated_value(x_values, y_values);
    draw_plot(x_values, y_values);

    return 0;
}
