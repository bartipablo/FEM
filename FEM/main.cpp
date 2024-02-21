#include <iostream>
#include <cmath>
#include <functional>
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <cstring>
#include "matplotlibcpp.h"

using namespace std;
using namespace Eigen;
namespace plt = matplotlibcpp;


function<double(double)> multiply_two_functions(function<double(double)> f1, function<double(double)> f2) {
    return [f1, f2](double x) { return f1(x) * f2(x); };
}


double max(double a, double b, double c) {
    double x = max(a, b);
    double y = max(b, c);
    return max(x, y);
}


double x_i(int i, int n) {
    return (double) (2*i)/n;
}


function<double(double)> e(int i, int n) {
    return [i, n](double x) {
        if (x > x_i(i - 1, n) && x <= x_i(i, n)) {
            return (double) n/2*x - i + 1;
        } else if (x > x_i(i, n) && x < x_i(i + 1, n)) {
            return (double) -n/2*x + i + 1;
        } else {
            return 0.0;
        }
    };
}


function<double(double)> e_der(int i, int n) {
    return [i, n](double x) {
        if (x > x_i(i - 1, n) && x <= x_i(i, n)) {
            return (double) n/2;
        } else if (x > x_i(i, n) && x < x_i(i + 1, n)) {
            return (double) -n/2;
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


double B(double x, int i, int j, int n) {
    function<double(double)> f = multiply_two_functions(e_der(i, n), e_der(j, n));
    double start = max(0.0, x_i(i - 1, n), x_i(j - 1, n));
    double end = min(x_i(i + 1, n), x_i(j + 1, n));
    return e(i, n)(x) * e(j, n)(x) - integration(f, start, end);
}


double L(double x, int i, int n) {
    return 20 * e(i, n) (x);
}


Eigen::Matrix<float, Dynamic, Dynamic> initialize_main_matrix(int n) {
    Eigen::Matrix<float, Dynamic, Dynamic> matrix;
    matrix.resize(n, n);
    matrix.setZero();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix(i, j) = B(0, i, j, n);
        }
    }
    return matrix;
}


Eigen::Matrix<float, Dynamic, Dynamic> initialize_constant_matrix(int n) {
    Eigen::Matrix<float, Dynamic, Dynamic> matrix;
    matrix.resize(n, 1);
    for (int i = 0; i < n; i++) {
        matrix(i, 0) = L(0, i, n);
    }
    return matrix;
}


Eigen::Matrix<float, Dynamic, Dynamic> solve_linear_matrix_equation(Eigen::Matrix<float, Dynamic, Dynamic> main_matrix, Eigen::Matrix<float, Dynamic, Dynamic> constant_matrix, int n) {
    Eigen::ColPivHouseholderQR<Eigen::Matrix<float, Dynamic, Dynamic>> dec(main_matrix);
    Eigen::Matrix<float, Dynamic, Dynamic> solution;
    solution.resize(n, 1);
    solution = dec.solve(constant_matrix);
    return solution;
}


vector<double> initialize_x_values(int n) {
    vector<double> result;
    double value = 0.0;
    double step = 2.0/n;

    for (int i = 0; i < n; i++) {
        value += step;
        result.push_back(value);
    }
    return result;
}


vector<double> initialize_y_values(Eigen::Matrix<float, Dynamic, 1> matrix, vector<double> x_values, int n) {
    vector<double> result;
    matrix.resize(n, 1);
    for (int i = 0; i < n; i++) {
        double sum = 0;
        for (int j = 0; j < n; j++) {
            sum += matrix(j, 0) * (e(j, n) (x_values[i]));
        }
        result.push_back(sum);
    }
    return result;
}


void print_calculated_value(vector<double> x, vector<double> y, int n) {
    cout << "Equation solved. Calculated values (x, y): " << endl;
    for (int i = 0; i < n; i++) {
        cout << "(" + to_string(x[i]) + ", " + to_string(y[i]) + ") ";
    }
}


void draw_plot(vector<double> x, vector<double> y) {
    plt::figure();
    plt::plot(x, y);
    plt::title("Solving the equation.");
    plt::show();
}


void check_arguments(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "invalid number of arguments!" << endl;
        exit(1);
    }
    const char *input_argument = argv[1];
    for (int i = 0; i < strlen(input_argument); i++) {
        if(!isdigit(input_argument[i])) {
            cout<<"The input argument must be an integer!" << endl;
            exit(2);
        }
    }
    if(atoi(argv[1]) < 5 || atoi(argv[1]) > 1000) {
        cout << "The input argument must be in the range of 5 to 1000!" << endl;
        exit(3);
    };
}


int main(int argc, char* argv[]) {
    check_arguments(argc, argv);
    int n = atoi(argv[1]);

    Eigen::Matrix<float, Dynamic, Dynamic> main_matrix = initialize_main_matrix(n);
    Eigen::Matrix<float, Dynamic, Dynamic> constant_matrix = initialize_constant_matrix(n);
    Eigen::Matrix<float, Dynamic, Dynamic> solution = solve_linear_matrix_equation(main_matrix, constant_matrix, n);

    vector<double> x_values = initialize_x_values(n);
    vector<double> y_values = initialize_y_values(solution, x_values, n);

    print_calculated_value(x_values, y_values, n);
    draw_plot(x_values, y_values);

    return 0;
}
