#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>

using namespace std;

class LinearRegression {
private:
    vector<double> weights;
    double bias;
    bool is_fitted;

    vector<vector<double> > transpose(const vector<vector<double> >& A) const {
        int r = A.size();
        int c = A[0].size();
        vector<vector<double> > At(c, vector<double>(r));
        for (int i = 0; i < r; ++i) {
            for (int j = 0; j < c; ++j) {
                At[j][i] = A[i][j];
            }
        }
        return At;
    }

    vector<vector<double> > multiply(const vector<vector<double> >& A, const vector<vector<double> >& B) const {
        int r1 = A.size();
        int c1 = A[0].size();
        int c2 = B[0].size();
        vector<vector<double> > C(r1, vector<double>(c2, 0.0));
        for (int i = 0; i < r1; ++i) {
            for (int k = 0; k < c1; ++k) {
                for (int j = 0; j < c2; ++j) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    vector<vector<double> > invert(const vector<vector<double> >& A) const {
        int n = A.size();
        vector<vector<double> > aug(n, vector<double>(2 * n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) aug[i][j] = A[i][j];
            aug[i][n + i] = 1.0;
        }

        for (int i = 0; i < n; ++i) {
            double pivot = aug[i][i];
            int max_r = i;
            for (int k = i + 1; k < n; ++k) {
                if (abs(aug[k][i]) > abs(pivot)) {
                    pivot = aug[k][i];
                    max_r = k;
                }
            }
            if (max_r != i) swap(aug[i], aug[max_r]);

            pivot = aug[i][i];
            if (abs(pivot) < 1e-11) {
                cerr << "Error: La matriz es singular y no se puede invertir." << endl;
                return {};
            }

            for (int j = 0; j < 2 * n; ++j) aug[i][j] /= pivot;

            for (int k = 0; k < n; ++k) {
                if (k != i) {
                    double factor = aug[k][i];
                    for (int j = 0; j < 2 * n; ++j) {
                        aug[k][j] -= factor * aug[i][j];
                    }
                }
            }
        }

        vector<vector<double> > inv(n, vector<double>(n));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                inv[i][j] = aug[i][n + j];
            }
        }
        return inv;
    }

public:
    LinearRegression() : bias(0.0), is_fitted(false) {}

    static void data_scaling(vector<vector<double> >& X) {
        if (X.empty()) return;
        int rows = X.size();
        int cols = X[0].size();
        for (int j = 0; j < cols; ++j) {
            double min_val = X[0][j];
            double max_val = X[0][j];
            for (int i = 1; i < rows; ++i) {
                if (X[i][j] < min_val) min_val = X[i][j];
                if (X[i][j] > max_val) max_val = X[i][j];
            }
            double range = max_val - min_val;
            if (range > 1e-9) {
                for (int i = 0; i < rows; ++i) {
                    X[i][j] = (X[i][j] - min_val) / range;
                }
            }
        }
    }

    void fit(const vector<vector<double> >& X, const vector<double>& y) {
        int m = X.size();
        int n = X[0].size();

        vector<vector<double> > A(m, vector<double>(n + 1));
        for (int i = 0; i < m; ++i) {
            A[i][0] = 1.0;
            for (int j = 0; j < n; ++j) {
                A[i][j + 1] = X[i][j];
            }
        }

        vector<vector<double> > At = transpose(A);
        vector<vector<double> > AtA = multiply(At, A);
        vector<vector<double> > AtA_inv = invert(AtA);

        vector<vector<double> > y_mat(m, vector<double>(1));
        for (int i = 0; i < m; ++i) y_mat[i][0] = y[i];

        vector<vector<double> > At_y = multiply(At, y_mat);
        vector<vector<double> > w_opt = multiply(AtA_inv, At_y);

        bias = w_opt[0][0];
        weights.resize(n);
        for (int j = 0; j < n; ++j) {
            weights[j] = w_opt[j + 1][0];
        }
        is_fitted = true;
    }

    vector<double> predict(const vector<vector<double> >& X) const {
        int m = X.size();
        int n = X[0].size();
        vector<double> y_hat(m, 0.0);
        for (int i = 0; i < m; ++i) {
            double pred = bias;
            for (int j = 0; j < n; ++j) {
                pred += weights[j] * X[i][j];
            }
            y_hat[i] = pred;
        }
        return y_hat;
    }

    double score(const vector<vector<double> >& X, const vector<double>& y) const {
        vector<double> y_hat = predict(X);
        double sum_sq_err = 0.0;
        int k = y.size();
        for (int i = 0; i < k; ++i) {
            double err = y_hat[i] - y[i];
            sum_sq_err += err * err;
        }
        return sum_sq_err / k;
    }

    vector<double> getWeights() const { return weights; }
    double getBias() const { return bias; }
};

void load_csv(const string& filename, vector<vector<double> >& X, vector<double>& y, bool skip_header = true) {
    ifstream file(filename.c_str());
    if (!file.is_open()) {
        cerr << "Error crítico: No se pudo abrir el archivo CSV -> " << filename << endl;
        return;
    }
    string line;
    if (skip_header && getline(file, line)) {}

    while (getline(file, line)) {
        stringstream ss(line);
        string cell;
        vector<double> row;
        while (getline(ss, cell, ',')) {
            if (!cell.empty()) {
                stringstream convert(cell);
                double val;
                convert >> val;
                row.push_back(val);
            }
        }
        if (!row.empty()) {
            y.push_back(row.back());
            row.pop_back();
            X.push_back(row);
        }
    }
    file.close();
}

void train_test_split(const vector<vector<double> >& X, const vector<double>& y, double train_ratio,
                      vector<vector<double> >& X_train, vector<double>& y_train,
                      vector<vector<double> >& X_test, vector<double>& y_test) {
    int train_size = static_cast<int>(X.size() * train_ratio);
    X_train.assign(X.begin(), X.begin() + train_size);
    y_train.assign(y.begin(), y.begin() + train_size);
    X_test.assign(X.begin() + train_size, X.end());
    y_test.assign(y.begin() + train_size, y.end());
}

int main() {
    cout << fixed << setprecision(4);

    cout << "======================================================================" << endl;
    cout << "   PRACTICA 2: REGRESION LINEAL MULTIPLE CON ORIENTACION A OBJETOS" << endl;
    cout << "======================================================================" << endl;
    cout << "\n[ANALISIS 1] Dataset de Ventas de Helados (Regresion Simple)" << endl;
    cout << "----------------------------------------------------------------------" << endl;
    vector<vector<double> > X_ice;
    vector<double> y_ice;
    load_csv("Ice_cream_selling_data.csv", X_ice, y_ice);

    vector<vector<double> > X_tr_ice, X_te_ice;
    vector<double> y_tr_ice, y_te_ice;
    train_test_split(X_ice, y_ice, 0.70, X_tr_ice, y_tr_ice, X_te_ice, y_te_ice);

    LinearRegression modelo_helados;
    modelo_helados.fit(X_tr_ice, y_tr_ice);

    cout << "-> Muestras totales usadas para entrenamiento (70%): " << X_tr_ice.size() << endl;
    cout << "-> Muestras totales usadas para prueba/score (30%): " << X_te_ice.size() << endl;
    cout << "-> Intercepto optimo del modelo (Bias / b_op): " << modelo_helados.getBias() << endl;
    cout << "-> Peso optimo de la pendiente (Weight / w1):   " << modelo_helados.getWeights()[0] << endl;
    cout << "-> Error Cuadratico Medio en Entrenamiento (Score Train): " << modelo_helados.score(X_tr_ice, y_tr_ice) << endl;
    cout << "-> Error Cuadratico Medio en Prueba (Score Test Externo): " << modelo_helados.score(X_te_ice, y_te_ice) << endl;
    cout << "\n[ANALISIS 2] Dataset de Calificaciones de Estudiantes (Regresion Multiple)" << endl;
    cout << "----------------------------------------------------------------------" << endl;
    vector<vector<double> > X_stud;
    vector<double> y_stud;
    load_csv("student_exam_scores.csv", X_stud, y_stud);

    LinearRegression::data_scaling(X_stud);

    vector<vector<double> > X_tr_stud, X_te_stud;
    vector<double> y_tr_stud, y_te_stud;
    train_test_split(X_stud, y_stud, 0.70, X_tr_stud, y_tr_stud, X_te_stud, y_te_stud);

    LinearRegression modelo_estudiantes;
    modelo_estudiantes.fit(X_tr_stud, y_tr_stud);

    cout << "-> Muestras totales usadas para entrenamiento (70%): " << X_tr_stud.size() << endl;
    cout << "-> Muestras totales usadas para prueba/score (30%): " << X_te_stud.size() << endl;
    cout << "-> Intercepto optimo del modelo (Bias / b_op): " << modelo_estudiantes.getBias() << endl;
    cout << "-> Vector de pesos óptimos para cada variable (Weights):" << endl;
    vector<double> pesos_est = modelo_estudiantes.getWeights();
    cout << "   [ Variable 1 (Horas estudio): " << pesos_est[0] << " ]" << endl;
    cout << "   [ Variable 2 (Horas sueño):   " << pesos_est[1] << " ]" << endl;
    cout << "   [ Variable 3 (Asistencia %):  " << pesos_est[2] << " ]" << endl;
    cout << "   [ Variable 4 (Notas previas): " << pesos_est[3] << " ]" << endl;
    cout << "-> Error Cuadratico Medio en Entrenamiento (Score Train): " << modelo_estudiantes.score(X_tr_stud, y_tr_stud) << endl;
    cout << "-> Error Cuadratico Medio en Prueba (Score Test Externo): " << modelo_estudiantes.score(X_te_stud, y_te_stud) << endl;
    vector<double> y_hat_prueba = modelo_estudiantes.predict(X_te_stud);
    cout << "\n[DEMOSTRACION] Comparacion de Predicciones vs Valores Reales (Primeras 3 muestras de Test):" << endl;
    for (int i = 0; i < 3 && i < (int)y_hat_prueba.size(); ++i) {
        cout << "   * Muestra #" << i + 1 
             << " -> Calificacion Predicha (y_hat): " << y_hat_prueba[i] 
             << " | Calificacion Real en CSV (y): " << y_te_stud[i] << endl;
    }

    cout << "\n======================================================================" << endl;
    cout << "   EJECUCION FINALIZADA EXITOSAMENTE" << endl;
    cout << "======================================================================" << endl;

    return 0;
}
