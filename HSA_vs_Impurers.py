import pandas as pd
import numpy as np
import random
from scipy.stats import wilcoxon
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt

#region Datasets
def Cancer_process(test_size=0):
    data_complete = pd.read_csv("breast-cancer-dataset.csv")

    data_complete.replace("#", np.nan, inplace=True)
    data_complete.dropna(inplace=True)
    data_complete.drop('S/N', axis=1, inplace=True)
    data_complete.drop('Year', axis=1, inplace=True)

    map_quadrant = {'Upper inner': 1, 'Upper outer': 2, 'Lower outer': 3, 'Lower inner': 4}
    map_breast_side = {'Right': 1, 'Left': 2}
    map_diagnosis = {'Benign': 1, 'Malignant': 2}

    data_complete["Breast"] = data_complete["Breast"].map(map_breast_side).astype(float)
    data_complete["Breast Quadrant"] = data_complete["Breast Quadrant"].map(map_quadrant).astype(float)
    data_complete["Diagnosis Result"] = data_complete["Diagnosis Result"].map(map_diagnosis).astype(float)

    scaler = MinMaxScaler()
    data_complete = pd.DataFrame(scaler.fit_transform(data_complete), columns=data_complete.columns).astype(float)

    # Dividir en conjuntos de entrenamiento y prueba
    train_data, test_data = train_test_split(data_complete, test_size=test_size, random_state=42)

    # Amputar datos en el conjunto de prueba
    data_with_nan = test_data.copy()

    missing_mask = np.random.rand(*test_data.shape) < test_size
    data_with_nan = test_data.mask(missing_mask)

    return train_data, test_data, data_with_nan

def Salary_process(test_size=0):
    data_complete = pd.read_csv("ds_salaries.csv")
    # borrado de celdas
    columns_to_delete = ['work_year', 'job_title', 'salary_currency', 'employee_residence', 'company_location']
    data_complete.drop(data_complete.columns[0], axis=1, inplace=True)
    data_complete.drop(columns_to_delete, axis=1, inplace=True)

    # mapeo de datos de manera manual usando diccionario de datos
    map_xp_lvl = {'EN': 1, 'MI': 2, 'SE': 3, 'EX': 4}

    map_emp_type = {'PT': 1, 'FT': 2, 'CT': 3, 'FL': 4}

    map_comp_size = {'S': 1, 'M': 2, 'L': 3}

    data_complete["experience_level"] = data_complete["experience_level"].map(map_xp_lvl)
    data_complete["experience_level"] = data_complete["experience_level"].astype(float)

    data_complete["employment_type"] = data_complete["employment_type"].map(map_emp_type)
    data_complete["employment_type"] = data_complete["employment_type"].astype(float)

    data_complete["company_size"] = data_complete["company_size"].map(map_comp_size)
    data_complete["company_size"] = data_complete["company_size"].astype(float)

    O_data = data_complete.copy()
    O_data = data_complete.astype(float)

    column_names = data_complete.columns

    # scaler = RobustScaler()
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    data_complete = pd.DataFrame(scaler.fit_transform(data_complete), columns=data_complete.columns).astype(float)

    # Dividir en conjuntos de entrenamiento y prueba
    train_data, test_data = train_test_split(data_complete, test_size=test_size, random_state=42)

    # Amputar datos en el conjunto de prueba
    data_with_nan = test_data.copy()

    missing_mask = np.random.rand(*test_data.shape) < test_size
    data_with_nan = test_data.mask(missing_mask)

    return train_data, test_data, data_with_nan

def Diabetes_process(test_size=0):
    data_complete = pd.read_csv("diabetes.csv")
    data_complete.dropna(inplace=True)

    O_data = data_complete.copy()
    O_data = data_complete.astype(float)

    column_names = data_complete.columns

    # scaler = RobustScaler()
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    data_complete = pd.DataFrame(scaler.fit_transform(data_complete), columns=data_complete.columns).astype(float)

    # Dividir en conjuntos de entrenamiento y prueba
    train_data, test_data = train_test_split(data_complete, test_size=test_size, random_state=42)

    # Amputar datos en el conjunto de prueba
    data_with_nan = test_data.copy()

    missing_mask = np.random.rand(*test_data.shape) < test_size
    data_with_nan = test_data.mask(missing_mask)

    return train_data, test_data, data_with_nan

def Wine_process(test_size=0):
    data_complete = pd.read_csv("Wine.csv")
    data_complete.dropna(inplace=True)

    O_data = data_complete.copy()
    O_data = data_complete.astype(float)

    column_names = data_complete.columns

    # scaler = RobustScaler()
    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    data_complete = pd.DataFrame(scaler.fit_transform(data_complete), columns=data_complete.columns).astype(float)

    # Dividir en conjuntos de entrenamiento y prueba
    train_data, test_data = train_test_split(data_complete, test_size=test_size, random_state=42)

    # Amputar datos en el conjunto de prueba
    data_with_nan = test_data.copy()

    missing_mask = np.random.rand(*test_data.shape) < test_size
    data_with_nan = test_data.mask(missing_mask)

    return train_data, test_data, data_with_nan
#endregion

#region Imputers
def mean_imputer(data_with_nan):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(data_with_nan), columns=data_with_nan.columns)

def median_imputer(data_with_nan):
    imputer = SimpleImputer(strategy='median')
    return pd.DataFrame(imputer.fit_transform(data_with_nan), columns=data_with_nan.columns)

def mode_imputer(data_with_nan):
    imputer = SimpleImputer(strategy='most_frequent')
    return pd.DataFrame(imputer.fit_transform(data_with_nan), columns=data_with_nan.columns)

def knn_imputer(data_with_nan):
    imputer = KNNImputer(n_neighbors=5)
    return pd.DataFrame(imputer.fit_transform(data_with_nan), columns=data_with_nan.columns)

def MICE_imputer(data_with_nan, maxiter):
    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=maxiter,  # Aumentado para 50% de datos faltantes
        tol=1e-3,
        imputation_order='ascending'
    )
    # imputer = IterativeImputer(random_state=0)
    return pd.DataFrame(imputer.fit_transform(data_with_nan), columns=data_with_nan.columns)

# def Gain_imputer():
def RandomForest_imputer(data_with_nan):
    data = data_with_nan.copy()
    for col in data.columns:
        if data[col].isnull().any():
            non_nan_data = data[data[col].notnull()]
            nan_data = data[data[col].isnull()]

            X_train = non_nan_data.drop(columns=[col])
            y_train = non_nan_data[col]

            rf = RandomForestRegressor()
            rf.fit(X_train, y_train)

            X_test = nan_data.drop(columns=[col])
            imputed_values = rf.predict(X_test)

            data.loc[data[col].isnull(), col] = imputed_values
    return data
#endregion

#region HSA
def objective_function(original, imputed, mask):
    # Crear una máscara para los valores NaN
    # mask = imputed.isna()

    # Filtrar los valores de acuerdo con la máscara
    original_filtered = original[mask]
    imputed_filtered = imputed[mask]

    # Calcular MAE, MSE y RMSE solo en las posiciones con datos faltantes
    mae = np.mean(np.abs(original_filtered - imputed_filtered))
    mse = np.mean((original_filtered - imputed_filtered) ** 2)
    rmse = np.sqrt(mse)
    objective_value = (0.2 * mae) + (0.3 * mse) + (0.5 * rmse)

    return objective_value

def initialize_harmony_memory(data, HMS):
    harmony_memory = []
    for _ in range(HMS):
        solution = data.copy()
        for col in data.columns:
            if data[col].isnull().any():
                min_val, max_val = data[col].min(), data[col].max()
                if min_val == max_val:
                    # Ajusta los valores para evitar divisiones por cero
                    min_val -= 1
                    max_val += 1
                fill_values = {col: random.uniform(min_val, max_val) for col in data.columns}
                solution[col] = solution[col].fillna(fill_values[col])
        harmony_memory.append(solution)
    return harmony_memory
def generate_new_solution(harmony_memory, data, HMCR, PAR, BW):
    new_solution = data.copy()
    for col in data.columns:
        if data[col].isnull().any():
            for idx in new_solution.index:
                if pd.isnull(new_solution.loc[idx, col]):
                    if random.random() < HMCR:
                        selected_solution = random.choice(harmony_memory)
                        new_solution.loc[idx, col] = selected_solution.loc[idx, col]
                    else:
                        min_val, max_val = data[col].min(), data[col].max()
                        if min_val == max_val:
                            min_val -= 1
                            max_val += 1
                        new_solution.loc[idx, col] = random.uniform(min_val, max_val)
                    if random.random() < PAR:
                        new_solution.loc[idx, col] += random.uniform(-BW, BW)
    return new_solution

def harmony_search(data_with_nan, original_data, mask, HMS, HMCR, PAR, BW, num_iterations):
    harmony_memory = initialize_harmony_memory(data_with_nan, HMS)
    best_solution = None
    best_score = float('inf')

    for iteration in range(num_iterations):
        new_solution = generate_new_solution(harmony_memory, data_with_nan, HMCR, PAR, BW)
        score = objective_function(original_data,new_solution, mask)
        # score = evaluate_solution(new_solution, original_data, data_with_nan, mask)
        scores = [objective_function(original_data,new_solution, mask) for solution in harmony_memory]
        worst_score = max(scores)
        if score < worst_score:
            worst_idx = scores.index(worst_score)
            harmony_memory[worst_idx] = new_solution

        if score < best_score:
            best_score = score
            best_solution = new_solution

        # if iteration % 10 == 0 or iteration == num_iterations - 1:
        #     print(f"Iteration {iteration + 1}/{num_iterations}, Best Score: {best_score}")

    return best_solution
#endregion

#region Results
def calculate_metrics(original, imputed, mask):
    # Crear una máscara para los valores NaN
    # mask = imputed.isna()

    # Filtrar los valores de acuerdo con la máscara
    original_filtered = original[mask]
    imputed_filtered = imputed[mask]

    # Calcular MAE, MSE y RMSE solo en las posiciones con datos faltantes
    mae = np.mean(np.abs(original_filtered - imputed_filtered))
    mse = np.mean((original_filtered - imputed_filtered) ** 2)
    rmse = np.sqrt(mse)

    return mae, mse, rmse

def plot_results(results):
    metrics = ['MAE', 'MSE', 'RMSE']
    for ds_name, missing_results in results.items():
        for metric in metrics:
            plt.figure(figsize=(12, 6))
            for imputer_name in list(missing_results.values())[0].keys():
                percentages = sorted(missing_results.keys())
                values = [missing_results[p][imputer_name][metrics.index(metric)] for p in percentages]

                plt.plot([p * 100 for p in percentages], values, marker='o', label=imputer_name)

            plt.title(f'{ds_name} - {metric}')
            plt.xlabel('Percentage of Missing Data (%)')
            plt.ylabel(metric)
            plt.legend()
            plt.grid(True)
            plt.show()
#endregion

def main_Imputers():
    #region Datos iniciales
    Ds_functions = {
        "Cancer": Cancer_process,
        "Salary": Salary_process,
        "Diabetes": Diabetes_process,
        "Wine": Wine_process
    }

    Imputer_functions = {
        'Mean': mean_imputer,
        'Median': median_imputer,
        'Mode': mode_imputer,
        'KNN': knn_imputer,
        'MICE': MICE_imputer,
        'RandomForest': RandomForest_imputer,  # Placeholder for RandomForest
        'HSA': harmony_search  # Placeholder for RandomForest
    }

    # Lista de porcentajes de datos faltantes
    # missing_percentages = [0.1, 0.2]
    missing_percentages = [0.1, 0.2, 0.3, 0.4, 0.5]

    np.random.seed(0)
    HMS = 10
    HMCR = 0.9
    PAR = 0.3
    BW = 0.01
    num_iterations = 1000

    results = {}
    #endregion

    print('Procesando, por favor espere...')

    for ds_name, ds_func in Ds_functions.items():
        results[ds_name] = {}
        for missing_percentage in missing_percentages:
            train_data, test_data, test_data_with_nan = ds_func(missing_percentage)

            results[ds_name][missing_percentage] = {}
            imputed_ds = {}  # Inicializar imputed_ds dentro del bucle de cada dataset y porcentaje
            mask = test_data_with_nan.isna()
            imputed_data = None

            for imputer_name, imputer_func in Imputer_functions.items():
                if imputer_name == 'HSA':
                    imputed_data = imputer_func(test_data_with_nan, test_data, mask, HMS, HMCR, PAR, BW, num_iterations)
                elif imputer_name =='MICE':
                    imputed_data = imputer_func(test_data_with_nan, int(missing_percentage*100))
                else:
                    imputed_data = imputer_func(test_data_with_nan)

                imputed_ds[imputer_name] = imputed_data
                results[ds_name][missing_percentage][imputer_name] = calculate_metrics(test_data,imputed_data,mask)
                imputed_data = None

            print(f'Terminé la instancia {ds_name} con {missing_percentage*100}% de datos faltantes')

    # Imprimir resultados en el formato deseado
    for ds_name, missing_results in results.items():
        for missing_percentage, imputer_results in missing_results.items():
            for method, (mse, mae, rmse) in imputer_results.items():
                print(f"{ds_name} - {missing_percentage*100}% missing - {method}: MAE = {mse:.4f}, MSE = {mae:.4f}, RMSE = {rmse:.4f}")

    # Guardar resultados en un archivo de texto
    # save_results_to_file(results)

    plot_results(results)


main_Imputers()