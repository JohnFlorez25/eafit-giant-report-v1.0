# Importación de paquetes
import Pkg; 
Pkg.add("Pkg")

# Usar el configurador de paquetes para poder garantizar la implementación de nuevos paquetes
using Pkg  

# Instalando los paquetes
Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("Plots")
Pkg.add("Lathe")
Pkg.add("GLM")
Pkg.add("StatsPlots")
Pkg.add("MLBase")
Pkg.add("StatsBase")

# Cargando los paquetes instalados
using DataFrames
using CSV
using Plots
using Lathe
using GLM
using Statistics
using StatsPlots
using MLBase
using StatsBase

# Se van a leer los datos usando un Archivo tipo CSV y lo vamos a convertir en un DataFrame
df = DataFrame(CSV.File("Life-Expectancy-Data.csv"))
first(df,5)

# Resumen del marco
println(size(df))

#descripción de los datos en el DataFrame
describe(df)

# Verificando nombres de las columnas
names(df)

# Reemplazando espacios vacios de la información en los datos
colnames = Symbol[]
for i in string.(names(df))
    push!(colnames,Symbol(replace(replace(replace(strip(i)," " => "_"),"-" => "_"), "/" => "_")))
end

rename!(df, colnames);

# Verificando nombres de las columnas
names(df)

# Borrando todos aquellos datos clasificados como missing
dropmissing!(df);

# Verificando el tipo de datos del conjunto de datos para saber que efectivamente trabajamos con un tipo de dato válido para le proceso de análisis de valores atípicos
typeof(df.Life_expectancy)

# Graficando en Box Plot para observar los datos atípicos
boxplot(df.Life_expectancy, title = "Box Plot - Life Expectancy", ylabel = "Life Expectancy (years)", size=(400,300))

# Removiendo los valores atípicos
first_percentile = percentile(df.Life_expectancy, 25)
iqr_value = iqr(df.Life_expectancy)
df = df[df.Life_expectancy .>  (first_percentile - 1.5*iqr_value),:];

# Gráfica de densidad
density(df.Life_expectancy , title = "Density Plot - Life Expectancy", ylabel = "Frequency", xlabel = "Life Expectancy", legend = false, size=(400,300))


#Analizando correlación
println("Corelación de la Esperanza de vida con La taza de mortalidad de adultos es ", cor(df.Adult_Mortality,df.Life_expectancy), "\n\n")

# Scatter plot
train_plot = scatter(df.Adult_Mortality,df.Life_expectancy, title = "Life Expectancy vs Adult Mortality Rate", ylabel = "Life Expectancy", xlabel = "Adult Mortality Rate",legend = false, size=(450,300))

# Instalando el paquete para Tsne para poder realizar el embebimiento de los datos
Pkg.add("TSne")
Pkg.add("MLDatasets")
using TSne, Statistics, MLDatasets

# Convirtiendo en Matrix nuestro set de datos 
df_matriz=convert(Matrix,df)

CSV.write("MatrizData.csv", df)

# Realizando el proceso de embebimiento
rescale(A; dims=1) = (A .- mean(A, dims=dims)) ./ max.(std(A, dims=dims), eps())


data = reshape(permutedims(df_matriz[:, :, 1:2500], (3, 1, 2)),
               2500, size(df_matriz, 1)*size(df_matriz, 2));
# Normalize the data, this should be done if there are large scale differences in the dataset
X = rescale(data, dims=1);

Y = tsne(X, 2, 50, 1000, 20.0);

theplot = scatter(Y[:,1], Y[:,2], marker=(2,2,:auto,stroke(0)), color=Int.(allabels[1:size(Y,1)]))
Plots.pdf(theplot, "life_expectancy_tsne.pdf")

# PREPROCESAMIENTO DE LOS DATOS

# One hot encoding
scaled_feature = Lathe.preprocess.OneHotEncode(df,:Status)
select!(df, Not([:Status,:Country]))
first(df,5)

#Partiendo el conjunto de datos

Pkg.add("MLDataUtils")
using(MLDataUtils)
train, test = splitobs(df, at = 0.75);

#Construyendo el modelo de regresión lineal

fm = @formula(Life_expectancy ~ Adult_Mortality)
linearRegressor = lm(fm, train)

#Prediccion y errores

# Prediction
ypredicted_test = predict(linearRegressor, test)
ypredicted_train = predict(linearRegressor, train)

# Test Performance DataFrame (compute squared error)
performance_testdf = DataFrame(y_actual = test[!,:Life_expectancy], y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

# Train Performance DataFrame (compute squared error)
performance_traindf = DataFrame(y_actual = train[!,:Life_expectancy], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error ;

# MAPE function defination
function mape(performance_df)
    mape = mean(abs.(performance_df.error./performance_df.y_actual))
    return mape
end

# RMSE function defination
function rmse(performance_df)
    rmse = sqrt(mean(performance_df.error.*performance_df.error))
    return rmse
end

# Error en los datos de Prueba
println("Error absoluto medio en Test: ",mean(abs.(performance_testdf.error)), "\n")
println("Error porcentual absoluto medio en Test: ",mape(performance_testdf), "\n")
println("Error cuadrático medio en Test: ",rmse(performance_testdf), "\n")
println("Error medio cuadrado en Test: ",mean(performance_testdf.error_sq), "\n")

#  Error en los datos de Entrenamiento
println("Error absoluto medio en Entrenamiento: ",mean(abs.(performance_traindf.error)), "\n")
println("Error porcentual absoluto medio en Entrenamiento: ",mape(performance_traindf), "\n")
println("Error cuadrático medio en Entrenamiento: ",rmse(performance_traindf), "\n")
println("Error medio cuadrado en Entrenamiento: ",mean(performance_traindf.error_sq), "\n")

# DISTRIBUCIÓN DE LOS errores

# Histograma del error en los datos de prueba
histogram(performance_testdf.error, bins = 50, title = "Análisis del eror en los datos de prueba", ylabel = "Frecuencia", xlabel = "Error",legend = false, size=(450,300))

# Histograma del error en los datos de entrenamiento
histogram(performance_traindf.error, bins = 50, title = "Training Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false, size=(450,300))

# Scatter plot de los valores actuales vs los datos de prueba
test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Valor de Predicción vs Valor Actual en Datos de Prueba", ylabel = "Valor de Predicción", xlabel = "Valor Actual", legend = false, size=(700,300))

# Scatter plot de los valores actuales vs los datos de entrenamiento
train_plot = scatter(performance_traindf[!,:y_actual],performance_traindf[!,:y_predicted], title = "Valor de Predicción vs Valor Actual en Datos de Entrenamiento", ylabel = "Valor de Predicción", xlabel = "Valor Actual", legend = false, size=(700,300))

# Definición de la función de validación cruzada
function cross_validation(train,k, fm = @formula(Life_expectancy ~ Adult_Mortality))
    a = collect(Kfold(size(train)[1], k))
    for i in 1:k
        row = a[i]
        temp_train = train[row,:]
        temp_test = train[setdiff(1:end, row),:]
        linearRegressor = lm(fm, temp_train)
        performance_testdf = DataFrame(y_actual = temp_test[!,:Life_expectancy], y_predicted = predict(linearRegressor, temp_test))
        performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]

        println("Error medio del conjunto $i is ",mean(abs.(performance_testdf.error)))
    end
end

#VALIDANDO EN REGESIÓN MULTIPLE

fm = @formula(Life_expectancy ~ Adult_Mortality + infant_deaths + Developing + BMI + Total_expenditure + HIV_AIDS   + Income_composition_of_resources)
linearRegressor = lm(fm, train)

# R Square value of the model
r2(linearRegressor)

#Diagnóstico del modelo
# Prediction
ypredicted_test = predict(linearRegressor, test)
ypredicted_train = predict(linearRegressor, train)

# Test Performance DataFrame
performance_testdf = DataFrame(y_actual = test[!,:Life_expectancy], y_predicted = ypredicted_test)
performance_testdf.error = performance_testdf[!,:y_actual] - performance_testdf[!,:y_predicted]
performance_testdf.error_sq = performance_testdf.error.*performance_testdf.error

# Train Performance DataFrame
performance_traindf = DataFrame(y_actual = train[!,:Life_expectancy], y_predicted = ypredicted_train)
performance_traindf.error = performance_traindf[!,:y_actual] - performance_traindf[!,:y_predicted]
performance_traindf.error_sq = performance_traindf.error.*performance_traindf.error ;

# Error en los datos de Prueba
println("Error absoluto medio en Test: ",mean(abs.(performance_testdf.error)), "\n")
println("Error porcentual absoluto medio en Test: ",mape(performance_testdf), "\n")
println("Error cuadrático medio en Test: ",rmse(performance_testdf), "\n")
println("Error medio cuadrado en Test: ",mean(performance_testdf.error_sq), "\n")

#  Error en los datos de Entrenamiento
println("Error absoluto medio en Entrenamiento: ",mean(abs.(performance_traindf.error)), "\n")
println("Error porcentual absoluto medio en Entrenamiento: ",mape(performance_traindf), "\n")
println("Error cuadrático medio en Entrenamiento: ",rmse(performance_traindf), "\n")
println("Error medio cuadrado en Entrenamiento: ",mean(performance_traindf.error_sq), "\n")

# Distribución de los errores

# Histograma del error en los datos de prueba
histogram(performance_testdf.error, bins = 50, title = "Análisis del eror en los datos de prueba", ylabel = "Frecuencia", xlabel = "Error",legend = false, size=(450,300))

# Histograma del error en los datos de entrenamiento
histogram(performance_traindf.error, bins = 50, title = "Training Error Analysis", ylabel = "Frequency", xlabel = "Error",legend = false, size=(450,300))

# Scatter plot de los valores actuales vs los datos de prueba
test_plot = scatter(performance_testdf[!,:y_actual],performance_testdf[!,:y_predicted], title = "Valor de Predicción vs Valor Actual en Datos de Prueba", ylabel = "Valor de Predicción", xlabel = "Valor Actual", legend = false, size=(700,300))


# Scatter plot de los valores actuales vs los datos de entrenamiento
train_plot = scatter(performance_traindf[!,:y_actual],performance_traindf[!,:y_predicted], title = "Valor de Predicción vs Valor Actual en Datos de Entrenamiento", ylabel = "Valor de Predicción", xlabel = "Valor Actual", legend = false, size=(700,300))


# validación cruzada
cross_validation(train,10, fm)


