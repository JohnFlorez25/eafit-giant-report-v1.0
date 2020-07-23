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

