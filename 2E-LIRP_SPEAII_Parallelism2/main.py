# Archivo main para la ejecucion del algoritmo genetico
# from functions import *
from operators_ga import *
import threading as thr
import multiprocessing as mp
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

# parametros de entrada iniciales
n_clientes = 7  # numero de clientes
n_productos = 3  # numero de productos
n_periodos = 3  # numero de periodos
n_vehiculos_p = 6  # numero de vehiculos de primer nivel
n_vehiculos_s = 9  # numero de vehiculos de segundo nivel
n_centrosregionales = 4  # numero de centros regionales
n_centroslocales = 7  # numero de centros locales
n_poblacion = 100  # numero de inidividuos a generar
prob_mut = 0.1
n_generaciones = 10
individuos = []
demand_cr_poblation = []
demand_cl_poblation = []
final_inventarioQ = []
final_inventarioI = []
valores_f1 = []
valores_f2 = []


TimeStart = time.time()
# obtencion de las demandas y capacidades dadas en matrices en un archivo de excel
demanda_clientes, capacidad_vehiculos_p, capacidad_vehiculos_s, capacidad_cr, capacidad_cl, inventario, costo_inventario, costo_instalaciones_cr, costo_instalaciones_cl, costo_vehiculos_p, costo_vehiculos_s, costo_compraproductos, costo_transporte, costo_rutas_p, costo_rutas_s, costo_humano = read_data(n_clientes, n_productos, n_periodos, n_vehiculos_p, n_vehiculos_s, n_centrosregionales, n_centroslocales)

print("Generando poblacion inicial...")
# generacion de la poblacion inicial
for i in tqdm(range(n_poblacion)):
    # Generacion de un individuo
    asignaciones_primer_lv, asignaciones_segundo_lv, rutas_primer_lv, rutas_segundo_lv, demandas_cr_full, demandas_cl = individuo(n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cl, capacidad_cr, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes)
    # almacenamiento del individuo en una lista
    individuos.append([asignaciones_primer_lv, rutas_primer_lv, asignaciones_segundo_lv, rutas_segundo_lv])
    # almacenamiento de las demandas de centros regionales en una lista
    demand_cr_poblation.append(demandas_cr_full)
    # almacenamiento de las demandas de centros locales en una lista
    demand_cl_poblation.append(demandas_cl)
    # Generacion de los valores Q e I de la gestion de inventarios
    valoresQ, valoresI = fun_inventario(demandas_cr_full, n_periodos, n_productos, n_centrosregionales, capacidad_cr, inventario)
    final_inventarioQ.append(valoresQ)
    final_inventarioI.append(valoresI)
    # costos f1
    cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1 = fitness_f1(n_periodos, n_productos, asignaciones_segundo_lv, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, valoresQ, valoresI, rutas_segundo_lv, rutas_primer_lv, costo_rutas_s, costo_rutas_p, n_centroslocales, n_centrosregionales, costo_vehiculos_s, costo_vehiculos_p)
    o = 10**-3
    costprod = costprod*o
    costtrans = costtrans*o
    costinv = costinv*o
    costo_f1 = round(np.sum([cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1]), 3)
    valores_f1.append(costo_f1)
    # costos f2
    cost_sufr_hum = round(fitness_f2(rutas_segundo_lv, n_periodos, costo_humano, n_centroslocales), 3)
    costo_f2 = -cost_sufr_hum
    valores_f2.append(costo_f2)
Q_poblation = final_inventarioQ
I_poblation = final_inventarioI
f1_poblation = valores_f1
f2_poblation = valores_f2

# fraccionamiento de la poblacion
n_poblacion_f = int(n_poblacion/2)
valores_f1_1 = valores_f1[:n_poblacion_f]
valores_f2_1 = valores_f2[:n_poblacion_f]
individuos_1 = individuos[:n_poblacion_f]
demand_cr_poblation_1 = demand_cr_poblation[:n_poblacion_f]
demand_cl_poblation_1 = demand_cl_poblation[:n_poblacion_f]
Q_poblation_1 = Q_poblation[:n_poblacion_f]
I_poblation_1 = I_poblation[:n_poblacion_f]
# calculo de fitnes de la poblacion fraccionada
p_externa1, f1_poblation_o1, f2_poblation_o1, fitness_dict_o1, demand_cr_poblation_o1, demand_cl_poblation_o1, Q_poblation_o1, I_poblation_o1 = fit_den_pext(n_poblacion_f, valores_f1_1, valores_f2_1, individuos_1, demand_cr_poblation_1, demand_cl_poblation_1, Q_poblation_1, I_poblation_1)

valores_f1_2 = valores_f1[n_poblacion_f:]
valores_f2_2 = valores_f2[n_poblacion_f:]
individuos_2 = individuos[n_poblacion_f:]
demand_cr_poblation_2 = demand_cr_poblation[n_poblacion_f:]
demand_cl_poblation_2 = demand_cl_poblation[n_poblacion_f:]
Q_poblation_2 = Q_poblation[n_poblacion_f:]
I_poblation_2 = I_poblation[n_poblacion_f:]
# calculo de fitness de la poblacion fraccionada
p_externa2, f1_poblation_o2, f2_poblation_o2, fitness_dict_o2, demand_cr_poblation_o2, demand_cl_poblation_o2, Q_poblation_o2, I_poblation_o2 = fit_den_pext(n_poblacion_f, valores_f1_2, valores_f2_2, individuos_2, demand_cr_poblation_2, demand_cl_poblation_2, Q_poblation_2, I_poblation_2)


# aplicacion de los operadores en cada generacion
colector = mp.Array('d', [0, 0])
colector.value = [[], []]
print("Ejecutando algoritmo genetico en hilos...")
t1 = thr.Thread(target=run_ga, args=(p_externa1, n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cr, capacidad_cl, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demand_cl_poblation_o1, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano, n_generaciones, fitness_dict_o1, demand_cr_poblation_o1, Q_poblation_o1, I_poblation_o1, f1_poblation_o1, f2_poblation_o1, prob_mut, colector, 0))
t2 = thr.Thread(target=run_ga, args=(p_externa2, n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cr, capacidad_cl, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demand_cl_poblation_o2, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano, n_generaciones, fitness_dict_o2, demand_cr_poblation_o2, Q_poblation_o2, I_poblation_o2, f1_poblation_o2, f2_poblation_o2, prob_mut, colector, 1))
t1.start()
t2.start()
t1.join()
t2.join()
TimeEnd = time.time()
print("El tiempo de ejecucion del algoritmo es de {} segundos".format(TimeEnd-TimeStart))
# extraccion de individuos de la ultima generacion de cada isla
ind_pareto = colector.value[0][0] + colector.value[1][0]
q_pareto = colector.value[0][1] + colector.value[1][1]
i_pareto = colector.value[0][2] + colector.value[1][2]
f1_pareto = colector.value[0][3] + colector.value[1][3]
f2_pareto = colector.value[0][4] + colector.value[1][4]
# extraccion de mejores individuos
fitness_p_ext, densidades_p_ext = fitness(len(ind_pareto), f1_pareto, f2_pareto)
idx_best = []
fit_best = []
for idx_pext, fit_pext in fitness_p_ext.items():
    if fit_pext <= 1:
        idx_best.append(idx_pext)
        fit_best.append(fit_pext)
# almacenamiento de las estructuras de los mejores individuos en variables independientes
best_inds = []
best_q = []
best_i = []
best_f1 = []
best_f2 = []
for idx_b in idx_best:
    best_inds.append(ind_pareto[idx_b])
    best_q.append(Q_poblation[idx_b])
    best_i.append(I_poblation[idx_b])
    best_f1.append(f1_poblation[idx_b])
    best_f2.append(f2_poblation[idx_b])

# visualizacion de los individuos
print("\n FITNESS DE LOS MEJORES INDIVIDUOS DE LA APROXIMACION DE PARETO - FITNESS MENOR O IGUAL A 1 \n")
print("individuo                     f1                                f2")
for idx in idx_best:
    print("{0:3d}                    {1:.3f}                      {2:.3f}".format(idx + 1, f1_pareto[idx], f2_pareto[idx]))
print("\n FITNESS DE LOS INDIVIDUOS DE LA APROXIMACION DE PARETO - POBLACION FINAL \n")
print("individuo                     f1                                f2")
for id_bob in range(len(ind_pareto)):
    print("{0:3d}                    {1:.3f}                      {2:.3f}".format(id_bob + 1, f1_pareto[id_bob], f2_pareto[id_bob]))


# graficos
# graficos de fitness
# valores del eje x - f1
# valores del eje y - f2
x = []
y = []
f1_pareto_g = {}
for xs in idx_best:
    f1_pareto_g[xs] = f1_pareto[xs]

f1_pareto_o = sorted(f1_pareto_g.items(), key=operator.itemgetter(1), reverse=False)  # ordenamos las distancias de apilamiento de mayor a menor

for xi in f1_pareto_o:
    x.append(xi[1])
    y.append(f2_pareto[xi[0]])

# parametros del grafico
fig, axs = plt.subplots()
# grafico fitness total
axs.plot(x, y, marker='o', ms=5, mec='r', mfc='r', linestyle=':', color='g')
axs.set_title("Frontera de pareto")
axs.set_xlabel("F1")
axs.set_ylabel("F2")
plt.show()
