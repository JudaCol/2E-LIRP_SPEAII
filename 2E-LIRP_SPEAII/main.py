# Archivo main para la ejecucion del algoritmo genetico
# from functions import *
from operators_ga import *
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
prob_mut = 0.01
individuos = []
idx_externa = []
p_externa = []
ind_k = []
demand_cr_poblation = []
demand_cl_poblation = []
final_inventarioQ = []
final_inventarioI = []
valores_f1 = []
valores_f2 = []
n_generaciones = 10

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
    costo_f1 = np.sum([cost_loc_cl, cost_loc_cr, costprod, costtrans, costinv, costrut2, costrut1, costveh2, costveh1])
    valores_f1.append(costo_f1)
    # costos f2
    cost_sufr_hum = fitness_f2(rutas_segundo_lv, n_periodos, costo_humano, n_centroslocales)
    costo_f2 = -cost_sufr_hum
    valores_f2.append(costo_f2)
Q_poblation = final_inventarioQ
I_poblation = final_inventarioI
f1_poblation = valores_f1
f2_poblation = valores_f2

# calculo del fitness de la poblacion
fitness_f, densidades_f = fitness(n_poblacion, valores_f1, valores_f2)
# seleccion de los individuos no dominados
for ind, fit in fitness_f.items():
    if fit <= 1:
        idx_externa.append(ind)
        p_externa.append(individuos[ind])
        ind_k.append(individuos[ind])
fitness_dict = {}
fitness_f_ordlist = sorted(fitness_f.items(), key=operator.itemgetter(1), reverse=False)
for domin in fitness_f_ordlist:
    fitness_dict[domin[0]] = [domin[1]]
# ordenamientos de las densidades
densidades_dict = {}
densidades_f_ordlist = sorted(densidades_f.items(), key=operator.itemgetter(1), reverse=True)
for domin_d in densidades_f_ordlist:
    densidades_dict[domin_d[0]] = [domin_d[1]]
# seleccion de los demas individuos para completar la poblacion
for idf, fit in fitness_dict.items():
    if idf in idx_externa:
        ""
    else:
        idx_externa.append(idf)
        p_externa.append(individuos[idf])
# reordenamiento de los valores de las funciones objetivo para proceder con los operadores geneticos
valores_f1_o = []
valores_f2_o = []
fitness_dict_o = {}
for f, idg in enumerate(idx_externa):
    valores_f1_o.append(valores_f1[idg])
    valores_f2_o.append(valores_f2[idg])
    fitness_dict_o[f] = fitness_dict[idg]

print("Ejecutando algoritmo genetico...")
# aplicacion de los operadores en cada generacion
for _ in tqdm(range(1, n_generaciones+1)):
    # inicio operadores geneticos
    # seleccion de los padres
    idx_parents = selection_padres(n_poblacion, fitness_dict_o)
    # cruce
    p_crossed, hijos, demand_cr_hijos, demand_cl_hijos, Q_hijos, I_hijos, f1_hijos, f2_hijos = crossover(p_externa, idx_parents, n_clientes, n_centroslocales, n_centrosregionales, n_periodos, n_productos, n_vehiculos_s, n_vehiculos_p, capacidad_cr, capacidad_cl, capacidad_vehiculos_p, capacidad_vehiculos_s, demanda_clientes, demand_cl_poblation, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, costo_vehiculos_s, costo_vehiculos_p, costo_humano)
    # mutacion
    hijos, demandas_cr_hijos, Q_hijos, I_hijos, f1_hijos, f2_hijos = mutation(hijos, demand_cr_hijos, n_centrosregionales, capacidad_cr, n_periodos, n_productos, inventario, costo_instalaciones_cl, costo_instalaciones_cr, costo_compraproductos, costo_transporte, costo_inventario, costo_rutas_s, costo_rutas_p, n_centroslocales, costo_vehiculos_s, costo_vehiculos_p, costo_humano, Q_hijos, I_hijos, f1_hijos, f2_hijos, prob_mut)
    # actualizacion del orden de los parametros de los padres
    demand_cr_poblation_o = []
    demand_cl_poblation_o = []
    Q_poblation_o = []
    I_poblation_o = []
    f1_poblation_o = []
    f2_poblation_o = []
    for padre in idx_parents:
        demand_cr_poblation_o.append(demand_cr_poblation[padre])
        demand_cl_poblation_o.append(demand_cl_poblation[padre])
        Q_poblation_o.append(Q_poblation[padre])
        I_poblation_o.append(I_poblation[padre])
        f1_poblation_o.append(f1_poblation[padre])
        f2_poblation_o.append(f2_poblation[padre])
    # consolidacion de la nueva poblacion
    big_poblation = p_crossed + hijos
    demand_cr_big_poblation = demand_cr_poblation_o + demandas_cr_hijos
    demand_cl_big_poblation = demand_cl_poblation_o + demand_cl_hijos
    Q_big_poblation = Q_poblation_o + Q_hijos
    I_big_poblation = I_poblation_o + I_hijos
    f1_big_poblation = f1_poblation_o + f1_hijos
    f2_big_poblation = f2_poblation_o + f2_hijos
    # calculo del fitness de la poblacion y reorganizacion de los individuos
    idx_externa = []
    ind_k = []
    p_externa = []
    demand_cr_pexterna = []
    demand_cl_pexterna = []
    Q_pexterna = []
    I_pexterna = []
    f1_pexterna = []
    f2_pexterna = []
    fitness_f, densidades_f = fitness(len(big_poblation), f1_big_poblation, f2_big_poblation)
    # seleccion de los individuos no dominados
    for ind, fit in fitness_f.items():
        if fit <= 1:
            idx_externa.append(ind)
            ind_k.append(big_poblation[ind])
    # ordenamiento de los fitness
    fitness_dict = {}
    fitness_f_ordlist = sorted(fitness_f.items(), key=operator.itemgetter(1), reverse=False)
    for domin in fitness_f_ordlist:
        fitness_dict[domin[0]] = [domin[1]]
    # ordenamientos de las densidades
    densidades_dict = {}
    densidades_f_ordlist = sorted(densidades_f.items(), key=operator.itemgetter(1), reverse=False)
    for domin_d in densidades_f_ordlist:
        densidades_dict[domin_d[0]] = [domin_d[1]]
    # seleccion de los demas individuos para completar la poblacion
    if len(ind_k) == n_poblacion:
        p_externa = ind_k
    elif len(ind_k) < n_poblacion:
        while len(ind_k) < n_poblacion:
            for idf, fit in fitness_dict.items():
                if idf in idx_externa:
                    ""
                else:
                    idx_externa.append(idf)
                    ind_k.append(big_poblation[idf])
                    if len(ind_k) < n_poblacion:
                        ""
                    else:
                        break
        p_externa = ind_k
    elif len(ind_k) > n_poblacion:
        while len(ind_k) > n_poblacion:
            for idd, den in densidades_dict.items():
                if idd in idx_externa:
                    no_den = np.where(idx_externa == idd)
                    ind_k.pop(no_den[0])
                    idx_externa.pop(idd)
                    if len(ind_k) < n_poblacion:
                        ""
                    else:
                        break
        p_externa = ind_k
    # reordenamiento de los valores de las funciones objetivo para proceder con los operadores geneticos
    demand_cr_poblation = []
    demand_cl_poblation = []
    Q_poblation = []
    I_poblation = []
    valores_f1_o = []
    valores_f2_o = []
    fitness_dict_o = {}
    for f, idg in enumerate(idx_externa):
        valores_f1_o.append(f1_big_poblation[idg])
        valores_f2_o.append(f2_big_poblation[idg])
        fitness_dict_o[f] = fitness_dict[idg]
        demand_cr_poblation.append(demand_cr_big_poblation[idg])
        demand_cl_poblation.append(demand_cl_big_poblation[idg])
        Q_poblation.append(Q_big_poblation[idg])
        I_poblation.append(I_big_poblation[idg])
    f1_poblation = valores_f1_o
    f2_poblation = valores_f2_o
TimeEnd = time.time()
print("El tiempo de ejecucion del algoritmo es de {} segundos".format(TimeEnd-TimeStart))
# extraccion de mejores individuos
fitness_p_ext, densidades_p_ext = fitness(len(p_externa), f1_poblation, f2_poblation)
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
    best_inds.append(p_externa[idx_b])
    best_q.append(Q_poblation[idx_b])
    best_i.append(I_poblation[idx_b])
    best_f1.append(f1_poblation[idx_b])
    best_f2.append(f2_poblation[idx_b])
# visualizacion de los individuos
print("\n FITNESS DE LOS MEJORES INDIVIDUOS DE LA APROXIMACION DE PARETO - FITNESS MENOR O IGUAL A 1 \n")
print("individuo                     f1                                f2")
for idx in idx_best:
    print("{0:3d}                    {1:.3f}                      {2:.3f}".format(idx + 1, f1_poblation[idx], f2_poblation[idx]))
print("\n FITNESS DE LOS INDIVIDUOS DE LA APROXIMACION DE PARETO - POBLACION FINAL \n")
print("individuo                     f1                                f2")
for id_bob in range(len(p_externa)):
    print("{0:3d}                    {1:.3f}                      {2:.3f}".format(id_bob + 1, f1_poblation[id_bob], f2_poblation[id_bob]))

# graficos
# graficos de fitness
# valores del eje x - f1
# valores del eje y - f2
x = []
y = []
f1_pareto_g = {}
for xs in idx_best:
    f1_pareto_g[xs] = f1_poblation[xs]

f1_pareto_o = sorted(f1_pareto_g.items(), key=operator.itemgetter(1), reverse=False)  # ordenamos las distancias de apilamiento de mayor a menor

for xi in f1_pareto_o:
    x.append(xi[1])
    y.append(f2_poblation[xi[0]])

# parametros del grafico
fig, axs = plt.subplots()
# grafico fitness total
axs.plot(x, y, marker='o', ms=5, mec='r', mfc='r', linestyle=':', color='g')
axs.set_title("Frontera de pareto")
axs.set_xlabel("F1")
axs.set_ylabel("F2")
plt.show()
