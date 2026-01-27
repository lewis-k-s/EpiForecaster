
![][image1]

**E3.3 \- Evaluación de casos de uso**

**(MePreCiSa)**

**Registro de cambios**

| Versión | Autor | Descripción de cambio | Fecha |
| :---: | :---: | :---: | :---: |
|  |  |  |  |
|  |  |  |  |

Tabla de Contenidos

[**Resumen ejecutivo 3**](#resumen-ejecutivo)

[**Introducción 4**](#introducción)

[**Resultados 5**](#resultados)

[Caso de Uso: Movilidad, Calidad del aire y Salud 5](#caso-de-uso:-movilidad,-calidad-del-aire-y-salud)

[Caso de Uso: Propagación de Epidemias 5](#caso-de-uso:-propagación-de-epidemias)

[Calibración de parámetros epidemiológicos y poblaciones ajustando datos reales 6](#calibración-de-parámetros-epidemiológicos-y-poblaciones-ajustando-datos-reales)

[Simulación de escenario con diferentes medidas de confinamiento 11](#simulación-de-escenario-con-diferentes-medidas-de-confinamiento)

[Caso de Uso: Movilidad y Epidemiología de Aguas residuales 13](#caso-de-uso:-movilidad-y-epidemiología-de-aguas-residuales)

[**Conclusiones 14**](#conclusiones)

# Resumen ejecutivo  {#resumen-ejecutivo}

# Introducción {#introducción}

# Resultados {#resultados}

## Caso de Uso: Movilidad, Calidad del aire y Salud {#caso-de-uso:-movilidad,-calidad-del-aire-y-salud}

##

## Caso de Uso: Propagación de Epidemias {#caso-de-uso:-propagación-de-epidemias}

El escenario epidemiológico comienza con la definición del modelo de referencia y los datos que se utilizarán para la calibración y la exploración de intervenciones eficaces. El modelo original se basó en una metapoblación que incluía 7156 áreas correspondientes a municipios. Esta decisión se basó principalmente en los datos de movilidad disponibles al inicio de la pandemia. Sin embargo, durante y después de la pandemia de COVID-19, el Ministerio de Transporte comenzó a publicar un conjunto de datos de movilidad poblacional mucho más completo. Este nuevo conjunto de datos, denominado zonificación MITMA, se compone de 2850 áreas de movilidad correspondientes a distritos o grupos de distritos. A su vez, durante el proyecto MePreCiSa, se desarrolla un conjunto de nuevos indicadores de movilidad para estudios en salud que incluyen indicadores de presencia y viajes en el ámbito territorial de Catalunya.

Para utilizar el conjunto de datos de movilidad MITMA, es necesario reformular el modelo de metapoblación. En consecuencia, el modelo de referencia para este proyecto se define dentro de una metapoblación compuesta por 2850 áreas de movilidad MITMA. Para la demografía poblacional, se utilizaron informes del Instituto Nacional de Estadística. La red de movilidad se construyó con base en matrices de origen-destino basadas en datos telefónicos, reportadas por el Ministerio de Transporte, como se explica en la sección correspondiente. A continuación, procesamos los informes de COVID-19 y los almacenamos en una matriz de datos multidimensional para facilitar su comparación con el resultado de la simulación agregada. El modelo de referencia incluye los mismos parámetros que el modelo original y utilizamos los mismos valores para todos ellos. Finalmente, trasladamos las condiciones iniciales de las zonificaciones originales (municipios) a la zonificación MITMA. Estas condiciones iniciales corresponden a los primeros casos notificados en España el 9 de febrero de 2020\.

A continuación, nos centramos en el desarrollo del flujo de trabajo de exploración del modelo (paso 6\) para la calibración de parámetros y la identificación de intervenciones eficaces. El flujo de trabajo describe todos los pasos necesarios para crear una instancia del modelo, ejecutar la simulación, evaluar los resultados y actualizar el estado interno del algoritmo de exploración del modelo (excepto en el caso del barrido de parámetros). El flujo de trabajo sirve como hoja de ruta para el proyecto, guiando la implementación de cada paso y garantizando que se tengan en cuenta todas las tareas necesarias. La implementación del flujo de trabajo para la exploración paralela del modelo se ha completado. El siguiente paso requerirá conectar el flujo de trabajo de exploración del modelo con los componentes interactivos de aprendizaje y análisis visual. Para conectar el flujo de trabajo de Exploración de Modelos con RapidMiner, desarrollaremos una API REST para que las simulaciones de MMCACovid19 se puedan ejecutar como servicios web. La implementación correcta del flujo de trabajo es clave para el correcto funcionamiento de las simulaciones y la generación de resultados precisos, y será el enfoque del trabajo en los próximos meses.

### Calibración de parámetros epidemiológicos y poblaciones ajustando datos reales {#calibración-de-parámetros-epidemiológicos-y-poblaciones-ajustando-datos-reales}

Para probar el flujo de trabajo de calibración de parámetros, realizamos diferentes experimentos con CMA-ES para evaluar la convergencia y el rendimiento del algoritmo. El objetivo es encontrar un conjunto de parámetros epidemiológicos que produzcan una salida de simulación que minimice una métrica de error para la serie temporal de datos reales. Si bien el flujo de trabajo permite el uso de diferentes métricas de error y observables para el ajuste, aquí presentamos resultados preliminares utilizando el Error Cuadrático Medio (RMSE) como métrica de error y la serie temporal de muertes reportadas a nivel de país, como se hizo en las publicaciones originales. Para CMA-ES, utilizamos la implementación incluida en la biblioteca DEAP Python. Establecimos el tamaño de la población en 191 individuos para que se puedan asignar eficientemente a dos nodos informáticos con 96 CPU y dejamos una CPU libre para el script del flujo de trabajo. Además, establecimos un total de 25 generaciones para el algoritmo a fin de garantizar la convergencia. Para los parámetros restantes de CMA-ES, utilizamos los valores predeterminados.

**Tabla 1\.** Conjunto de parámetros seleccionados para probar el flujo de trabajo de calibración

| Parámetro | Valor de referencia  | Valor calibrado  | unidades  |
| :---- | :---- | :---- | :---- |
| βᴵ  | 0.075  | 0.06  | sin unidad |
| βᴬ  | 0.0375  | 0.0451  | sin unidad  |
| ηᵍ  | 2.444  | 1.474  | 1/​days​  |
| αᵍ  | 5.671; 2.756; 2.756  | 4.74; 2.74; 2.74  | 1/​days​  |
| δ  | 0.207  | 0.001174  | sin unidad  |
| ϕ  | 0.174  | 0.153146  | sin unidad |

Primero, nos centramos en recalibrar los parámetros de la publicación original. El conjunto incluye seis parámetros diferentes: cuatro epidemiológicos y dos relacionados con el distanciamiento social y el confinamiento. La Tabla 1 muestra los parámetros, incluyendo los valores reportados originalmente y los nuevos valores encontrados en el procedimiento de calibración. Algunos parámetros se mantienen cercanos a los valores reportados originalmente, mientras que otros muestran diferencias significativas. Estas discrepancias pueden deberse a diversos factores, como las diferencias en los datos de zonificación y movilidad utilizados, así como al conjunto de datos más completo de informes de COVID-19 que guía la búsqueda de parámetros.

| ​​![][image2]  Figura 1\. Resultados de la calibración de parámetros. El panel superior muestra la incidencia diaria, las hospitalizaciones y las muertes simuladas con el mejor conjunto de parámetros, junto con los datos reales reportados. El panel inferior muestra la convergencia algorítmica de CMA-ES para la minimización del RMSE entre los datos simulados y los informes sobre las tasas de mortalidad de España a nivel nacional.  Hemos observado que los parámetros óptimos muestran una fuerte concordancia en la predicción de hospitalizaciones y fallecimientos diarios. Sin embargo, existe una disparidad significativa en los resultados relativos al número de casos diarios. Nuestros hallazgos indican que los casos notificados en la primera ola fueron dos órdenes de magnitud inferiores a los predichos por las simulaciones. Evaluamos la convergencia del algoritmo CMA-ES y determinamos que normalmente converge a un RMSE mínimo en aproximadamente 15 generaciones. La Figura 1 ilustra una rápida disminución del valor de la función de coste durante las diez iteraciones iniciales, que se estabiliza, lo que sugiere convergencia. En consecuencia, calibrar los seis parámetros seleccionados requiere evaluar alrededor de 4000 simulaciones diferentes. |
| ----- |
| ​![][image3]​  |

**Figura 2\.** Distribución final de los parámetros aprendidos por el algoritmo CMA-ES.

Finalmente, comparamos la serie temporal de muertes producida por el mejor conjunto de parámetros con los datos reales. Sin embargo, en lugar de comparar a nivel de país, nos centramos en la predicción a nivel de comunidad autónoma. Los resultados se muestran en la Figura 3, que muestra cómo la primera ola se refleja muy bien en la mayoría de las comunidades, excepto en Castilla-La Mancha y Galicia. Curiosamente, la segunda ola también se recupera cualitativamente en varias comunidades. Observamos que las mayores discrepancias en la segunda ola se dan en los casos de las regiones correspondientes a las islas (Canarias y Baleares) y en las ciudades autónomas de Ceuta y Melilla. Nuestra hipótesis es que, en el primer caso, la discrepancia podría deberse a cambios en la red de movilidad que protege a la isla de la importación de casos de otras regiones. Actualmente, el simulador considera una red de movilidad estática y, por lo tanto, cualquier cambio en la estructura de la red de movilidad debido al comportamiento humano no es considerado por el modelo. En los casos de Ceuta y Melilla, observamos que los informes son ruidosos y probablemente estén sujetos a subregistro.

![][image4]

**Figura 3\.** Muertes diarias en España durante 2020, tanto a nivel nacional como por comunidades autónomas. Las líneas continuas y discontinuas corresponden a informes reales y datos simulados, respectivamente.

En conjunto, hemos probado el trabajo de calibración de parámetros epidemiológicos mediante optimización mediante simulación y hemos constatado que los parámetros candidatos se ajustan adecuadamente a las series temporales de fallecimientos y hospitalizaciones de la primera y la segunda ola a nivel nacional. Curiosamente, las simulaciones también se ajustan a las series temporales de varias comunidades autónomas. En trabajos futuros, utilizaremos análisis visual (T5.3) para evaluar la incertidumbre de las predicciones y aplicaremos enfoques de inferencia causal para comprender los patrones de propagación.

### Simulación de escenario con diferentes medidas de confinamiento {#simulación-de-escenario-con-diferentes-medidas-de-confinamiento}

El motor de simulación *MMCACovi19-vac[^1],* accesible a través de *EpiSim.jl*, permite modelar el efecto de la aplicación de medidas de confinamiento y vacunación. Las estrategias de confinamiento se basan en el distanciamiento social y el aislamiento de una parte de la población. Este enfoque disminuye el número promedio de contactos y la movilidad, lo que reduce la probabilidad de infección y, en última instancia, la prevalencia general de la enfermedad. Las políticas de confinamiento se introducen en el formalismo mediante diferentes parámetros descritos en la Tabla 2\.

**Tabla 2\.** Parámetros de reducción de la movilidad

| Symbol  | Description  |
| :---- | :---- |
| tᶜs  | Time steps when the containment measures will be applied  |
| κ₀s  | Reducción de la movilidad |
| δs  | Factor de distanciamiento social |
| ϕs  | Permeabilidad del hogar |

El parámetro κ₀g(t) representa la fracción de la población dentro del estrato de edad g que está bajo confinamiento en el tiempo t. El parámetro está acotado en el intervalo \[0,1\] y un κ₀g(t) igual a 0 o 1 representa ninguna restricción o un confinamiento total, respectivamente. Suponemos el mismo valor para todos los estratos de edad. El parámetro δ modela la reducción del número de contactos realizados por la población no confinada y también está acotado en el intervalo \[0,1\]. Finalmente, el parámetro de permeabilidad del hogar ϕ da cuenta de la mezcla social entre los miembros de diferentes hogares en aquellas situaciones en las que los miembros de un hogar determinado deben salir para actividades esenciales como comprar alimentos, medicamentos, etc y, por lo tanto, interactuar con miembros de diferentes hogares. Aunque las interacciones entre los miembros de diferentes hogares confinados influyen en el aislamiento del hogar, asumimos que no alteran significativamente el número promedio de contactos dentro de la población. Por lo tanto, en la implementación actual, la permeabilidad del hogar ϕ se mantiene constante a lo largo del tiempo.

![][image5]

**Figura 4\.** Muertes diarias en España durante 2020 para todo el país y a nivel de comunidades autónomas. Las líneas continuas y discontinuas corresponden a informes reales y datos simulados, respectivamente. El panel superior muestra los resultados de aplicar una política de reducción de la movilidad y el panel inferior sin ninguna medición aplicada.

Se realizaron dos simulaciones sin aplicar el NPI y con el parámetro que modela, el NPI aplicado en Spin durante la primera ola. La Figura 4 muestra los resultados de las dos simulaciones y cómo, en ausencia del NPI, el número de hospitalizaciones en el pico de la ola es casi el doble del valor obtenido cuando se aplicó el NPI. En cuanto a la mortalidad, la simulación predice que, en ausencia del NPI, el número total es de aproximadamente 100.000 personas, mientras que con el NPI, este número se reduce a aproximadamente 25.000. Estos resultados demuestran claramente que, en ausencia de vacunas, el NPI puede ser una buena estrategia para controlar las epidemias.

*EpiSim.jl* permite modelar el efecto de la vacunación a través del modelo matemático implementado en el motor *MMCACovi19-vac*. La vacunación se modela añadiendo dos compartimentos adicionales para cada estado del modelo. Estos nuevos compartimentos se utilizan para representar a los individuos que han sido vacunados y aquellos que fueron vacunados pero que han perdido parte de la immunidad. De esta manera, los parámetros epidemiológicos para los individuos vacunados son diferentes a los utilizados para las personas no vacunadas. Las estrategias de vacunación se introducen en el formalismo mediante diferentes parámetros descritos en la Tabla 3\.

Tabla 3: Parámetros que definen la estrategia de vacunación.

| Symbol  | Description  |
| :---- | :---- |
| ϵᵍ  | Vacunas totales por estrato de edad |
| tᵛs  | \[start\_vacc, dur\_vacc\]  |
| start\_vacc  | Inicio de la vacunación |
| dur\_vacc  | Duración de la vacunación |

Se realizó un exploración de estrategias mediante un barrido de parámetros variando la fecha de inicio, duración de la vacunación y prioridades de edad. Los resultados de la aplicación de cada estrategia se compararon con un escenario base sin vacunación. El objetivo consiste en identificar aquellas estrategias que minimicen dos objetivos: la variación relativa de la altura del segundo pico y la variación relativa en el número de muertes. Las estrategias que superaron a las demás en al menos una de estas medidas se encuentran en el frente de Pareto definido por los dos objetivos. En conjunto, los resultados muestran que, según nuestro modelo, las mejores estrategias reducen las muertes entre un 20 % y un 50 % y el segundo pico entre un 5 % y un 17 %. Se encontraron dos tipos de estrategias óptimas: la primera incluye estrategias que comenzaron temprano y fueron eficaces para reducir el número de muertes, mientras que la segunda comenzó más tarde y tuvo un mejor rendimiento para reducir el segundo pico. En ambos casos, el grupo de edad prioritario corresponde a los adultos mayores. Se obtuvieron resultados similares al intercambiar el pico del segundo pico por el pico de hospitalizaciones.

## Caso de Uso: Movilidad y Epidemiología de Aguas residuales {#caso-de-uso:-movilidad-y-epidemiología-de-aguas-residuales}

El objetivo de este caso de uso es aprovechar la complementariedad entre la movilidad humana y los biomarcadores de aguas residuales para anticipar
la evolución de casos a nivel municipal. Las matrices origen-destino MITMA se convierten en grafos temporales mediante MobilityDataLoader, que filtra por umbral de flujo, normaliza y mantiene la coherencia territorial a través del ZoneRegistry. Sobre cada corte temporal,
RegionDataProcessor agrega los flujos, la adyacencia geoespacial y los atributos demográficos en tensores listos para PyG.

La serie histórica de biomarcadores (N1, N2, IP4) se refina con EDARBiomarkerLoader, que incorpora interpolaciones, selección de variantes y cálculo
de caudales COVID cuando se activa el pipeline avanzado. Las proporciones EDAR–
municipio se codifican con EDARAttentionLoader, que genera máscaras normalizadas y puede extender la cobertura a todas las zonas del registro territorial
para mantener municipios “control” sin planta asignada.

En la etapa de modelado, DualGraphSAGE procesa por separado los grafos de movilidad y EDAR y los fusiona con atención informada por dominio y compuertas
adaptativas. El AttentionMaskProcessor aplica las máscaras de contribución y, cuando
hay datos suficientes, ajusta pesos aprendibles sobre las relaciones EDAR–municipio. El entrenador temporal alinea
secuencias de subgrafos de movilidad con snapshots de EDAR y aplica la máscara extendida cuando está disponible, evaluando MSE, MAE, RMSE, MAPE y R² para medir la ganancia predictiva frente a modelos monomodale.

Este flujo permite incorporar señales líderes de cargas virales para corregir la demora de los casos notificados, mejorar la sensibilidad en municipios
con escaso testeo y analizar el impacto diferencial de la movilidad según la cobertura de depuradoras. Las salidas incluyen métricas globales, gráficas de
residuos y mapas de atención que facilitan la interpretación de las rutas de propagación y la priorización de intervenciones.

Propuesta Conclusiones

- La integración dual de grafos captura simultáneamente rutas de contagio impulsadas por movilidad y señales tempranas derivadas de EDAR, proporcionando una
  base robusta para pronósticos municipale.
- El ecosistema de cargadores y mascarillas asegura coherencia territorial y manejo explícito de huecos de datos, habilitando experimentos comparables y
  escalables.
- Las rutinas de entrenamiento y evaluación ya contemplan métricas clásicas de vigilancia epidemiológica, lo que facilita su incorporación en cuadros de
  mando y procesos de alerta temprana.
- Próximos pasos:
  1) cerrar la calibración con series reales de casos y EDAR para cuantificar la ganancia frente a baselines;
  2) documentar escenarios de sensibilidad al umbral de flujo y a la cobertura EDAR;

# Conclusiones {#conclusiones}
