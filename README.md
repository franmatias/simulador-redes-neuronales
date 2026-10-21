# NeuroSim Edu

Simulador educativo e interactivo para explorar visualmente conceptos de redes neuronales. La aplicación se ejecuta en el navegador y está implementada con HTML, CSS y JavaScript, sin proceso de compilación ni dependencias que instalar.

## Contenido

El laboratorio contiene siete módulos:

1. **La neurona**: explora cómo las entradas, los pesos y el sesgo determinan la salida de una neurona.
2. **Desafío XOR**: visualiza un problema que no puede resolverse con una única frontera lineal.
3. **Backpropagation**: sigue paso a paso la propagación hacia delante y el ajuste de pesos.
4. **Sandbox**: experimenta con datos y entrenamiento, y observa el mapa de decisión y las métricas.
5. **Gradiente 3D**: explora la superficie de pérdida y el descenso del gradiente.
6. **Visión CNN**: interactúa con una cuadrícula y observa operaciones de convolución.
7. **Clasificador**: configura entre dos y cinco clases, entrena una salida Softmax y explora sus probabilidades y regiones de decisión.

Cada módulo incluye un panel pedagógico con tres pestañas: intuición y analogía, rigor matemático y reto práctico.

## Cómo usar el simulador

Selecciona uno de los siete módulos en la barra superior. Los controles correspondientes aparecerán en el panel izquierdo y la visualización interactiva en el lienzo principal. Los cambios en deslizadores, selectores y casillas se reflejan en la visualización al instante. En el panel inferior puedes alternar entre la explicación intuitiva, las fórmulas y una actividad práctica.

Usa el botón de la barra superior para alternar entre el modo oscuro y el modo claro; la preferencia se conserva en este navegador. En pantallas pequeñas, los controles y la visualización se apilan y la navegación de módulos puede desplazarse horizontalmente.

## Publicación en GitHub Pages

El flujo de GitHub Actions de `.github/workflows/deploy-pages.yml` publica automáticamente el sitio al hacer push a `main` y también se puede ejecutar manualmente desde la pestaña **Actions**. En el repositorio, selecciona **Settings > Pages > Build and deployment > Source > GitHub Actions**. Cuando termine correctamente el primer despliegue, el simulador estará disponible en:

<https://franmatias.github.io/simulador-redes-neuronales/>

### Controles comunes

- **Navegar por el lienzo:** usa la rueda del ratón para acercar o alejar. Los botones de la esquina inferior derecha permiten activar/desactivar el desplazamiento (pan), cambiar el zoom, restablecer la vista y alternar pantalla completa.
- **Desplazar la vista:** con el modo pan activo, arrastra el lienzo. El módulo 5 usa el arrastre izquierdo para rotar la superficie 3D; mantén `Shift` al arrastrar para desplazarla.
- **Conservar la vista:** al cambiar de módulo la cámara se restablece al encuadre inicial.

### 1. La neurona

Este módulo muestra una neurona con dos entradas. Su suma ponderada es `z = w₁x₁ + w₂x₂ + b`; la función de activación transforma `z` en la salida y la recta `z = 0` marca la frontera de decisión.

- **Función de activación:** Escalón, Sigmoide, `tanh` o ReLU. Permite comparar una salida binaria o continua y observar cómo cambia la respuesta.
- **Entradas `x₁` y `x₂`:** cada una va de −2 a 2, en pasos de 0,05. También puedes arrastrar el punto de entrada por el plano.
- **Pesos `w₁` y `w₂`:** de −4 a 4, en pasos de 0,1. Cambian la orientación de la frontera; los pesos negativos invierten el efecto de esa entrada.
- **Sesgo `b`:** de −4 a 4, en pasos de 0,1. Desplaza la frontera sin cambiar su orientación.
- **Inspector al pasar el cursor:** activa o desactiva la información contextual de la visualización.

### 2. Desafío XOR

La tabla muestra los cuatro casos de XOR: la salida es 1 cuando las entradas son distintas y 0 cuando son iguales. Ajusta los **pesos `w₁`, `w₂` y el sesgo `b`** (todos de −4 a 4, en pasos de 0,1) para mover la frontera lineal de una sola neurona e intentar clasificar los cuatro puntos. El indicador muestra los aciertos.

Pulsa **Añadir Capa Oculta (Resolver XOR)** para activar la solución multicapa ilustrada (arquitectura 2→2→1). **Volver a 1 Sola Neurona** regresa a la vista del perceptrón simple y permite continuar experimentando con una frontera lineal.

### 3. Retropropagación (Backpropagation)

Observa el ciclo de una red pequeña: propagación hacia delante (*forward*), cálculo del error (*loss*), cálculo de gradientes (*backward*) y actualización de pesos. El reproductor superpuesto al lienzo permite:

- **Reiniciar:** restablece los pesos iniciales y el contador de épocas.
- **Fase anterior / siguiente fase:** recorre el ciclo paso a paso.
- **Reproducir / pausar:** anima las fases continuamente.
- **+15 épocas:** aplica quince actualizaciones completas rápidamente.

En **Configuración del caso**, `x₁` y `x₂` se ajustan de 0 a 1 en pasos de 0,1; el **valor objetivo `y`** va de 0,05 a 0,95 en pasos de 0,05; y la **tasa de aprendizaje `η`** va de 0,1 a 2,0 en pasos de 0,1. Un valor mayor de `η` produce actualizaciones más grandes. Las entradas y el objetivo recalculan la predicción y el error para el caso actual.

### 4. Sandbox multicapa

Entrena una red para separar puntos en un mapa de decisión. Puedes elegir uno de nueve conjuntos: Dos Lunas, Círculos Concéntricos, Espiral Doble, Tablero 3×3, Triple Anillo, Frontera Sinusoidal, Cuatro Islas, Cuadrantes XOR y Grupos Lineales.

- **Ruido:** de 0 a 40 %, en pasos de 2 %. Modifica la dispersión de los datos; al moverlo se genera una muestra nueva del conjunto seleccionado.
- **Batch / Frame:** de 4 a 40, en pasos de 4. Controla cuántas épocas de entrenamiento se avanzan por fotograma (aproximadamente la mitad del valor, con un mínimo de dos); no es el tamaño de lote matemático usado para calcular gradientes.
- **Arquitectura de capas ocultas:** elige 1 capa de 3 o 6 neuronas; 2 capas de 5→4 o 6→6; o 3 capas de 7→6→4 u 8→8→6.
- **Activación de capas ocultas:** `tanh`, ReLU, Leaky ReLU, seno (`sin`) o Sigmoide. Cambiarla reinicia los pesos de la red.
- **Tasa de aprendizaje `η`:** de 0,01 a 0,50, en pasos de 0,01. Ajusta el tamaño de cada actualización.
- **Regularización L2 `λ`:** de 0 a 0,010, en pasos de 0,001. Penaliza pesos grandes para ayudar a limitar el sobreajuste.
- **Frontera p=0,5 / Pulsos sinápticos:** casillas para mostrar u ocultar la frontera de clasificación y la animación de las conexiones.
- **Añadir datos propios:** selecciona Azul (1) o Rojo (0) y haz clic en el mapa. El botón de escoba regenera el conjunto elegido y elimina los puntos añadidos manualmente.
- **Entrenar / pausar:** inicia o detiene el entrenamiento continuo. **+1 Época** avanza una época manualmente; el botón de reinicio inicializa de nuevo los pesos de la arquitectura seleccionada.

El panel **Telemetría en Vivo** informa de las épocas, la precisión, la pérdida MSE y su evolución.

### 5. Descenso del gradiente 3D

La superficie representa una función de pérdida `L(w₁,w₂)` y las bolas muestran la trayectoria de los optimizadores hacia regiones de menor pérdida.

- **Topografía:** elige mínimos locales (no convexa), cuenco convexo, punto de silla, cañón estrecho o valle de Rosenbrock.
- **Optimizador:** compara SGD, Momentum y Adam a la vez, o muestra individualmente uno de ellos.
- **Learning Rate:** de 0,01 a 0,20, en pasos de 0,01. Controla el tamaño de los pasos del descenso.
- **Inercia `β`:** de 0,50 a 0,98, en pasos de 0,02. Controla la contribución de la velocidad acumulada en Momentum.
- **Base de Contorno / Vector `−∇L`:** muestra u oculta las curvas de nivel y la dirección opuesta al gradiente.
- **Punto Inicial A / B:** reinicia la trayectoria desde uno de dos puntos predefinidos. También puedes hacer clic en la superficie para elegir una posición inicial.
- **Descender / Pausar, +1 Paso y Reiniciar:** ejecuta el descenso continuamente, avanza un paso o restablece la trayectoria. El panel muestra el paso, la posición, la pérdida y la norma del gradiente.

Puedes rotar la superficie arrastrando el lienzo; `Shift` + arrastrar desplaza la cámara.

### 6. Visión artificial (CNN)

El módulo aplica una convolución a una cuadrícula de entrada de 10×10 usando un kernel de 3×3, produce un mapa de activación de 8×8 y reduce ese mapa a 4×4 mediante Max-Pooling 2×2 con paso 2.

- **Patrón de entrada:** carga el dígito 7, una cruz, un contorno cuadrado o una diagonal. También puedes dibujar en la cuadrícula de entrada; el borrador limpia el lienzo.
- **Filtro convolucional:** selecciona Sobel X (bordes verticales), Sobel Y (horizontales), Laplaciano (contornos), Sharpen (nitidez) o Emboss (relieve).
- **Matriz de pesos del kernel:** edita individualmente sus nueve valores; los mapas se recalculan al cambiar un valor.
- **ReLU:** activa o desactiva la transformación `max(0,z)` aplicada a la salida de la convolución.
- **Escaneo automático:** activa el recorrido automático de la ventana 3×3 por el mapa. Los botones **Anterior** y **Siguiente** avanzan manualmente y desactivan el modo automático; haz clic en una celda del mapa 8×8 para inspeccionar su cálculo.

### 7. Clasificador multiclase

Este módulo clasifica dos características de entrada con una red neuronal configurable: una a tres capas ocultas ReLU y una neurona de salida por clase. Softmax convierte las puntuaciones de salida en probabilidades que suman 100 %, y la clase con mayor probabilidad es la predicción.

- **Neuronas de salida / clases:** selecciona de dos a cinco categorías para cambiar el tamaño de la capa de salida.
- **Dataset y ruido:** elige nubes, círculos concéntricos o espirales; ajusta la dispersión con el control de ruido. El dataset manual permite crear ejemplos etiquetados pulsando el mapa.
- **Modo de interacción:** **Probar entrada** mueve el marcador de consulta; **Añadir punto** coloca un ejemplo de la clase seleccionada. Puedes borrar los ejemplos para crear un dataset propio desde cero.
- **Arquitectura:** configura de una a tres capas ocultas y de cuatro a doce neuronas por capa. La visualización muestra las conexiones, la activación ReLU oculta y las salidas Softmax.
- **Algoritmo de entrenamiento:** Batch GD actualiza usando el dataset completo; SGD actualiza tras cada ejemplo; Mini-batch actualiza en grupos de 16. Los tres usan retropropagación y entropía cruzada.
- **Entrenar clasificador / Reiniciar pesos:** ajusta los parámetros sobre los ejemplos actuales o inicializa pesos nuevos conservando los datos.
- **Mapa de decisión:** los colores indican la clase predicha en cada región; los círculos son ejemplos sintéticos, los cuadrados son puntos manuales y el marcador grande es la entrada de consulta.

## Requisitos

- Un navegador moderno con soporte para Canvas y JavaScript.
- Conexión a Internet para cargar Tailwind CSS, Font Awesome y las fuentes externas utilizadas por la interfaz.

## Cómo ejecutarlo

Al ser una aplicación estática, no necesita instalación. Sirve la carpeta del proyecto desde un servidor HTTP local para que se carguen correctamente el HTML, los estilos y el JavaScript.

Desde PowerShell, en la carpeta del proyecto:

```powershell
py -m http.server 8000
```

Después abre [http://localhost:8000/](http://localhost:8000/). También puedes usar una extensión como Live Server en Visual Studio Code, asegurándote de servir la raíz del proyecto.

## Estructura

```text
simulador-redes-neuronales/
├── index.html            # Interfaz y lienzos de visualización
├── css/
│   └── styles.css        # Estilos propios
└── js/
    └── app.js            # Simulación, controles y animaciones
```

## Desarrollo y pruebas

No hay un gestor de paquetes, paso de compilación ni conjunto de pruebas automatizadas configurados. Para comprobar cambios, sirve la carpeta localmente y prueba los módulos y sus controles en el navegador.
