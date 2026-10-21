// ==========================================
// REFERENCIAS DEL DOM Y LIENZOS
// ==========================================
const canvas = document.getElementById("simCanvas");
const ctx = canvas.getContext("2d");
const lossChartCanvas = document.getElementById("lossChartCanvas");
const lossChartCtx = lossChartCanvas ? lossChartCanvas.getContext("2d") : null;
let width = 800,
  height = 600;

// Lienzo Offscreen para el Heatmap de Alta Resolución (Módulos 2 y 4)
const heatmapCanvas = document.createElement("canvas");
const heatCtx = heatmapCanvas.getContext("2d");
const heatRes = 75;
heatmapCanvas.width = heatRes;
heatmapCanvas.height = heatRes;
const heatImgData = heatCtx.createImageData(heatRes, heatRes);

// ==========================================
// SISTEMA UNIVERSAL DE CÁMARA (PAN & ZOOM)
// ==========================================
const camera = {
  x: 0, // Desplazamiento Pan X (en píxeles de pantalla respecto al centro)
  y: 0, // Desplazamiento Pan Y
  zoom: 1, // Nivel de Zoom actual
  targetZoom: 1, // Para interpolación suave
  targetX: 0,
  targetY: 0,
  isDragging: false,
  dragStartX: 0,
  dragStartY: 0,
  camStartX: 0,
  camStartY: 0,
  panEnabled: true,
};

// Coordenadas del cursor
let mouseScreenX = 0,
  mouseScreenY = 0;
let mouseWorldX = 0,
  mouseWorldY = 0;
let isDraggingInputPoint = false; // Arrastrar el punto (x1,x2) en el Módulo 1

// Transformar de Pantalla -> Mundo Lógico
function screenToWorld(sx, sy) {
  const cx = width / 2;
  const cy = height / 2;
  return {
    x: (sx - cx - camera.x) / camera.zoom + cx,
    y: (sy - cy - camera.y) / camera.zoom + cy,
  };
}

// Transformar de Mundo Lógico -> Pantalla
function worldToScreen(wx, wy) {
  const cx = width / 2;
  const cy = height / 2;
  return {
    x: (wx - cx) * camera.zoom + cx + camera.x,
    y: (wy - cy) * camera.zoom + cy + camera.y,
  };
}

function updateZoomUI() {
  const pct = Math.round(camera.targetZoom * 100);
  const lbl = document.getElementById("zoom-pct-label");
  if (lbl) lbl.innerText = `${pct}%`;
}

function cameraZoomStep(
  delta,
  pivotScreenX = width / 2,
  pivotScreenY = height / 2,
) {
  const oldZoom = camera.targetZoom;
  const newZoom = Math.min(3.5, Math.max(0.35, +(oldZoom + delta).toFixed(2)));
  if (newZoom === oldZoom) return;

  // Mantener el punto bajo el cursor fijo durante el zoom
  const cx = width / 2;
  const cy = height / 2;
  const worldPtX = (pivotScreenX - cx - camera.targetX) / oldZoom + cx;
  const worldPtY = (pivotScreenY - cy - camera.targetY) / oldZoom + cy;

  camera.targetZoom = newZoom;
  camera.targetX = pivotScreenX - cx - (worldPtX - cx) * newZoom;
  camera.targetY = pivotScreenY - cy - (worldPtY - cy) * newZoom;

  updateZoomUI();
}

function resetCamera() {
  camera.targetZoom = 1;
  camera.targetX = 0;
  camera.targetY = 0;
  updateZoomUI();
}

function togglePanLock() {
  camera.panEnabled = !camera.panEnabled;
  const btn = document.getElementById("btn-pan-mode");
  if (camera.panEnabled) {
    btn.classList.add("active-tool");
    canvas.style.cursor = "grab";
  } else {
    btn.classList.remove("active-tool");
    canvas.style.cursor = "default";
  }
}

function applyTheme(theme, persist = true) {
  const isLight = theme === "light";
  const toggle = document.getElementById("theme-toggle");
  const icon = document.getElementById("theme-toggle-icon");
  const label = document.getElementById("theme-toggle-label");

  document.body.classList.toggle("light-mode", isLight);
  if (toggle) {
    const nextThemeLabel = isLight ? "modo oscuro" : "modo claro";
    toggle.setAttribute("aria-label", `Cambiar a ${nextThemeLabel}`);
    toggle.setAttribute("title", `Cambiar a ${nextThemeLabel}`);
  }
  if (icon) icon.className = `fas ${isLight ? "fa-moon" : "fa-sun"}`;
  if (label) label.textContent = isLight ? "Modo oscuro" : "Modo claro";
  if (persist) localStorage.setItem("neurosim-theme", isLight ? "light" : "dark");
}

function toggleTheme() {
  applyTheme(document.body.classList.contains("light-mode") ? "dark" : "light");
}

// ==========================================
// ESTADO GLOBAL DEL SIMULADOR
// ==========================================
const state = {
  module: 1,
  time: 0,
  // Módulo 1 y 2
  m1: {
    w1: 1.2,
    w2: 1.0,
    b: -0.5,
    x1: 0.8,
    x2: 0.8,
    actFn: "step", // 'step', 'sigmoid', 'relu', 'tanh'
    solvedXor: false,
    showHelp: true,
  },
  // Módulo 3 (Backpropagation)
  m3: {
    runMode: "paused",
    phase: 0,
    progress: 0,
    waitFrames: 0,
    epoch: 0,
    lr: 0.8,
    x: [0.8, 0.2],
    target: 1.0,
    w_i_h: [
      [0.5, -0.3],
      [0.8, 0.2],
    ],
    b_h: [0.1, -0.1],
    w_h_o: [0.9, -0.6],
    b_o: -0.2,
    prevWeights: null,
    lossHistory: [],
    fwd: null,
    bwd: null,
  },
  // Módulo 4 (Sandbox)
  m4: {
    lr: 0.12,
    hidden: [5, 4],
    dataset: "moons",
    isTraining: false,
    epoch: 0,
    loss: 0,
    accuracy: 0,
    noise: 0.08,
    activation: "tanh",
    regL2: 0.0,
    batchSize: 16,
    showContour: true,
    showWeights: true,
    addClassMode: 1, // 1 = Clase Azul, 0 = Clase Roja al hacer clic
    lossHistory: [],
  },
  // Módulo 5 (Descenso del Gradiente 3D)
  m5: {
    surface: "local_minima", // 'bowl', 'saddle', 'local_minima', 'ravine', 'rosenbrock'
    optimizer: "sgd", // 'sgd', 'momentum', 'adam', 'compare_all'
    lr: 0.04,
    momentum: 0.88,
    isRunning: false,
    stepCount: 0,
    rotYaw: -0.65,
    rotPitch: 0.62,
    showWireframe: true,
    showContours: true,
    showGradientVec: true,
    balls: [],
  },
  // Módulo 6 (Redes Convolucionales CNN y Visión Artificial)
  m6: {
    presetImg: "digit7",
    kernelType: "edge_vert",
    kernel: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
    useRelu: true,
    scanX: 0,
    scanY: 0,
    autoScan: true,
    paintMode: 1, // 1 = Pintar blanco, 0 = Borrar
    isPainting: false,
    inputGrid: [], // 10x10
    convGrid: [], // 8x8
    poolGrid: [], // 4x4
  },
  // Módulo 7 (Clasificador multiclase)
  m7: {
    classCount: 3,
    x1: 0.35,
    x2: 0.15,
    datasetType: "blobs",
    noise: 0.12,
    hiddenLayers: 2,
    neuronsPerLayer: 8,
    optimizer: "minibatch",
    manualClass: 0,
    interactionMode: "probe",
    weights: [],
    networkBiases: [],
    dataset: [],
    trained: false,
    loss: null,
  },
};

// ==========================================
// SISTEMA DE EXPLICACIONES PEDAGÓGICAS POR PESTAÑAS
// ==========================================
let currentExplTab = "intuition";

const moduleExplanations = {
  1: {
    intuition: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">🧠 La Neurona como un Juez:</span> Imagina que la neurona decide si apruebas una asignatura basándose en dos notas: <span class="font-mono text-blue-300">x₁</span> (Examen) y <span class="font-mono text-blue-300">x₂</span> (Proyecto).
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">⚖️ Pesos Sinápticos (w₁, w₂):</span> Son la <em>importancia</em> que el juez da a cada dato. Si <span class="font-mono text-emerald-300">w₁ = 3</span> y <span class="font-mono text-emerald-300">w₂ = 0</span>, solo importa el examen y la frontera se vuelve totalmente vertical. Un peso negativo invierte la preferencia.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">🎚️ Sesgo o Bias (b):</span> Es cuán estricto o generoso es el juez de partida. Mover <span class="font-mono text-amber-300">b</span> traslada la línea recta sin cambiar su inclinación.
                        </div>
                    </div>`,
    math: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">1. Pre-activación Afín (z):</span> Es el producto escalar entre el vector de pesos y el de entrada más el sesgo: <span class="font-mono text-white">z = w₁·x₁ + w₂·x₂ + b</span>. Mide la distancia con signo desde el punto hasta la recta.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">2. Hiperplano Separador:</span> La frontera de decisión ocurre exactamente donde <span class="font-mono text-white">z = 0</span>. Despejando <span class="font-mono">x₂</span> obtenemos la recta <span class="font-mono text-blue-300">x₂ = -(w₁/w₂)x₁ - (b/w₂)</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">3. Vector Normal (w⃗):</span> El vector <span class="font-mono text-emerald-300">w⃗ = (w₁, w₂)</span> dibujado en verde es siempre ortogonal (perpendicular a 90°) a la línea de frontera y apunta hacia la región donde <span class="font-mono">z > 0</span>.
                        </div>
                    </div>`,
    lab: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Experimento 1 (Rotación):</span> Pon <span class="font-mono">b = 0</span> y varía <span class="font-mono">w₁</span> y <span class="font-mono">w₂</span>. Verás que la recta gira como una hélice anclada en el origen <span class="font-mono">(0,0)</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Experimento 2 (Traslación):</span> Deja fijos los pesos y mueve únicamente el sesgo <span class="font-mono">b</span>. Observa cómo la recta se desliza paralelamente a sí misma.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Experimento 3 (Suavizado):</span> Cambia la función de activación de <em>Escalón</em> a <em>Sigmoide</em> y arrastra el punto <span class="font-mono">(x₁, x₂)</span> en el gráfico para ver la transición gradual de probabilidad.
                        </div>
                    </div>`,
  },
  2: {
    intuition: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-rose-400 font-bold">🚧 El Invierno de la IA (1969):</span> Minsky y Papert demostraron que un perceptrón simple es incapaz de aprender la puerta lógica <strong>XOR</strong> ("O exclusivo": azul si las entradas son distintas, rojo si son iguales).
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">✂️ El Límite de un Solo Corte:</span> Intentar separar las esquinas opuestas de un cuadrado con una sola línea recta es geométricamente imposible: como máximo acertarás <strong>3 de 4 puntos (75%)</strong>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">✨ La Solución Multicapa:</span> Al añadir una <strong>Capa Oculta</strong> de 2 neuronas, la primera traza un corte superior y la segunda un corte inferior, encerrando los puntos azules en un pasillo central.
                        </div>
                    </div>`,
    math: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">1. Neurona Oculta h₁ (Puerta OR):</span> Activa toda la región donde <span class="font-mono text-white">x₁ + x₂ > 0.45</span>, dejando fuera únicamente al punto <span class="font-mono">(0,0)</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">2. Neurona Oculta h₂ (Umbral Alto):</span> Se activa solo cuando <span class="font-mono text-white">x₁ + x₂ > 1.55</span>, aislando a la esquina superior <span class="font-mono">(1,1)</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">3. Combinación en Salida:</span> La neurona final calcula <span class="font-mono text-emerald-300">y = σ(h₁ - h₂)</span>: resta ambas regiones y deja activa exclusivamente la banda diagonal donde viven <span class="font-mono">(0,1)</span> y <span class="font-mono">(1,0)</span>.
                        </div>
                    </div>`,
    lab: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Paso 1:</span> Mueve los deslizadores <span class="font-mono">w₁, w₂, b</span> intentando poner el marcador en <em>4/4 Aciertos</em>. Comprobarás que al capturar un punto azul pierdes el otro.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">Paso 2:</span> Pulsa el botón morado <strong>Añadir Capa Oculta (Resolver XOR)</strong> y observa cómo aparecen las dos rectas discontinuas <span class="font-mono">h₁</span> y <span class="font-mono">h₂</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Conclusión Clave:</span> Las capas ocultas transforman problemas linealmente inseparables en representaciones intermedias separables.
                        </div>
                    </div>`,
  },
  3: {
    intuition: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">1. Disparar al Blanco (Forward):</span> La señal viaja de izquierda a derecha produciendo una predicción inicial <span class="font-mono text-blue-300">a</span>. Al principio los pesos no están ajustados y fallamos el tiro.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-rose-400 font-bold">2. Repartir la Culpa (Backward):</span> El algoritmo mide el error en la salida y viaja hacia atrás preguntando a cada peso: <em>"¿Cuánto contribuiste tú a este fallo?"</em>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">3. Corregir la Puntería (Update):</span> Cada peso se corrige un pequeño porcentaje (<span class="font-mono text-amber-300">Learning Rate η</span>) en dirección opuesta a su error.
                        </div>
                    </div>`,
    math: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Error Cuadrático:</span> <span class="font-mono text-white">L = ½(Target - a_o)²</span>. Su derivada respecto a la activación de salida es <span class="font-mono text-amber-300">∂L/∂a_o = (a_o - Target)</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-rose-400 font-bold">Regla de la Cadena:</span> Para un peso de salida, <span class="font-mono text-white">∂L/∂w = (∂L/∂a_o)·(∂a_o/∂z_o)·(∂z_o/∂w) = δ_o · a_h</span>, donde <span class="font-mono">σ'(z) = a(1-a)</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Regla Delta Oculta:</span> El error se propaga a la capa oculta ponderado por los pesos que las conectan: <span class="font-mono text-emerald-300">δ_h = (δ_o · w_ho) · a_h(1 - a_h)</span>.
                        </div>
                    </div>`,
    lab: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Inspección Fina:</span> Pulsa el botón de <strong>Siguiente Fase (⏭)</strong> en el reproductor superior y observa cómo en la Fase 3 aparecen las etiquetas rojas <span class="font-mono">∂L/∂w</span> en cada conexión.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Convergencia Rápida:</span> Pulsa <strong>+15 Épocas</strong> un par de veces y mira cómo la gráfica de <em>Loss</em> cae en picado hacia 0.0000 mientras la predicción clava el Objetivo.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Cambio de Objetivo:</span> Cambia el deslizador <em>Valor Objetivo (Target y)</em> de 1.0 a 0.1 y observa cómo los gradientes cambian de signo inmediatamente.
                        </div>
                    </div>`,
  },
  4: {
    intuition: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">🎨 Plegando el Lienzo:</span> Cada neurona oculta añade un "pliegue" al espacio 2D. Con más neuronas y capas, la red puede esculpir islas, anillos o espirales complejas.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">⚡ El Papel de la Activación:</span> <em>ReLU</em> construye fronteras con recortes rectos poligonales, <em>Tanh</em> crea curvas suaves y <em>Seno sin(z)</em> genera ondas periódicas repetitivas.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">🛡️ Sobreajuste (Overfitting):</span> Si hay mucho ruido y una red gigante, la frontera intentará rodear cada punto rebelde. La <strong>Regularización L2</strong> penaliza pesos extremos para mantener fronteras limpias.
                        </div>
                    </div>`,
    math: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Composición Funcional:</span> Una red de 2 capas ocultas calcula <span class="font-mono text-white">ŷ = σ(W₃ · f(W₂ · f(W₁x + b₁) + b₂) + b₃)</span>, un aproximador universal de funciones.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Penalización L2 (Weight Decay):</span> Añade <span class="font-mono text-white">(λ/2)Σw²</span> a la función de coste. Su derivada resta <span class="font-mono text-amber-300">λ·w</span> en cada actualización, evitando pesos desbocados.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Contorno de Decisión:</span> La línea blanca punteada sobre el mapa representa el conjunto de nivel <span class="font-mono text-white">{x ∈ ℝ² : ŷ(x) = 0.5}</span> donde ambas clases son equiprobables.
                        </div>
                    </div>`,
    lab: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Reto 1 (Geometría ReLU vs Tanh):</span> Elige <em>Círculos Concéntricos</em> con 1 Capa (4 Neuronas). Entrena con <strong>ReLU</strong> (verás un cuadrado/polígono) y luego con <strong>Tanh</strong> (verás un círculo suave).
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">Reto 2 (Tablero 3x3):</span> Selecciona <em>Tablero de Ajedrez 3x3</em>, pon arquitectura <em>2 Capas (6→6)</em> y activación <strong>Seno sin(z)</strong>. ¡Verás cómo resuelve las 9 casillas en segundos!
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Reto 3 (Datos en Vivo):</span> Mientras entrena, haz clic directamente dentro del mapa de calor para inyectar puntos azules o rojos y obligar a la frontera a adaptarse en vivo.
                        </div>
                    </div>`,
  },
  5: {
    intuition: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">🏔️ El Alpinista en la Niebla:</span> Entrenar una red es como bajar de una montaña a ciegas sintiendo solo la inclinación bajo tus pies y dando pasos cuesta abajo.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">🎱 El Poder de la Inercia (Momentum):</span> Una bola pesada acumula velocidad al bajar. Si encuentra un pequeño bache (<em>mínimo local</em>) o una meseta plana (<em>punto de silla</em>), su inercia la ayuda a cruzarlo.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">🧭 Adam (Paso Adaptativo):</span> Ajusta automáticamente el tamaño de zancada para cada peso: frena en pendientes bruscas y acelera en llanuras suaves.
                        </div>
                    </div>`,
    math: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-sky-400 font-bold">SGD Estándar:</span> Actualiza sin memoria: <span class="font-mono text-white">w_{t+1} = w_t - η·∇L(w_t)</span>. Si el gradiente se anula (<span class="font-mono">∇L = 0</span>), se detiene por completo.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">SGD + Momentum:</span> Acumula velocidad exponencial: <span class="font-mono text-white">v_t = β·v_{t-1} - η·∇L</span> y actualiza <span class="font-mono">w_{t+1} = w_t + v_t</span>. Amortigua oscilaciones en cañones estrechos.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Optimizador Adam:</span> Combina el primer momento <span class="font-mono">m_t</span> (media del gradiente) y el segundo momento <span class="font-mono">r_t</span> (varianza no centrada): <span class="font-mono text-emerald-300">Δw = -η · m̂_t / (√r̂_t + ε)</span>.
                        </div>
                    </div>`,
    lab: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Experimento 1 (Trampa Local):</span> En <em>Mínimos Locales y Global</em>, pulsa <strong>Punto Inicial A</strong> y luego <strong>Descender</strong>. Verás cómo SGD (azul) queda atrapado arriba mientras Momentum y Adam llegan al valle global.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Experimento 2 (Cañón Estrecho):</span> Cambia la topografía a <em>Cañón Estrecho</em> y sube el <em>Learning Rate</em> a 0.14. Observa cómo SGD rebota violentamente de pared a pared mientras Adam va directo al centro.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Experimento 3 (Lanzamiento Libre):</span> Haz clic sobre cualquier cresta de la montaña 3D para soltar las tres bolas desde ese punto exacto.
                        </div>
                    </div>`,
  },
  6: {
    intuition: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">🔦 La Linterna Detectora (Kernel):</span> En lugar de conectar todos los píxeles con todos, una CNN desliza una pequeña lupa de <span class="font-mono">3×3</span> llamada <strong>Filtro o Kernel</strong> que busca un patrón concreto (ej. bordes verticales u horizontales).
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">🗺️ Mapa de Características:</span> Donde el dibujo coincide con la forma del filtro, el mapa se ilumina intensamente. Así la red descubre trazos, esquinas y texturas.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">📉 Max-Pooling (2×2):</span> Resume cada bloque de <span class="font-mono">2×2</span> quedándose solo con el valor máximo. Reduce el tamaño a la mitad y hace que la red reconozca el objeto aunque se mueva un poco.
                        </div>
                    </div>`,
    math: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">1. Convolución Discreta 2D:</span> En cada posición <span class="font-mono">(i, j)</span> se calcula el producto escalar Frobenius: <span class="font-mono text-white">S(i,j) = Σ_{m=0..2} Σ_{n=0..2} I(i+m, j+n) · K(m, n)</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">2. Rectificación ReLU:</span> Aplicamos <span class="font-mono text-white">F(i,j) = max(0, S(i,j))</span> para eliminar respuestas negativas (inhibiciones) y conservar solo las detecciones positivas del patrón.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">3. Submuestreo Max-Pool:</span> Con ventana <span class="font-mono">2×2</span> y paso (stride) 2: <span class="font-mono text-emerald-300">P(r,c) = max_{u,v ∈ {0,1}} F(2r+u, 2c+v)</span>. Reduce de <span class="font-mono">8×8</span> a <span class="font-mono">4×4</span>.
                        </div>
                    </div>`,
    lab: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Experimento 1 (Bordes Verticales vs Horizontales):</span> Con la figura <em>Cruz (+)</em> seleccionada, alterna entre el filtro <strong>Sobel Vertical</strong> y <strong>Sobel Horizontal</strong>. ¡Mira cómo desaparece un brazo de la cruz y se ilumina el otro!
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Experimento 2 (Dibuja a Mano):</span> Pinta directamente con el ratón sobre la cuadrícula de <em>Imagen de Entrada (10×10)</em> y observa cómo se recalculan todas las capas al instante.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Experimento 3 (Inspección Celda a Celda):</span> Pausa el escaneo automático y haz clic en cualquier celda del <em>Mapa de Características (8×8)</em> para ver exactamente qué región <span class="font-mono">3×3</span> la generó.
                        </div>
                    </div>`,
  },
  7: {
    intuition: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">🎯 Una salida por clase:</span> Cada neurona de salida representa una categoría. Las capas ocultas aprenden combinaciones de características para separar patrones complejos.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">📊 Probabilidades que compiten:</span> Softmax convierte las puntuaciones en probabilidades que suman 100 %. La clase predicha es la que obtiene la probabilidad más alta.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">🗺️ Frontera de decisión:</span> Elige entre blobs, círculos o espirales, añade ruido y coloca ejemplos etiquetados a mano para explorar cómo aprende el modelo.
                        </div>
                    </div>`,
    math: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">1. Puntuación (logit):</span> Para cada clase <span class="font-mono">k</span>, la red calcula <span class="font-mono text-white">zₖ = wₖ₁x₁ + wₖ₂x₂ + bₖ</span>.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-purple-400 font-bold">2. Softmax:</span> <span class="font-mono text-white">P(k|x) = exp(zₖ) / Σⱼ exp(zⱼ)</span>. Normaliza los logits: cada probabilidad está entre 0 y 1 y todas suman 1.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">3. Aprendizaje:</span> La retropropagación calcula el gradiente de la entropía cruzada a través de cada capa. Batch, SGD y mini-batch se diferencian en cuántos ejemplos usan por actualización.
                        </div>
                    </div>`,
    lab: `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Reto 1 (Más categorías):</span> Cambia de dos a cinco clases y observa cómo aparece una neurona de salida por categoría.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Reto 2 (No linealidad):</span> Prueba círculos o espirales. Compara una capa oculta con varias y observa cómo cambia la frontera.
                        </div>
                        <div class="bg-slate-950/80 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Reto 3 (Algoritmos):</span> Entrena el mismo dataset con Batch GD, SGD y Mini-batch y compara el resultado. Después coloca algunos puntos manualmente.
                        </div>
                    </div>`,
  },
};

function setExplanationTab(tabKey) {
  currentExplTab = tabKey;
  const tabs = ["intuition", "math", "lab"];
  tabs.forEach((t) => {
    const btn = document.getElementById(`tab-expl-${t}`);
    if (!btn) return;
    if (t === tabKey) {
      btn.className =
        "px-2.5 py-1 rounded-md font-semibold bg-blue-600 text-white transition";
    } else {
      btn.className =
        "px-2.5 py-1 rounded-md font-semibold text-slate-400 hover:text-white transition";
    }
  });
  renderModuleExplanation();
}

function renderModuleExplanation() {
  const explEl = document.getElementById("explanation-content");
  if (!explEl) return;
  const modData = moduleExplanations[state.module];
  if (modData && modData[currentExplTab]) {
    explEl.innerHTML = modData[currentExplTab];
  }
}

// ==========================================
// LÓGICA MATEMÁTICA MÓDULO 6 (VISIÓN CNN)
// ==========================================
const cnnPresets = {
  digit7: [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  cross: [
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
  ],
  box: [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
  ],
  diag: [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
  ],
};

const cnnKernels = {
  edge_vert: [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1],
  ],
  edge_horiz: [
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1],
  ],
  laplacian: [
    [0, -1, 0],
    [-1, 4, -1],
    [0, -1, 0],
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0],
  ],
  emboss: [
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2],
  ],
};

function loadCNNPreset(name) {
  state.m6.presetImg = name;
  const src = cnnPresets[name] || cnnPresets.digit7;
  state.m6.inputGrid = src.map((row) => [...row]);
  recomputeCNN();
}

function clearCNNCanvas() {
  state.m6.inputGrid = Array.from({ length: 10 }, () => Array(10).fill(0));
  recomputeCNN();
}

function setCNNKernel(type) {
  state.m6.kernelType = type;
  const k = cnnKernels[type] || cnnKernels.edge_vert;
  state.m6.kernel = k.map((row) => [...row]);
  syncKernelInputsUI();
  recomputeCNN();
}

function syncKernelInputsUI() {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      const el = document.getElementById(`k-cell-${r}-${c}`);
      if (el) el.value = state.m6.kernel[r][c];
    }
  }
}

function updateKernelCell(r, c, val) {
  const num = parseFloat(val);
  state.m6.kernel[r][c] = isNaN(num) ? 0 : num;
  recomputeCNN();
}

function recomputeCNN() {
  const m = state.m6;
  if (!m.inputGrid.length) return;

  // 1. Convolución 10x10 con Kernel 3x3 -> Mapa 8x8
  m.convGrid = [];
  for (let r = 0; r < 8; r++) {
    const row = [];
    for (let c = 0; c < 8; c++) {
      let sum = 0;
      for (let kr = 0; kr < 3; kr++) {
        for (let kc = 0; kc < 3; kc++) {
          sum += m.inputGrid[r + kr][c + kc] * m.kernel[kr][kc];
        }
      }
      row.push(m.useRelu ? Math.max(0, sum) : sum);
    }
    m.convGrid.push(row);
  }

  // 2. Max-Pooling 2x2 (stride 2) -> Mapa 4x4
  m.poolGrid = [];
  for (let pr = 0; pr < 4; pr++) {
    const row = [];
    for (let pc = 0; pc < 4; pc++) {
      const v1 = m.convGrid[pr * 2][pc * 2];
      const v2 = m.convGrid[pr * 2][pc * 2 + 1];
      const v3 = m.convGrid[pr * 2 + 1][pc * 2];
      const v4 = m.convGrid[pr * 2 + 1][pc * 2 + 1];
      row.push(Math.max(v1, v2, v3, v4));
    }
    m.poolGrid.push(row);
  }
}

function getCNNLayoutGeometry() {
  const cellIn = Math.min(28, Math.max(20, Math.floor(width * 0.028)));
  const cellConv = Math.min(28, Math.max(20, Math.floor(width * 0.028)));
  const cellPool = Math.min(36, Math.max(26, Math.floor(width * 0.035)));

  const totalW = 10 * cellIn + 8 * cellConv + 4 * cellPool + 180;
  const startX = (width - totalW) / 2;
  const centerY = height * 0.47;

  const inX = startX;
  const inY = centerY - (10 * cellIn) / 2;

  const convX = inX + 10 * cellIn + 95;
  const convY = centerY - (8 * cellConv) / 2;

  const poolX = convX + 8 * cellConv + 85;
  const poolY = centerY - (4 * cellPool) / 2;

  return { cellIn, cellConv, cellPool, inX, inY, convX, convY, poolX, poolY };
}

function handleCNNPointerInteraction(wx, wy, isClickStart = false) {
  const { cellIn, cellConv, inX, inY, convX, convY } = getCNNLayoutGeometry();

  // 1. Pintar o borrar en la matriz de Entrada 10x10
  if (
    wx >= inX &&
    wx < inX + 10 * cellIn &&
    wy >= inY &&
    wy < inY + 10 * cellIn
  ) {
    const col = Math.floor((wx - inX) / cellIn);
    const row = Math.floor((wy - inY) / cellIn);
    if (row >= 0 && row < 10 && col >= 0 && col < 10) {
      if (isClickStart) {
        state.m6.paintMode = state.m6.inputGrid[row][col] > 0.5 ? 0 : 1;
        state.m6.isPainting = true;
      }
      if (state.m6.isPainting) {
        state.m6.inputGrid[row][col] = state.m6.paintMode;
        recomputeCNN();
      }
      return true;
    }
  }

  // 2. Seleccionar celda del Mapa de Características 8x8 para inspeccionar su campo receptivo 3x3
  if (
    isClickStart &&
    wx >= convX &&
    wx < convX + 8 * cellConv &&
    wy >= convY &&
    wy < convY + 8 * cellConv
  ) {
    const col = Math.floor((wx - convX) / cellConv);
    const row = Math.floor((wy - convY) / cellConv);
    if (row >= 0 && row < 8 && col >= 0 && col < 8) {
      state.m6.scanX = col;
      state.m6.scanY = row;
      state.m6.autoScan = false;
      const chk = document.getElementById("chk-cnn-autoscan");
      if (chk) chk.checked = false;
      return true;
    }
  }
  return false;
}

// ==========================================
// FUNCIONES MATEMÁTICAS Y RED MULTICAPA
// ==========================================
const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-15, Math.min(15, x))));
const relu = (x) => Math.max(0, x);
const leakyRelu = (x) => (x > 0 ? x : 0.1 * x);
const sinAct = (x) => Math.sin(x);
const stepFn = (x) => (x >= 0 ? 1 : 0);

function evalActivation(z, type) {
  if (type === "step") return stepFn(z);
  if (type === "sigmoid") return sigmoid(z);
  if (type === "relu") return relu(z);
  if (type === "leaky_relu") return leakyRelu(z);
  if (type === "sin") return sinAct(z);
  if (type === "tanh") return Math.tanh(z);
  return stepFn(z);
}

class SimpleMLP {
  constructor(layers, lr = 0.1, activation = "tanh", regL2 = 0.0) {
    this.layers = layers;
    this.lr = lr;
    this.activation = activation;
    this.regL2 = regL2;
    this.weights = [];
    this.biases = [];
    this.activations = [];
    this.zs = [];

    for (let i = 0; i < layers.length - 1; i++) {
      const w = [];
      const scale = Math.sqrt(2 / layers[i]);
      for (let j = 0; j < layers[i]; j++) {
        const row = [];
        for (let k = 0; k < layers[i + 1]; k++) {
          row.push((Math.random() * 2 - 1) * scale);
        }
        w.push(row);
      }
      this.weights.push(w);
      this.biases.push(new Array(layers[i + 1]).fill(0));
    }
  }

  forward(inputs) {
    this.activations = [inputs];
    this.zs = [];
    let a = inputs;

    for (let i = 0; i < this.weights.length; i++) {
      const z = new Array(this.layers[i + 1]).fill(0);
      for (let j = 0; j < this.layers[i + 1]; j++) {
        let sum = this.biases[i][j];
        for (let k = 0; k < this.layers[i]; k++) {
          sum += a[k] * this.weights[i][k][j];
        }
        z[j] = sum;
      }
      this.zs.push(z);

      let actFn;
      if (this.activation === "relu") actFn = relu;
      else if (this.activation === "leaky_relu") actFn = leakyRelu;
      else if (this.activation === "sin") actFn = sinAct;
      else if (this.activation === "sigmoid") actFn = sigmoid;
      else actFn = Math.tanh;

      a = i === this.weights.length - 1 ? z.map(sigmoid) : z.map(actFn);
      this.activations.push(a);
    }
    return a;
  }

  backward(target) {
    const y = this.activations[this.activations.length - 1];
    let deltas = [];

    const outDelta = new Array(this.layers[this.layers.length - 1]);
    for (let i = 0; i < y.length; i++) {
      outDelta[i] = (y[i] - target[i]) * y[i] * (1 - y[i]);
    }
    deltas.unshift(outDelta);

    for (let l = this.weights.length - 1; l > 0; l--) {
      const hiddenDelta = new Array(this.layers[l]).fill(0);
      const a = this.activations[l];
      const z = this.zs[l - 1];
      for (let i = 0; i < this.layers[l]; i++) {
        let error = 0;
        for (let j = 0; j < this.layers[l + 1]; j++) {
          error += deltas[0][j] * this.weights[l][i][j];
        }
        let deriv = 1;
        if (this.activation === "tanh") deriv = 1 - a[i] * a[i];
        else if (this.activation === "relu") deriv = a[i] > 0 ? 1 : 0.01;
        else if (this.activation === "leaky_relu") deriv = z[i] > 0 ? 1 : 0.1;
        else if (this.activation === "sin") deriv = Math.cos(z[i]);
        else if (this.activation === "sigmoid") deriv = a[i] * (1 - a[i]);

        hiddenDelta[i] = error * deriv;
      }
      deltas.unshift(hiddenDelta);
    }

    for (let l = 0; l < this.weights.length; l++) {
      for (let i = 0; i < this.layers[l]; i++) {
        for (let j = 0; j < this.layers[l + 1]; j++) {
          const grad =
            deltas[l][j] * this.activations[l][i] +
            this.regL2 * this.weights[l][i][j];
          this.weights[l][i][j] -= this.lr * grad;
        }
      }
      for (let j = 0; j < this.layers[l + 1]; j++) {
        this.biases[l][j] -= this.lr * deltas[l][j];
      }
    }
  }
}

let sandboxNet = new SimpleMLP([2, 5, 4, 1], 0.12, "tanh", 0.0);
let currentData = [];

// ==========================================
// INTERACCIÓN DE RATÓN Y RUEDA (CANVAS)
// ==========================================
function getPlotGeometryM1() {
  const cx = width * 0.73;
  const cy = height * 0.5;
  const scale = Math.min(width, height) * 0.155;
  return { cx, cy, scale };
}

function getSandboxHeatmapRect() {
  const pad = 42;
  const mapSize = Math.min(width * 0.46, height - pad * 2);
  const mapX = width - mapSize - pad - 15;
  const mapY = (height - mapSize) / 2;
  return { mapX, mapY, mapSize };
}

let isRotating3D = false;
let rotStartX = 0,
  rotStartY = 0;
let startYaw = 0,
  startPitch = 0;

canvas.addEventListener("mousedown", (e) => {
  const rect = canvas.getBoundingClientRect();
  const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
  const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
  const wPt = screenToWorld(sx, sy);

  // En Módulo 6 (CNN): Comprobar si el usuario está dibujando en la cuadrícula 10x10 o eligiendo celda en 8x8
  if (state.module === 6) {
    if (handleCNNPointerInteraction(wPt.x, wPt.y, true)) {
      return;
    }
  }
  if (state.module === 7 && setClassifierPointFromCanvas(wPt.x, wPt.y)) {
    return;
  }

  // En Módulo 1: Comprobar si el usuario hace clic sobre el punto (x1, x2) para arrastrarlo directamente
  if (state.module === 1) {
    const { cx, cy, scale } = getPlotGeometryM1();
    const ptWorldX = cx + state.m1.x1 * scale;
    const ptWorldY = cy - state.m1.x2 * scale;
    if (Math.hypot(wPt.x - ptWorldX, wPt.y - ptWorldY) < 18) {
      isDraggingInputPoint = true;
      canvas.classList.add("dragging-canvas");
      return;
    }
  }

  // En Módulo 5 (3D): Arrastrar con botón izquierdo rota la superficie 3D; Shift+Arrastrar o botón central hace Pan
  if (state.module === 5 && !e.shiftKey && e.button === 0) {
    isRotating3D = true;
    camera.dragMoved = false;
    rotStartX = sx;
    rotStartY = sy;
    startYaw = state.m5.rotYaw;
    startPitch = state.m5.rotPitch;
    canvas.classList.add("dragging-canvas");
    return;
  }

  // En Módulo 4: Si hace clic rápido dentro del Heatmap y no está arrastrando, permite añadir punto o hacer pan
  camera.isDragging = true;
  camera.dragMoved = false;
  camera.dragStartX = sx;
  camera.dragStartY = sy;
  camera.camStartX = camera.targetX;
  camera.camStartY = camera.targetY;
  if (camera.panEnabled) canvas.classList.add("dragging-canvas");
});

window.addEventListener("mousemove", (e) => {
  const rect = canvas.getBoundingClientRect();
  mouseScreenX = (e.clientX - rect.left) * (canvas.width / rect.width);
  mouseScreenY = (e.clientY - rect.top) * (canvas.height / rect.height);
  const wPt = screenToWorld(mouseScreenX, mouseScreenY);
  mouseWorldX = wPt.x;
  mouseWorldY = wPt.y;

  if (state.module === 6 && state.m6.isPainting) {
    handleCNNPointerInteraction(mouseWorldX, mouseWorldY, false);
    return;
  }

  if (isRotating3D && state.module === 5) {
    const dx = mouseScreenX - rotStartX;
    const dy = mouseScreenY - rotStartY;
    if (Math.hypot(dx, dy) > 4) camera.dragMoved = true;
    state.m5.rotYaw = startYaw + dx * 0.008;
    state.m5.rotPitch = Math.max(0.15, Math.min(1.45, startPitch + dy * 0.007));
    return;
  }

  if (isDraggingInputPoint && state.module === 1) {
    const { cx, cy, scale } = getPlotGeometryM1();
    const newX1 = Math.max(-2, Math.min(2, (mouseWorldX - cx) / scale));
    const newX2 = Math.max(-2, Math.min(2, (cy - mouseWorldY) / scale));
    state.m1.x1 = +newX1.toFixed(2);
    state.m1.x2 = +newX2.toFixed(2);
    const sl1 = document.getElementById("slider-x1");
    const sl2 = document.getElementById("slider-x2");
    const vl1 = document.getElementById("val-x1");
    const vl2 = document.getElementById("val-x2");
    if (sl1) sl1.value = state.m1.x1;
    if (sl2) sl2.value = state.m1.x2;
    if (vl1) vl1.innerText = state.m1.x1.toFixed(2);
    if (vl2) vl2.innerText = state.m1.x2.toFixed(2);
    return;
  }

  if (camera.isDragging && camera.panEnabled) {
    const dx = mouseScreenX - camera.dragStartX;
    const dy = mouseScreenY - camera.dragStartY;
    if (Math.hypot(dx, dy) > 4) camera.dragMoved = true;
    camera.targetX = camera.camStartX + dx;
    camera.targetY = camera.camStartY + dy;
  }

  // Cambiar cursor si está sobre el punto interactivo del Módulo 1
  if (state.module === 1 && !camera.isDragging) {
    const { cx, cy, scale } = getPlotGeometryM1();
    const ptWorldX = cx + state.m1.x1 * scale;
    const ptWorldY = cy - state.m1.x2 * scale;
    if (Math.hypot(mouseWorldX - ptWorldX, mouseWorldY - ptWorldY) < 16) {
      canvas.classList.add("hover-interactive");
    } else {
      canvas.classList.remove("hover-interactive");
    }
  }
});

window.addEventListener("mouseup", () => {
  if (isRotating3D && !camera.dragMoved && state.module === 5) {
    // Clic limpio en el Módulo 5: intentar colocar la bola en el punto más cercano de la superficie 3D
    place3DBallAtCursor(mouseWorldX, mouseWorldY);
  }
  if (camera.isDragging && !camera.dragMoved && state.module === 4) {
    // Si fue un clic limpio dentro del mapa de calor en el Módulo 4, añadir un punto de entrenamiento
    const { mapX, mapY, mapSize } = getSandboxHeatmapRect();
    if (
      mouseWorldX >= mapX &&
      mouseWorldX <= mapX + mapSize &&
      mouseWorldY >= mapY &&
      mouseWorldY <= mapY + mapSize
    ) {
      const normX = (mouseWorldX - mapX) / mapSize;
      const normY = (mouseWorldY - mapY) / mapSize;
      currentData.push({ x: [normX, normY], y: [state.m4.addClassMode] });
      evaluateSandboxMetrics();
    }
  }
  isRotating3D = false;
  isDraggingInputPoint = false;
  state.m6.isPainting = false;
  camera.isDragging = false;
  canvas.classList.remove("dragging-canvas");
});

canvas.addEventListener(
  "wheel",
  (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const sx = (e.clientX - rect.left) * (canvas.width / rect.width);
    const sy = (e.clientY - rect.top) * (canvas.height / rect.height);
    const delta = e.deltaY < 0 ? 0.14 : -0.14;
    cameraZoomStep(delta, sx, sy);
  },
  { passive: false },
);

// Redimensionar Canvas
function resize() {
  const container = document.getElementById("canvas-container");
  if (!container) return;
  width = container.clientWidth;
  height = container.clientHeight;
  canvas.width = width;
  canvas.height = height;
  draw();
}
window.addEventListener("resize", resize);

// ==========================================
// PANTALLA COMPLETA (NATIVA + FALLBACK CSS)
// ==========================================
let isSimulatedFullscreen = false;
function toggleFullScreen() {
  const container = document.getElementById("canvas-container");
  const iconHud = document.getElementById("icon-fullscreen-hud");

  isSimulatedFullscreen = !isSimulatedFullscreen;

  if (isSimulatedFullscreen) {
    container.classList.remove(
      "flex-1",
      "relative",
      "border",
      "border-slate-800",
      "rounded-xl",
    );
    container.classList.add("fixed", "inset-0", "z-50", "bg-slate-950");
    if (iconHud) iconHud.classList.replace("fa-expand", "fa-compress");
    if (document.documentElement.requestFullscreen) {
      document.documentElement.requestFullscreen().catch(() => {});
    }
  } else {
    container.classList.remove("fixed", "inset-0", "z-50", "bg-slate-950");
    container.classList.add(
      "flex-1",
      "relative",
      "border",
      "border-slate-800",
      "rounded-xl",
    );
    if (iconHud) iconHud.classList.replace("fa-compress", "fa-expand");
    if (document.fullscreenElement && document.exitFullscreen) {
      document.exitFullscreen().catch(() => {});
    }
  }
  setTimeout(resize, 60);
}

// Alternar Visibilidad del Panel de Explicación
let showExplanation = true;
function toggleExplanation() {
  showExplanation = !showExplanation;
  const content = document.getElementById("explanation-content");
  const icon = document.getElementById("expl-toggle-icon");
  if (showExplanation) {
    content.style.display = "block";
    icon.classList.replace("fa-chevron-up", "fa-chevron-down");
  } else {
    content.style.display = "none";
    icon.classList.replace("fa-chevron-down", "fa-chevron-up");
  }
  setTimeout(resize, 50);
}

// ==========================================
// GENERACIÓN DE DATASETS (MÓDULO 4)
// ==========================================
function generateData(type) {
  const data = [];
  const noiseFactor = state.m4.noise;

  if (type === "xor") {
    const addNoise = (val) => val + (Math.random() - 0.5) * noiseFactor * 0.55;
    for (let i = 0; i < 45; i++) {
      data.push({ x: [addNoise(0.22), addNoise(0.22)], y: [0] });
      data.push({ x: [addNoise(0.78), addNoise(0.78)], y: [0] });
      data.push({ x: [addNoise(0.22), addNoise(0.78)], y: [1] });
      data.push({ x: [addNoise(0.78), addNoise(0.22)], y: [1] });
    }
  } else if (type === "circles") {
    for (let i = 0; i < 200; i++) {
      const isInner = i % 2 === 0;
      const angle = Math.random() * Math.PI * 2;
      const radius = isInner
        ? Math.random() * 0.14
        : 0.28 + Math.random() * 0.1;
      const nx = (Math.random() - 0.5) * noiseFactor * 0.5;
      const ny = (Math.random() - 0.5) * noiseFactor * 0.5;
      data.push({
        x: [
          0.5 + radius * Math.cos(angle) + nx,
          0.5 + radius * Math.sin(angle) + ny,
        ],
        y: [isInner ? 1 : 0],
      });
    }
  } else if (type === "moons") {
    for (let i = 0; i < 200; i++) {
      const isTop = i % 2 === 0;
      const angle = Math.random() * Math.PI;
      const nx = (Math.random() - 0.5) * noiseFactor * 0.6;
      const ny = (Math.random() - 0.5) * noiseFactor * 0.6;
      if (isTop) {
        data.push({
          x: [
            0.42 + 0.26 * Math.cos(angle) + nx,
            0.58 - 0.26 * Math.sin(angle) + ny,
          ],
          y: [1],
        });
      } else {
        data.push({
          x: [
            0.58 - 0.26 * Math.cos(angle) + nx,
            0.42 + 0.26 * Math.sin(angle) + ny,
          ],
          y: [0],
        });
      }
    }
  } else if (type === "spiral") {
    for (let i = 0; i < 220; i++) {
      const n = Math.sqrt(Math.random());
      const angle = n * Math.PI * 3.4;
      const radius = n * 0.42;
      const isClass1 = i % 2 === 0;
      const offset = isClass1 ? 0 : Math.PI;
      const nx = (Math.random() - 0.5) * noiseFactor * 0.4;
      const ny = (Math.random() - 0.5) * noiseFactor * 0.4;
      data.push({
        x: [
          0.5 + radius * Math.cos(angle + offset) + nx,
          0.5 + radius * Math.sin(angle + offset) + ny,
        ],
        y: [isClass1 ? 1 : 0],
      });
    }
  } else if (type === "blobs") {
    for (let i = 0; i < 160; i++) {
      const isClass1 = i % 2 === 0;
      const nx = (Math.random() - 0.5) * (0.18 + noiseFactor * 0.8);
      const ny = (Math.random() - 0.5) * (0.18 + noiseFactor * 0.8);
      if (isClass1) {
        data.push({ x: [0.28 + nx, 0.72 + ny], y: [1] });
      } else {
        data.push({ x: [0.72 + nx, 0.28 + ny], y: [0] });
      }
    }
  } else if (type === "checkerboard") {
    // Tablero de Ajedrez 3x3
    for (let i = 0; i < 240; i++) {
      const x = 0.08 + Math.random() * 0.84;
      const y = 0.08 + Math.random() * 0.84;
      const gx = Math.floor(((x - 0.08) / 0.84) * 3);
      const gy = Math.floor(((y - 0.08) / 0.84) * 3);
      const cls = (gx + gy) % 2 === 0 ? 1 : 0;
      const nx = (Math.random() - 0.5) * noiseFactor * 0.35;
      const ny = (Math.random() - 0.5) * noiseFactor * 0.35;
      data.push({
        x: [
          Math.max(0.02, Math.min(0.98, x + nx)),
          Math.max(0.02, Math.min(0.98, y + ny)),
        ],
        y: [cls],
      });
    }
  } else if (type === "rings") {
    // Triple Anillo Alterno
    for (let i = 0; i < 240; i++) {
      const ringIdx = i % 3;
      const baseR = ringIdx === 0 ? 0.07 : ringIdx === 1 ? 0.22 : 0.38;
      const cls = ringIdx === 1 ? 0 : 1;
      const angle = Math.random() * Math.PI * 2;
      const r = baseR + (Math.random() - 0.5) * (0.04 + noiseFactor * 0.15);
      data.push({
        x: [0.5 + r * Math.cos(angle), 0.5 + r * Math.sin(angle)],
        y: [cls],
      });
    }
  } else if (type === "sinwave") {
    // Frontera Sinusoidal
    for (let i = 0; i < 210; i++) {
      const x = 0.06 + Math.random() * 0.88;
      const y = 0.06 + Math.random() * 0.88;
      const waveY = 0.5 + 0.26 * Math.sin((x - 0.5) * Math.PI * 3);
      const ny = (Math.random() - 0.5) * noiseFactor * 0.45;
      const cls = y + ny > waveY ? 1 : 0;
      data.push({ x: [x, y], y: [cls] });
    }
  } else if (type === "fourclusters") {
    // 4 Islas Multiclase Binaria
    const centers = [
      { cx: 0.25, cy: 0.25, c: 1 },
      { cx: 0.75, cy: 0.25, c: 0 },
      { cx: 0.5, cy: 0.52, c: 1 },
      { cx: 0.5, cy: 0.82, c: 0 },
    ];
    for (let i = 0; i < 200; i++) {
      const ctr = centers[i % centers.length];
      const spread = 0.11 + noiseFactor * 0.35;
      data.push({
        x: [
          ctr.cx + (Math.random() - 0.5) * spread,
          ctr.cy + (Math.random() - 0.5) * spread,
        ],
        y: [ctr.c],
      });
    }
  }
  return data;
}

// ==========================================
// MATEMÁTICA Y OPTIMIZADORES DEL MÓDULO 5 (3D)
// ==========================================
// Dominio de parámetros w1, w2 en [-2.2, 2.2]
function evalLossSurface3D(w1, w2, type = state.m5.surface) {
  if (type === "bowl") {
    // Cuenco Convexo Simple
    return 0.38 * (w1 * w1 + w2 * w2);
  } else if (type === "saddle") {
    // Punto de Silla (Saddle Point)
    return 0.35 * (w1 * w1 - w2 * w2) + 0.9;
  } else if (type === "ravine") {
    // Valle Estrecho (Ill-conditioned Canyon)
    return 0.12 * w1 * w1 + 1.15 * w2 * w2;
  } else if (type === "rosenbrock") {
    // Valle Curvo tipo Rosenbrock suavizado
    const a = 0.8 - w1;
    const b = w2 - w1 * w1 * 0.45;
    return Math.min(2.6, 0.25 * a * a + 1.15 * b * b);
  } else {
    // Paisaje No Convexo con Mínimos Locales y Global
    const quad = 0.18 * (w1 * w1 + w2 * w2);
    const waves = 0.45 * Math.cos(2.3 * w1) * Math.sin(2.3 * w2);
    const biasTilt = 0.12 * w1 - 0.08 * w2;
    return quad + waves + biasTilt + 0.65;
  }
}

function evalSurfaceGradient3D(w1, w2, type = state.m5.surface) {
  const eps = 1e-4;
  const dw1 =
    (evalLossSurface3D(w1 + eps, w2, type) -
      evalLossSurface3D(w1 - eps, w2, type)) /
    (2 * eps);
  const dw2 =
    (evalLossSurface3D(w1, w2 + eps, type) -
      evalLossSurface3D(w1, w2 - eps, type)) /
    (2 * eps);
  return { dw1, dw2 };
}

function init3DBalls(startW1 = -1.65, startW2 = 1.55) {
  const mode = state.m5.optimizer;
  state.m5.stepCount = 0;
  state.m5.balls = [];

  const createBall = (optType, color, label) => ({
    optType,
    color,
    label,
    w1: startW1,
    w2: startW2,
    vx: 0,
    vy: 0,
    mx: 0,
    my: 0,
    rx: 0,
    ry: 0,
    t: 0,
    trail: [
      { w1: startW1, w2: startW2, loss: evalLossSurface3D(startW1, startW2) },
    ],
  });

  if (mode === "compare_all") {
    state.m5.balls.push(createBall("sgd", "#38bdf8", "SGD"));
    state.m5.balls.push(createBall("momentum", "#fbbf24", "Momentum"));
    state.m5.balls.push(createBall("adam", "#34d399", "Adam"));
  } else if (mode === "sgd") {
    state.m5.balls.push(createBall("sgd", "#38bdf8", "SGD Estándar"));
  } else if (mode === "momentum") {
    state.m5.balls.push(createBall("momentum", "#fbbf24", "SGD + Momentum"));
  } else if (mode === "adam") {
    state.m5.balls.push(createBall("adam", "#34d399", "Adam Adaptativo"));
  }
  updateM5Telemetry();
}

function step3DOptimizers() {
  const lr = state.m5.lr;
  const beta = state.m5.momentum;
  const beta1 = 0.9,
    beta2 = 0.999,
    eps = 1e-6;

  state.m5.balls.forEach((b) => {
    const { dw1, dw2 } = evalSurfaceGradient3D(b.w1, b.w2);
    b.t += 1;

    if (b.optType === "sgd") {
      b.w1 -= lr * dw1;
      b.w2 -= lr * dw2;
    } else if (b.optType === "momentum") {
      b.vx = beta * b.vx - lr * dw1;
      b.vy = beta * b.vy - lr * dw2;
      b.w1 += b.vx;
      b.w2 += b.vy;
    } else if (b.optType === "adam") {
      b.mx = beta1 * b.mx + (1 - beta1) * dw1;
      b.my = beta1 * b.my + (1 - beta1) * dw2;
      b.rx = beta2 * b.rx + (1 - beta2) * (dw1 * dw1);
      b.ry = beta2 * b.ry + (1 - beta2) * (dw2 * dw2);
      const mxHat = b.mx / (1 - Math.pow(beta1, b.t));
      const myHat = b.my / (1 - Math.pow(beta1, b.t));
      const rxHat = b.rx / (1 - Math.pow(beta2, b.t));
      const ryHat = b.ry / (1 - Math.pow(beta2, b.t));
      b.w1 -= (lr * 1.15 * mxHat) / (Math.sqrt(rxHat) + eps);
      b.w2 -= (lr * 1.15 * myHat) / (Math.sqrt(ryHat) + eps);
    }

    b.w1 = Math.max(-2.15, Math.min(2.15, b.w1));
    b.w2 = Math.max(-2.15, Math.min(2.15, b.w2));
    const currentLoss = evalLossSurface3D(b.w1, b.w2);
    b.trail.push({ w1: b.w1, w2: b.w2, loss: currentLoss });
    if (b.trail.length > 140) b.trail.shift();
  });

  state.m5.stepCount++;
  updateM5Telemetry();
}

function project3DPoint(w1, w2, loss) {
  const cx = width * 0.5;
  const cy = height * 0.56;
  const scale = Math.min(width, height) * 0.145;

  const yaw = state.m5.rotYaw;
  const pitch = state.m5.rotPitch;

  // Centrar altura z
  const z = (loss - 0.75) * 1.15;

  // Rotación horizontal (Yaw)
  const x1 = w1 * Math.cos(yaw) - w2 * Math.sin(yaw);
  const y1 = w1 * Math.sin(yaw) + w2 * Math.cos(yaw);

  // Rotación de inclinación (Pitch)
  const screenX = cx + x1 * scale;
  const screenY = cy + (y1 * Math.sin(pitch) - z * Math.cos(pitch)) * scale;
  const depth = y1 * Math.cos(pitch) + z * Math.sin(pitch);

  return { x: screenX, y: screenY, depth };
}

function place3DBallAtCursor(worldX, worldY) {
  // Buscar el punto (w1, w2) de la malla 3D cuya proyección esté más cerca del clic
  let bestW1 = -1.5,
    bestW2 = 1.5,
    minDist = Infinity;
  for (let w1 = -2.0; w1 <= 2.0; w1 += 0.15) {
    for (let w2 = -2.0; w2 <= 2.0; w2 += 0.15) {
      const loss = evalLossSurface3D(w1, w2);
      const pt = project3DPoint(w1, w2, loss);
      const d = Math.hypot(worldX - pt.x, worldY - pt.y);
      if (d < minDist) {
        minDist = d;
        bestW1 = w1;
        bestW2 = w2;
      }
    }
  }
  if (minDist < 48) {
    init3DBalls(+bestW1.toFixed(2), +bestW2.toFixed(2));
  }
}

function updateM5Telemetry() {
  const stepEl = document.getElementById("m5-step-lbl");
  const posEl = document.getElementById("m5-pos-lbl");
  const lossEl = document.getElementById("m5-loss-lbl");
  const gradEl = document.getElementById("m5-grad-lbl");
  if (!stepEl || state.m5.balls.length === 0) return;

  const mainBall = state.m5.balls[0];
  const loss = evalLossSurface3D(mainBall.w1, mainBall.w2);
  const { dw1, dw2 } = evalSurfaceGradient3D(mainBall.w1, mainBall.w2);
  const gradMag = Math.hypot(dw1, dw2);

  stepEl.innerText = state.m5.stepCount;
  posEl.innerText = `(${mainBall.w1.toFixed(2)}, ${mainBall.w2.toFixed(2)})`;
  lossEl.innerText = loss.toFixed(4);
  gradEl.innerText = gradMag.toFixed(4);
}

// ==========================================
// LÓGICA MATEMÁTICA MÓDULO 3 (BACKPROP REAL)
// ==========================================
function m3Compute() {
  const m = state.m3;
  const z_h = [
    m.x[0] * m.w_i_h[0][0] + m.x[1] * m.w_i_h[1][0] + m.b_h[0],
    m.x[0] * m.w_i_h[0][1] + m.x[1] * m.w_i_h[1][1] + m.b_h[1],
  ];
  const a_h = z_h.map(sigmoid);
  const z_o = a_h[0] * m.w_h_o[0] + a_h[1] * m.w_h_o[1] + m.b_o;
  const a_o = sigmoid(z_o);
  const loss = 0.5 * Math.pow(m.target - a_o, 2);
  m.fwd = { z_h, a_h, z_o, a_o, loss };

  const dL_dao = a_o - m.target;
  const dao_dzo = a_o * (1 - a_o);
  const delta_o = dL_dao * dao_dzo;
  const dw_h_o = [a_h[0] * delta_o, a_h[1] * delta_o];
  const db_o = delta_o;

  const delta_h = [
    delta_o * m.w_h_o[0] * (a_h[0] * (1 - a_h[0])),
    delta_o * m.w_h_o[1] * (a_h[1] * (1 - a_h[1])),
  ];
  const dw_i_h = [
    [m.x[0] * delta_h[0], m.x[0] * delta_h[1]],
    [m.x[1] * delta_h[0], m.x[1] * delta_h[1]],
  ];
  const db_h = delta_h;
  m.bwd = { delta_o, dw_h_o, db_o, delta_h, dw_i_h, db_h };
}

function m3ApplyWeights() {
  const m = state.m3;
  const lr = m.lr;
  m.prevWeights = {
    w_h_o: [...m.w_h_o],
    w_i_h: [[...m.w_i_h[0]], [...m.w_i_h[1]]],
  };
  m.w_h_o[0] -= lr * m.bwd.dw_h_o[0];
  m.w_h_o[1] -= lr * m.bwd.dw_h_o[1];
  m.b_o -= lr * m.bwd.db_o;
  m.w_i_h[0][0] -= lr * m.bwd.dw_i_h[0][0];
  m.w_i_h[0][1] -= lr * m.bwd.dw_i_h[0][1];
  m.w_i_h[1][0] -= lr * m.bwd.dw_i_h[1][0];
  m.w_i_h[1][1] -= lr * m.bwd.dw_i_h[1][1];
  m.b_h[0] -= lr * m.bwd.db_h[0];
  m.b_h[1] -= lr * m.bwd.db_h[1];
  m.epoch++;
  m3Compute();
  m.lossHistory.push(m.fwd.loss);
  if (m.lossHistory.length > 40) m.lossHistory.shift();
}

function bpFastTrain(steps = 15) {
  for (let i = 0; i < steps; i++) {
    m3Compute();
    m3ApplyWeights();
  }
  state.m3.phase = 4;
  state.m3.progress = 1;
  state.m3.runMode = "paused";
  document.getElementById("btn-play").innerHTML =
    '<i class="fas fa-play text-xs"></i>';
}

// ==========================================
// CLASIFICADOR MULTICLASE (MÓDULO 7)
// ==========================================
const classifierColors = ["#38bdf8", "#fb7185", "#34d399", "#fbbf24", "#c084fc"];
const classifierLetters = ["A", "B", "C", "D", "E"];

function classifierForward(input) {
  const m = state.m7;
  const activations = [input];
  const preActivations = [];

  for (let layerIndex = 0; layerIndex < m.weights.length; layerIndex++) {
    const previous = activations[layerIndex];
    const z = m.weights[layerIndex].map((weights, neuronIndex) =>
      weights.reduce(
        (sum, weight, inputIndex) => sum + weight * previous[inputIndex],
        m.networkBiases[layerIndex][neuronIndex],
      ),
    );
    preActivations.push(z);
    if (layerIndex === m.weights.length - 1) {
      const maxLogit = Math.max(...z);
      const expLogits = z.map((value) => Math.exp(value - maxLogit));
      const sum = expLogits.reduce((total, value) => total + value, 0);
      activations.push(expLogits.map((value) => value / sum));
    } else {
      activations.push(z.map((value) => Math.max(0, value)));
    }
  }

  return { activations, preActivations, probabilities: activations.at(-1) };
}

function classifierGaussianNoise() {
  const first = Math.max(Number.EPSILON, Math.random());
  const second = Math.random();
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * second);
}

function generateClassifierDataset() {
  const m = state.m7;
  m.dataset = [];
  if (m.datasetType === "manual") return;

  const samplesPerClass = 36;
  for (let classIndex = 0; classIndex < m.classCount; classIndex++) {
    const classAngle = (2 * Math.PI * classIndex) / m.classCount;
    for (let sampleIndex = 0; sampleIndex < samplesPerClass; sampleIndex++) {
      let x1;
      let x2;
      const noise = m.noise;

      if (m.datasetType === "circles") {
        const angle = Math.random() * Math.PI * 2;
        const radius =
          0.16 + (0.68 * classIndex) / Math.max(1, m.classCount - 1);
        x1 = Math.cos(angle) * radius + classifierGaussianNoise() * noise * 0.45;
        x2 = Math.sin(angle) * radius + classifierGaussianNoise() * noise * 0.45;
      } else if (m.datasetType === "spirals") {
        const progress = 0.08 + Math.random() * 0.84;
        const angle = progress * Math.PI * 4 + classAngle;
        x1 = progress * Math.cos(angle) + classifierGaussianNoise() * noise;
        x2 = progress * Math.sin(angle) + classifierGaussianNoise() * noise;
      } else {
        const centerX = 0.62 * Math.cos(classAngle);
        const centerY = 0.62 * Math.sin(classAngle);
        x1 = centerX + classifierGaussianNoise() * (0.11 + noise);
        x2 = centerY + classifierGaussianNoise() * (0.11 + noise);
      }

      m.dataset.push({
        x: [Math.max(-1, Math.min(1, x1)), Math.max(-1, Math.min(1, x2))],
        label: classIndex,
      });
    }
  }
}

function initializeClassifier(resetDataset = false) {
  const m = state.m7;
  const layerSizes = [
    2,
    ...Array(m.hiddenLayers).fill(m.neuronsPerLayer),
    m.classCount,
  ];
  m.weights = [];
  m.networkBiases = [];

  for (let layerIndex = 0; layerIndex < layerSizes.length - 1; layerIndex++) {
    const fanIn = layerSizes[layerIndex];
    const fanOut = layerSizes[layerIndex + 1];
    const limit = Math.sqrt(6 / (fanIn + fanOut));
    m.weights.push(
      Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, () => (Math.random() * 2 - 1) * limit),
      ),
    );
    m.networkBiases.push(Array.from({ length: fanOut }, () => 0));
  }

  if (resetDataset || !m.dataset.length) generateClassifierDataset();
  m.trained = false;
  m.loss = null;
}

function getClassifierProbabilities() {
  return classifierForward([state.m7.x1, state.m7.x2]).probabilities;
}

function classifierLoss() {
  const m = state.m7;
  if (!m.dataset.length) return null;
  const totalLoss = m.dataset.reduce((total, sample) => {
    const probability = classifierForward(sample.x).probabilities[sample.label];
    return total - Math.log(Math.max(probability, 1e-12));
  }, 0);
  return totalLoss / m.dataset.length;
}

function getClassifierOptimizerLabel() {
  return {
    batch: "Batch GD",
    sgd: "SGD",
    minibatch: "Mini-batch",
  }[state.m7.optimizer];
}

function renderClassifierResults() {
  const resultsEl = document.getElementById("classifier-results");
  if (!resultsEl) return;

  const probabilities = getClassifierProbabilities();
  const prediction = probabilities.indexOf(Math.max(...probabilities));
  const confidence = Math.round(probabilities[prediction] * 100);
  const displayedPercentages = probabilities.map(
    (probability) => Math.round(probability * 1000) / 10,
  );
  const displayedTotal = displayedPercentages.reduce(
    (total, percentage) => total + Math.round(percentage * 10),
    0,
  );
  displayedPercentages[displayedPercentages.length - 1] +=
    (1000 - displayedTotal) / 10;
  const predictionEl = document.getElementById("classifier-prediction");
  const statusEl = document.getElementById("classifier-training-status");

  predictionEl.innerHTML = `<span style="color:${classifierColors[prediction]}">Clase ${classifierLetters[prediction]}</span> · ${confidence}%`;
  resultsEl.innerHTML = probabilities
    .map(
      (_probability, index) => `
        <div class="flex items-center gap-2">
            <span class="w-14 shrink-0 text-[11px] text-slate-300">Clase ${classifierLetters[index]}</span>
            <div class="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                <div class="h-full rounded-full transition-all duration-200" style="width:${displayedPercentages[index].toFixed(1)}%;background:${classifierColors[index]}"></div>
            </div>
            <span class="w-10 text-right font-mono text-[10px] text-slate-300">${displayedPercentages[index].toFixed(1)}%</span>
        </div>`,
    )
    .join("");

  const loss = state.m7.trained ? classifierLoss() : null;
  state.m7.loss = loss;
  if (statusEl) {
    statusEl.textContent = state.m7.trained
      ? `${getClassifierOptimizerLabel()} · pérdida ${loss.toFixed(3)} · ${state.m7.dataset.length} ejemplos`
      : state.m7.dataset.length
        ? `${state.m7.dataset.length} ejemplos · listo para entrenar`
        : "Dataset vacío · añade puntos en modo manual";
  }
}

function setClassifierInput(axis, value) {
  const numericValue = Math.round(
    Math.max(-1, Math.min(1, Number(value))) * 100,
  ) / 100;
  if (axis === "x1") state.m7.x1 = numericValue;
  else state.m7.x2 = numericValue;

  const valueEl = document.getElementById(`classifier-${axis}-value`);
  if (valueEl) valueEl.textContent = numericValue.toFixed(2);
  renderClassifierResults();
}

function setClassifierClassCount(value) {
  state.m7.classCount = Math.max(2, Math.min(5, Number(value)));
  state.m7.manualClass = Math.min(
    state.m7.manualClass,
    state.m7.classCount - 1,
  );
  generateClassifierDataset();
  initializeClassifier();
  setModule(7);
}

function resetClassifier() {
  initializeClassifier();
  renderClassifierResults();
}

function setClassifierDataset(datasetType) {
  state.m7.datasetType = datasetType;
  state.m7.interactionMode = datasetType === "manual" ? "add" : "probe";
  generateClassifierDataset();
  initializeClassifier();
  state.m7.trained = false;
  state.m7.loss = null;
  setModule(7);
}

function setClassifierNoise(value) {
  state.m7.noise = Number(value);
  const valueEl = document.getElementById("classifier-noise-value");
  if (valueEl) valueEl.textContent = state.m7.noise.toFixed(2);
  if (state.m7.datasetType !== "manual") {
    generateClassifierDataset();
    initializeClassifier();
  }
  state.m7.trained = false;
  state.m7.loss = null;
  renderClassifierResults();
}

function setClassifierArchitecture(hiddenLayers, neuronsPerLayer) {
  state.m7.hiddenLayers = Number(hiddenLayers);
  state.m7.neuronsPerLayer = Number(neuronsPerLayer);
  initializeClassifier();
  renderClassifierResults();
}

function setClassifierManualClass(value) {
  state.m7.manualClass = Math.max(
    0,
    Math.min(state.m7.classCount - 1, Number(value)),
  );
  if (state.m7.interactionMode === "add") setClassifierInteractionMode("add");
}

function clearClassifierDataset() {
  state.m7.dataset = [];
  state.m7.trained = false;
  state.m7.loss = null;
  renderClassifierResults();
}

function setClassifierInteractionMode(mode) {
  state.m7.interactionMode = mode;
  const probeButton = document.getElementById("classifier-mode-probe");
  const addButton = document.getElementById("classifier-mode-add");
  if (probeButton) {
    probeButton.className =
      mode === "probe"
        ? "flex-1 rounded-lg border border-blue-500/50 bg-blue-600/20 px-2 py-1.5 text-[10px] font-semibold text-blue-200"
        : "flex-1 rounded-lg border border-slate-700 bg-slate-800 px-2 py-1.5 text-[10px] font-semibold text-slate-300";
  }
  if (addButton) {
    addButton.className =
      mode === "add"
        ? "flex-1 rounded-lg border border-amber-500/50 bg-amber-600/20 px-2 py-1.5 text-[10px] font-semibold text-amber-200"
        : "flex-1 rounded-lg border border-slate-700 bg-slate-800 px-2 py-1.5 text-[10px] font-semibold text-slate-300";
  }
  const hint = document.getElementById("classifier-map-hint");
  if (hint) {
    hint.textContent =
      mode === "add"
        ? `Modo añadir: pulsa el mapa para colocar un punto de clase ${classifierLetters[state.m7.manualClass]}.`
        : "Pulsa el mapa para probar una entrada; cambia a modo añadir para etiquetar ejemplos.";
  }
  const topHint = document.getElementById("top-hint-text");
  if (topHint) {
    topHint.textContent =
      mode === "add"
        ? "Modo añadir: pulsa en el mapa para colocar un ejemplo etiquetado"
        : "Modo probar: pulsa en el mapa para cambiar x₁ y x₂";
  }
}

function trainClassifier() {
  const m = state.m7;
  if (!m.dataset.length) {
    renderClassifierResults();
    return;
  }

  const learningRate = {
    batch: 0.45,
    sgd: 0.02,
    minibatch: 0.1,
  }[m.optimizer];
  const batchSize =
    m.optimizer === "batch" ? m.dataset.length : m.optimizer === "sgd" ? 1 : 16;
  const epochs = 80;

  const updateBatch = (batch) => {
    const weightGradients = m.weights.map((layer) =>
      layer.map((neuron) => neuron.map(() => 0)),
    );
    const biasGradients = m.networkBiases.map((layer) =>
      layer.map(() => 0),
    );

    for (const sample of batch) {
      const { activations, preActivations, probabilities } =
        classifierForward(sample.x);
      let deltas = probabilities.map(
        (probability, classIndex) =>
          probability - (classIndex === sample.label ? 1 : 0),
      );

      for (let layerIndex = m.weights.length - 1; layerIndex >= 0; layerIndex--) {
        const previousActivations = activations[layerIndex];
        for (let neuronIndex = 0; neuronIndex < deltas.length; neuronIndex++) {
          biasGradients[layerIndex][neuronIndex] += deltas[neuronIndex];
          for (
            let inputIndex = 0;
            inputIndex < previousActivations.length;
            inputIndex++
          ) {
            weightGradients[layerIndex][neuronIndex][inputIndex] +=
              deltas[neuronIndex] * previousActivations[inputIndex];
          }
        }

        if (layerIndex > 0) {
          const previousDeltas = m.weights[layerIndex][0].map((_, inputIndex) => {
            const propagated = deltas.reduce(
              (sum, delta, neuronIndex) =>
                sum + delta * m.weights[layerIndex][neuronIndex][inputIndex],
              0,
            );
            return preActivations[layerIndex - 1][inputIndex] > 0
              ? propagated
              : 0;
          });
          deltas = previousDeltas;
        }
      }
    }

    const scale = learningRate / batch.length;
    for (let layerIndex = 0; layerIndex < m.weights.length; layerIndex++) {
      for (
        let neuronIndex = 0;
        neuronIndex < m.weights[layerIndex].length;
        neuronIndex++
      ) {
        m.networkBiases[layerIndex][neuronIndex] -=
          biasGradients[layerIndex][neuronIndex] * scale;
        for (
          let inputIndex = 0;
          inputIndex < m.weights[layerIndex][neuronIndex].length;
          inputIndex++
        ) {
          m.weights[layerIndex][neuronIndex][inputIndex] -=
            weightGradients[layerIndex][neuronIndex][inputIndex] * scale;
        }
      }
    }
  };

  for (let epoch = 0; epoch < epochs; epoch++) {
    const shuffled = [...m.dataset];
    for (let index = shuffled.length - 1; index > 0; index--) {
      const swapIndex = Math.floor(Math.random() * (index + 1));
      [shuffled[index], shuffled[swapIndex]] = [
        shuffled[swapIndex],
        shuffled[index],
      ];
    }
    for (let start = 0; start < shuffled.length; start += batchSize) {
      updateBatch(shuffled.slice(start, start + batchSize));
    }
  }

  m.loss = classifierLoss();
  m.trained = true;
  renderClassifierResults();
}

function getClassifierPlotGeometry() {
  const size = Math.min(width * 0.56, height * 0.72, 560);
  return {
    x: width * 0.36 - size / 2,
    y: height / 2 - size / 2,
    size,
  };
}

function setClassifierPointFromCanvas(x, y) {
  const plot = getClassifierPlotGeometry();
  if (x < plot.x || x > plot.x + plot.size || y < plot.y || y > plot.y + plot.size) {
    return false;
  }

  const x1 = Math.round((((x - plot.x) / plot.size) * 2 - 1) * 100) / 100;
  const x2 = Math.round((1 - ((y - plot.y) / plot.size) * 2) * 100) / 100;
  if (state.m7.interactionMode === "add") {
    state.m7.dataset.push({
      x: [x1, x2],
      label: state.m7.manualClass,
      manual: true,
    });
    state.m7.trained = false;
    state.m7.loss = null;
    renderClassifierResults();
  } else {
    setClassifierInput("x1", x1);
    setClassifierInput("x2", x2);
  }
  const x1Slider = document.getElementById("classifier-x1");
  const x2Slider = document.getElementById("classifier-x2");
  if (state.m7.interactionMode !== "add") {
    if (x1Slider) x1Slider.value = state.m7.x1;
    if (x2Slider) x2Slider.value = state.m7.x2;
  }
  return true;
}

// ==========================================
// CONFIGURACIÓN DE MÓDULOS Y SIDEBAR
// ==========================================
function setModule(modIndex) {
  state.module = modIndex;
  resetCamera();

  document
    .querySelectorAll(".module-btn")
    .forEach((b) => b.classList.remove("active"));
  document.getElementById(`btn-mod-${modIndex}`).classList.add("active");
  document.getElementById("module-badge").innerText = `Módulo ${modIndex}`;

  const titleEl = document.getElementById("module-title");
  const descEl = document.getElementById("module-desc");
  const controlsEl = document.getElementById("controls-container");
  const explEl = document.getElementById("explanation-content");
  const hintText = document.getElementById("top-hint-text");

  document.getElementById("player-controls").classList.add("hidden");
  document.getElementById("metrics-overlay").classList.add("hidden");
  state.m4.isTraining = false;

  if (modIndex === 1) {
    titleEl.innerText = "La Neurona Artificial";
    descEl.innerText =
      "Explora cómo una neurona pondera entradas, suma el sesgo y traza una frontera de decisión lineal en el espacio 2D.";
    hintText.innerText =
      "Tip: Arrastra el punto (x₁, x₂) directamente en el plano 2D o usa el menú inferior derecho para hacer Zoom/Pan";
    controlsEl.innerHTML = `
                    <div class="flex items-center justify-between bg-slate-950/60 p-2.5 rounded-lg border border-slate-800">
                        <label class="flex items-center gap-2 cursor-pointer">
                            <input type="checkbox" class="w-4 h-4 accent-blue-500 rounded cursor-pointer" 
                                   onchange="state.m1.showHelp = this.checked;" ${state.m1.showHelp ? "checked" : ""}>
                            <span class="text-xs text-slate-300 font-medium"><i class="fas fa-wand-magic-sparkles text-amber-400 mr-1"></i> Inspector al Pasar Cursor</span>
                        </label>
                    </div>

                    <div class="flex flex-col gap-1.5">
                        <label class="text-[11px] text-slate-400 font-bold uppercase tracking-wider">Función de Activación f(z)</label>
                        <select onchange="state.m1.actFn = this.value;" class="bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 focus:border-blue-500 outline-none">
                            <option value="step" ${state.m1.actFn === "step" ? "selected" : ""}>Escalón (Heaviside Step)</option>
                            <option value="sigmoid" ${state.m1.actFn === "sigmoid" ? "selected" : ""}>Sigmoide σ(z) (Probabilidad Suave)</option>
                            <option value="tanh" ${state.m1.actFn === "tanh" ? "selected" : ""}>Tangente Hiperbólica tanh(z)</option>
                            <option value="relu" ${state.m1.actFn === "relu" ? "selected" : ""}>ReLU max(0, z)</option>
                        </select>
                    </div>

                    <div class="h-px bg-slate-800"></div>

                    <div class="space-y-3 bg-slate-950/40 p-3 rounded-xl border border-slate-800/80">
                        <div class="text-[10px] font-bold uppercase tracking-wider text-blue-400 flex justify-between">
                            <span>Coordenadas de Entrada</span>
                            <span class="text-slate-500">Arrastrables en 2D</span>
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span class="text-slate-300">Entrada <b class="font-mono text-blue-400">x₁</b></span>
                                <span id="val-x1" class="font-mono text-blue-400 font-bold">${state.m1.x1.toFixed(2)}</span>
                            </div>
                            <input id="slider-x1" type="range" class="slider-custom" min="-2" max="2" step="0.05" value="${state.m1.x1}" oninput="updateState('m1','x1',this.value); document.getElementById('val-x1').innerText=parseFloat(this.value).toFixed(2)">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span class="text-slate-300">Entrada <b class="font-mono text-blue-400">x₂</b></span>
                                <span id="val-x2" class="font-mono text-blue-400 font-bold">${state.m1.x2.toFixed(2)}</span>
                            </div>
                            <input id="slider-x2" type="range" class="slider-custom" min="-2" max="2" step="0.05" value="${state.m1.x2}" oninput="updateState('m1','x2',this.value); document.getElementById('val-x2').innerText=parseFloat(this.value).toFixed(2)">
                        </div>
                    </div>

                    <div class="space-y-3 bg-slate-950/40 p-3 rounded-xl border border-slate-800/80">
                        <div class="text-[10px] font-bold uppercase tracking-wider text-emerald-400">Parámetros Sinápticos</div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span class="text-slate-300">Peso <b class="font-mono text-emerald-400">w₁</b></span>
                                <span id="val-w1" class="font-mono text-emerald-400 font-bold">${state.m1.w1.toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="-4" max="4" step="0.1" value="${state.m1.w1}" oninput="updateState('m1','w1',this.value); document.getElementById('val-w1').innerText=parseFloat(this.value).toFixed(2)">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span class="text-slate-300">Peso <b class="font-mono text-emerald-400">w₂</b></span>
                                <span id="val-w2" class="font-mono text-emerald-400 font-bold">${state.m1.w2.toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="-4" max="4" step="0.1" value="${state.m1.w2}" oninput="updateState('m1','w2',this.value); document.getElementById('val-w2').innerText=parseFloat(this.value).toFixed(2)">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span class="text-slate-300">Sesgo (Bias) <b class="font-mono text-amber-400">b</b></span>
                                <span id="val-b" class="font-mono text-amber-400 font-bold">${state.m1.b.toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="-4" max="4" step="0.1" value="${state.m1.b}" oninput="updateState('m1','b',this.value); document.getElementById('val-b').innerText=parseFloat(this.value).toFixed(2)">
                        </div>
                    </div>
                `;
    explEl.innerHTML = `
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                        <div class="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-emerald-400 font-bold">Vector de Pesos (w₁, w₂):</span> Define la orientación de la frontera. La flecha verde <span class="font-mono text-emerald-300">w⃗</span> en el gráfico es siempre perpendicular a la recta de decisión y apunta hacia la región positiva.
                        </div>
                        <div class="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-amber-400 font-bold">Sesgo / Bias (b):</span> Desplaza la recta paralelamente sin rotarla. Permite que la frontera no esté obligada a cruzar por el origen <span class="font-mono">(0,0)</span>.
                        </div>
                        <div class="bg-slate-950/70 p-2.5 rounded-lg border border-slate-800">
                            <span class="text-blue-400 font-bold">Ecuación Lineal:</span> La frontera exacta ocurre donde <span class="font-mono text-white">w₁x₁ + w₂x₂ + b = 0</span>. Cambia entre <em>Escalón</em> y <em>Sigmoide</em> para ver cómo se suaviza la decisión.
                        </div>
                    </div>
                `;
    renderModuleExplanation();
  } else if (modIndex === 2) {
    state.m1.solvedXor = false;
    titleEl.innerText = "El Desafío XOR";
    descEl.innerText =
      "Comprueba por qué una sola neurona es incapaz de resolver la puerta lógica XOR y cómo una Capa Oculta deforma el espacio.";
    hintText.innerText =
      "Intenta separar los 2 puntos azules de los 2 puntos rojos con una sola línea recta";
    controlsEl.innerHTML = `
                    <div class="text-xs text-slate-300 bg-slate-950/80 p-3 rounded-xl border border-slate-800 space-y-1.5">
                        <div class="font-bold text-white flex items-center justify-between">
                            <span>Tabla de Verdad XOR</span>
                            <span id="xor-score-badge" class="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[11px]">3/4 Aciertos</span>
                        </div>
                        <div class="grid grid-cols-2 gap-1.5 font-mono text-[11px] pt-1">
                            <div class="bg-slate-900 p-1.5 rounded border border-slate-800 text-rose-400">(0,0) → 0 (Rojo)</div>
                            <div class="bg-slate-900 p-1.5 rounded border border-slate-800 text-blue-400">(0,1) → 1 (Azul)</div>
                            <div class="bg-slate-900 p-1.5 rounded border border-slate-800 text-blue-400">(1,0) → 1 (Azul)</div>
                            <div class="bg-slate-900 p-1.5 rounded border border-slate-800 text-rose-400">(1,1) → 0 (Rojo)</div>
                        </div>
                    </div>

                    <div id="xor-linear-controls" class="space-y-3 bg-slate-950/40 p-3 rounded-xl border border-slate-800">
                        <div class="text-[10px] font-bold uppercase tracking-wider text-slate-400">Ajuste Manual (1 Neurona)</div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs"><span>Peso w₁</span><span id="xor-w1" class="font-mono text-emerald-400">${state.m1.w1.toFixed(1)}</span></div>
                            <input type="range" class="slider-custom" min="-4" max="4" step="0.1" value="${state.m1.w1}" oninput="updateState('m1','w1',this.value); state.m1.solvedXor=false; document.getElementById('xor-w1').innerText=parseFloat(this.value).toFixed(1);">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs"><span>Peso w₂</span><span id="xor-w2" class="font-mono text-emerald-400">${state.m1.w2.toFixed(1)}</span></div>
                            <input type="range" class="slider-custom" min="-4" max="4" step="0.1" value="${state.m1.w2}" oninput="updateState('m1','w2',this.value); state.m1.solvedXor=false; document.getElementById('xor-w2').innerText=parseFloat(this.value).toFixed(1);">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs"><span>Sesgo b</span><span id="xor-b" class="font-mono text-amber-400">${state.m1.b.toFixed(1)}</span></div>
                            <input type="range" class="slider-custom" min="-4" max="4" step="0.1" value="${state.m1.b}" oninput="updateState('m1','b',this.value); state.m1.solvedXor=false; document.getElementById('xor-b').innerText=parseFloat(this.value).toFixed(1);">
                        </div>
                    </div>

                    <button onclick="solveXOR()" class="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-500 hover:to-indigo-500 text-white font-bold py-2.5 px-4 rounded-xl shadow-lg shadow-purple-600/30 transition duration-200 w-full border border-purple-400/30 text-xs flex items-center justify-center gap-2">
                        <i class="fas fa-layer-group"></i> Añadir Capa Oculta (Resolver XOR)
                    </button>
                    <button onclick="state.m1.solvedXor = false;" class="bg-slate-800 hover:bg-slate-700 text-slate-300 font-semibold py-2 px-4 rounded-xl transition text-xs w-full border border-slate-700">
                        Volver a 1 Sola Neurona
                    </button>
                `;
    renderModuleExplanation();
  } else if (modIndex === 3) {
    titleEl.innerText = "Retropropagación (Backprop)";
    descEl.innerText =
      "Inspecciona en tiempo real el flujo hacia adelante (Forward), el cálculo del Error y los gradientes de la Regla de la Cadena.";
    hintText.innerText =
      "Usa los controles superiores para avanzar fase por fase o entrenar épocas completas";
    controlsEl.innerHTML = `
                    <div class="bg-slate-950/70 p-3 rounded-xl border border-slate-800 space-y-2.5">
                        <div class="text-[10px] font-bold uppercase tracking-wider text-blue-400">Configuración del Caso</div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span>Entrada <b class="font-mono text-blue-400">x₁</b></span>
                                <span id="bp-x1-lbl" class="font-mono text-blue-400">${state.m3.x[0].toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0" max="1" step="0.1" value="${state.m3.x[0]}" oninput="state.m3.x[0]=parseFloat(this.value); document.getElementById('bp-x1-lbl').innerText=this.value; m3Compute();">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span>Entrada <b class="font-mono text-blue-400">x₂</b></span>
                                <span id="bp-x2-lbl" class="font-mono text-blue-400">${state.m3.x[1].toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0" max="1" step="0.1" value="${state.m3.x[1]}" oninput="state.m3.x[1]=parseFloat(this.value); document.getElementById('bp-x2-lbl').innerText=this.value; m3Compute();">
                        </div>
                        <div class="flex flex-col gap-1">
                            <div class="flex justify-between text-xs">
                                <span>Valor Objetivo (<b class="text-emerald-400">Target y</b>)</span>
                                <span id="bp-target-lbl" class="font-mono text-emerald-400 font-bold">${state.m3.target.toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0.05" max="0.95" step="0.05" value="${state.m3.target}" oninput="state.m3.target=parseFloat(this.value); document.getElementById('bp-target-lbl').innerText=parseFloat(this.value).toFixed(2); m3Compute();">
                        </div>
                        <div class="flex flex-col gap-1 pt-1">
                            <div class="flex justify-between text-xs">
                                <span>Learning Rate (<b class="text-amber-400">η</b>)</span>
                                <span id="bp-lr-lbl" class="font-mono text-amber-400">${state.m3.lr.toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0.1" max="2.0" step="0.1" value="${state.m3.lr}" oninput="state.m3.lr=parseFloat(this.value); document.getElementById('bp-lr-lbl').innerText=parseFloat(this.value).toFixed(2);">
                        </div>
                    </div>

                    <div class="text-xs text-slate-300 bg-slate-950/60 p-3 rounded-xl border border-slate-800 space-y-2">
                        <div class="font-bold text-white text-[11px] uppercase tracking-wider mb-1">Fases del Ciclo</div>
                        <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-blue-400"></span><span><b>1. Forward:</b> Señal <span class="font-mono">a = σ(Σwx+b)</span></span></div>
                        <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-amber-400"></span><span><b>2. Loss:</b> Error <span class="font-mono">L = ½(y - a)²</span></span></div>
                        <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-rose-400"></span><span><b>3. Backward:</b> Gradientes <span class="font-mono">∂L/∂w</span></span></div>
                        <div class="flex items-center gap-2"><span class="w-2.5 h-2.5 rounded-full bg-emerald-400"></span><span><b>4. Update:</b> <span class="font-mono">w ← w - η·∂L/∂w</span></span></div>
                    </div>
                `;
    renderModuleExplanation();
    document.getElementById("player-controls").classList.remove("hidden");
    state.m3.runMode = "paused";
    state.m3.phase = 1;
    state.m3.progress = 1;
    state.m3.waitFrames = 0;
    m3Compute();
    if (state.m3.lossHistory.length === 0)
      state.m3.lossHistory.push(state.m3.fwd.loss);
  } else if (modIndex === 4) {
    titleEl.innerText = "Sandbox Multicapa Avanzado";
    descEl.innerText =
      "Experimenta con 9 conjuntos de datos, arquitecturas personalizables, funciones de activación no lineales y regularización.";
    hintText.innerText =
      "Haz clic en el mapa de calor para añadir puntos · Usa el panel izquierdo para personalizar capas y activación";
    controlsEl.innerHTML = `
                    <div class="flex flex-col gap-1">
                        <label class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Conjunto de Datos (9 Datasets)</label>
                        <select onchange="changeDataset(this.value)" class="bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 outline-none focus:border-blue-500">
                            <option value="moons" ${state.m4.dataset === "moons" ? "selected" : ""}>🌙 Dos Lunas Entrelazadas</option>
                            <option value="circles" ${state.m4.dataset === "circles" ? "selected" : ""}>🎯 Círculos Concéntricos</option>
                            <option value="spiral" ${state.m4.dataset === "spiral" ? "selected" : ""}>🌀 Espiral Doble (Reto Profundo)</option>
                            <option value="checkerboard" ${state.m4.dataset === "checkerboard" ? "selected" : ""}>🏁 Tablero de Ajedrez 3x3 (Nuevo)</option>
                            <option value="rings" ${state.m4.dataset === "rings" ? "selected" : ""}>🪐 Triple Anillo Alterno (Nuevo)</option>
                            <option value="sinwave" ${state.m4.dataset === "sinwave" ? "selected" : ""}>〰️ Frontera Sinusoidal (Nuevo)</option>
                            <option value="fourclusters" ${state.m4.dataset === "fourclusters" ? "selected" : ""}>🏝️ Cuatro Islas Multiclase (Nuevo)</option>
                            <option value="xor" ${state.m4.dataset === "xor" ? "selected" : ""}>❌ Cuadrantes XOR</option>
                            <option value="blobs" ${state.m4.dataset === "blobs" ? "selected" : ""}>⚪ Grupos Lineales</option>
                        </select>
                    </div>
                    
                    <div class="grid grid-cols-2 gap-2">
                        <div class="flex flex-col gap-1 bg-slate-950/50 p-2 rounded-lg border border-slate-800">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Ruido</span>
                                <span id="val-noise" class="text-blue-400 font-mono">${Math.round(state.m4.noise * 100)}%</span>
                            </div>
                            <input type="range" class="slider-custom" min="0" max="0.4" step="0.02" value="${state.m4.noise}" oninput="state.m4.noise = parseFloat(this.value); document.getElementById('val-noise').innerText = Math.round(this.value*100) + '%'; changeDataset(state.m4.dataset);">
                        </div>
                        <div class="flex flex-col gap-1 bg-slate-950/50 p-2 rounded-lg border border-slate-800">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Batch / Frame</span>
                                <span id="val-batch" class="text-purple-400 font-mono">${state.m4.batchSize}</span>
                            </div>
                            <input type="range" class="slider-custom" min="4" max="40" step="4" value="${state.m4.batchSize}" oninput="state.m4.batchSize = parseInt(this.value); document.getElementById('val-batch').innerText = this.value;">
                        </div>
                    </div>

                    <div class="flex flex-col gap-1">
                        <label class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Arquitectura Capas Ocultas</label>
                        <select id="net-struct" onchange="changeNetwork(this.value)" class="bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 outline-none focus:border-blue-500">
                            <option value="3">1 Capa (3 Neuronas)</option>
                            <option value="6">1 Capa (6 Neuronas)</option>
                            <option value="5,4" selected>2 Capas (5 → 4 Neuronas)</option>
                            <option value="6,6">2 Capas (6 → 6 Neuronas)</option>
                            <option value="7,6,4">3 Capas (7 → 6 → 4 Neuronas)</option>
                            <option value="8,8,6">3 Capas Profundas (8 → 8 → 6)</option>
                        </select>
                    </div>

                    <div class="flex flex-col gap-1">
                        <label class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Activación Capas Ocultas</label>
                        <select id="net-act" onchange="state.m4.activation = this.value; resetSandbox();" class="bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 outline-none focus:border-blue-500">
                            <option value="tanh" ${state.m4.activation === "tanh" ? "selected" : ""}>Tanh (Curvas Suaves Continuas)</option>
                            <option value="relu" ${state.m4.activation === "relu" ? "selected" : ""}>ReLU (Poligonal Estándar)</option>
                            <option value="leaky_relu" ${state.m4.activation === "leaky_relu" ? "selected" : ""}>Leaky ReLU (Evita Neuronas Muertas)</option>
                            <option value="sin" ${state.m4.activation === "sin" ? "selected" : ""}>Seno sin(z) (Ondas Periódicas SIREN)</option>
                            <option value="sigmoid" ${state.m4.activation === "sigmoid" ? "selected" : ""}>Sigmoide σ(z) (Clásica)</option>
                        </select>
                    </div>

                    <div class="grid grid-cols-2 gap-2">
                        <div class="flex flex-col gap-1 bg-slate-950/50 p-2 rounded-lg border border-slate-800">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">LR (η)</span>
                                <span id="val-lr" class="text-emerald-400 font-mono">${state.m4.lr}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0.01" max="0.5" step="0.01" value="${state.m4.lr}" oninput="state.m4.lr = parseFloat(this.value); sandboxNet.lr = state.m4.lr; document.getElementById('val-lr').innerText=this.value">
                        </div>
                        <div class="flex flex-col gap-1 bg-slate-950/50 p-2 rounded-lg border border-slate-800">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Reg. L2 (λ)</span>
                                <span id="val-reg" class="text-amber-400 font-mono">${state.m4.regL2.toFixed(3)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0" max="0.01" step="0.001" value="${state.m4.regL2}" oninput="state.m4.regL2 = parseFloat(this.value); sandboxNet.regL2 = state.m4.regL2; document.getElementById('val-reg').innerText=parseFloat(this.value).toFixed(3)">
                        </div>
                    </div>

                    <div class="flex items-center justify-between bg-slate-950/60 px-2.5 py-2 rounded-lg border border-slate-800 text-[11px]">
                        <label class="flex items-center gap-1.5 cursor-pointer text-slate-300">
                            <input type="checkbox" class="accent-blue-500 rounded" onchange="state.m4.showContour = this.checked;" ${state.m4.showContour ? "checked" : ""}>
                            <span>Frontera p=0.5</span>
                        </label>
                        <label class="flex items-center gap-1.5 cursor-pointer text-slate-300">
                            <input type="checkbox" class="accent-blue-500 rounded" onchange="state.m4.showWeights = this.checked;" ${state.m4.showWeights ? "checked" : ""}>
                            <span>Pulsos Sinápticos</span>
                        </label>
                    </div>

                    <div class="bg-slate-950/60 p-2 rounded-xl border border-slate-800 flex items-center justify-between">
                        <span class="text-[11px] text-slate-300">Clic en Mapa añade:</span>
                        <div class="flex gap-1">
                            <button id="btn-class-1" onclick="setAddClassMode(1)" class="px-2 py-1 rounded text-[10px] font-bold bg-blue-600 text-white">Azul (1)</button>
                            <button id="btn-class-0" onclick="setAddClassMode(0)" class="px-2 py-1 rounded text-[10px] font-bold bg-slate-800 text-rose-400">Rojo (0)</button>
                            <button onclick="clearCustomPoints()" class="px-2 py-1 rounded text-[10px] bg-slate-800 hover:bg-slate-700 text-slate-300" title="Regenerar muestra limpia"><i class="fas fa-broom"></i></button>
                        </div>
                    </div>
                    
                    <div class="flex gap-1.5 pt-0.5">
                        <button id="btn-train" onclick="toggleTraining()" class="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2 px-3 rounded-xl shadow-lg shadow-emerald-600/25 transition text-xs flex justify-center items-center gap-1.5">
                            <i class="fas fa-play"></i> Entrenar
                        </button>
                        <button onclick="stepSingleSandboxEpoch()" class="bg-blue-600/20 hover:bg-blue-600/40 text-blue-300 border border-blue-500/30 font-bold py-2 px-2.5 rounded-xl transition text-xs" title="Avanzar 1 Época Paso a Paso">
                            +1 Época
                        </button>
                        <button onclick="resetSandbox()" class="bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 px-2.5 rounded-xl transition text-xs border border-slate-700" title="Reiniciar Pesos">
                            <i class="fas fa-rotate-right"></i>
                        </button>
                    </div>
                `;
    renderModuleExplanation();
    document.getElementById("metrics-overlay").classList.remove("hidden");
    changeDataset(state.m4.dataset);
    changeNetwork(state.m4.hidden.join(","));
  } else if (modIndex === 5) {
    titleEl.innerText = "Descenso del Gradiente 3D";
    descEl.innerText =
      "Explora en 3D la superficie de la función de pérdida L(w₁, w₂) y cómo distintos optimizadores descienden hacia el mínimo.";
    hintText.innerText =
      "Arrastra con el ratón para rotar la superficie 3D · Haz clic sobre cualquier colina para lanzar la bola desde ahí";
    controlsEl.innerHTML = `
                    <div class="flex flex-col gap-1">
                        <label class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Topografía de Pérdida L(w₁, w₂)</label>
                        <select onchange="state.m5.surface = this.value; init3DBalls();" class="bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 outline-none focus:border-blue-500">
                            <option value="local_minima" ${state.m5.surface === "local_minima" ? "selected" : ""}>⛰️ Mínimos Locales y Global (No Convexo)</option>
                            <option value="bowl" ${state.m5.surface === "bowl" ? "selected" : ""}>🥣 Cuenco Cuadrático Convexo</option>
                            <option value="saddle" ${state.m5.surface === "saddle" ? "selected" : ""}>🐎 Punto de Silla (Saddle Point)</option>
                            <option value="ravine" ${state.m5.surface === "ravine" ? "selected" : ""}>🏞️ Cañón Estrecho (Oscilación SGD)</option>
                            <option value="rosenbrock" ${state.m5.surface === "rosenbrock" ? "selected" : ""}>🍌 Valle Curvo de Rosenbrock</option>
                        </select>
                    </div>

                    <div class="flex flex-col gap-1">
                        <label class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Optimizador</label>
                        <select onchange="state.m5.optimizer = this.value; init3DBalls();" class="bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 outline-none focus:border-blue-500">
                            <option value="compare_all" ${state.m5.optimizer === "compare_all" ? "selected" : ""}>🏁 Comparar Todos (SGD vs Momentum vs Adam)</option>
                            <option value="sgd" ${state.m5.optimizer === "sgd" ? "selected" : ""}>🔵 SGD Estándar (Descenso Puro)</option>
                            <option value="momentum" ${state.m5.optimizer === "momentum" ? "selected" : ""}>🟡 SGD + Momentum (Con Inercia Física)</option>
                            <option value="adam" ${state.m5.optimizer === "adam" ? "selected" : ""}>🟢 Adam (Paso Adaptativo Inteligente)</option>
                        </select>
                    </div>

                    <div class="grid grid-cols-2 gap-2">
                        <div class="flex flex-col gap-1 bg-slate-950/50 p-2 rounded-lg border border-slate-800">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Learning Rate</span>
                                <span id="val-m5-lr" class="text-emerald-400 font-mono">${state.m5.lr.toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0.01" max="0.20" step="0.01" value="${state.m5.lr}" oninput="state.m5.lr = parseFloat(this.value); document.getElementById('val-m5-lr').innerText=parseFloat(this.value).toFixed(2);">
                        </div>
                        <div class="flex flex-col gap-1 bg-slate-950/50 p-2 rounded-lg border border-slate-800">
                            <div class="flex justify-between text-[11px]">
                                <span class="text-slate-400">Inercia (β)</span>
                                <span id="val-m5-mom" class="text-amber-400 font-mono">${state.m5.momentum.toFixed(2)}</span>
                            </div>
                            <input type="range" class="slider-custom" min="0.50" max="0.98" step="0.02" value="${state.m5.momentum}" oninput="state.m5.momentum = parseFloat(this.value); document.getElementById('val-m5-mom').innerText=parseFloat(this.value).toFixed(2);">
                        </div>
                    </div>

                    <div class="bg-slate-950/70 p-2.5 rounded-xl border border-slate-800 space-y-1.5 font-mono text-[11px]">
                        <div class="flex justify-between"><span class="text-slate-400">PASO ACTUAL:</span><span id="m5-step-lbl" class="text-white font-bold">0</span></div>
                        <div class="flex justify-between"><span class="text-slate-400">PESOS (w₁, w₂):</span><span id="m5-pos-lbl" class="text-blue-400">(0.00, 0.00)</span></div>
                        <div class="flex justify-between"><span class="text-slate-400">PÉRDIDA L(w):</span><span id="m5-loss-lbl" class="text-rose-400 font-bold">0.0000</span></div>
                        <div class="flex justify-between"><span class="text-slate-400">NORMA ‖∇L‖:</span><span id="m5-grad-lbl" class="text-emerald-400">0.0000</span></div>
                    </div>

                    <div class="flex items-center justify-between bg-slate-950/60 px-2.5 py-2 rounded-lg border border-slate-800 text-[11px]">
                        <label class="flex items-center gap-1.5 cursor-pointer text-slate-300">
                            <input type="checkbox" class="accent-blue-500 rounded" onchange="state.m5.showContours = this.checked;" ${state.m5.showContours ? "checked" : ""}>
                            <span>Base de Contorno</span>
                        </label>
                        <label class="flex items-center gap-1.5 cursor-pointer text-slate-300">
                            <input type="checkbox" class="accent-blue-500 rounded" onchange="state.m5.showGradientVec = this.checked;" ${state.m5.showGradientVec ? "checked" : ""}>
                            <span>Vector −∇L</span>
                        </label>
                    </div>

                    <div class="grid grid-cols-2 gap-1.5">
                        <button onclick="init3DBalls(-1.65, 1.55)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-2 rounded-lg text-[11px] border border-slate-700">📍 Punto Inicial A</button>
                        <button onclick="init3DBalls(1.65, -1.45)" class="bg-slate-800 hover:bg-slate-700 text-slate-300 py-1.5 px-2 rounded-lg text-[11px] border border-slate-700">📍 Punto Inicial B</button>
                    </div>

                    <div class="flex gap-1.5 pt-1">
                        <button id="btn-m5-run" onclick="toggleM5Run()" class="flex-1 bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-2.5 px-3 rounded-xl shadow-lg shadow-emerald-600/25 transition text-xs flex justify-center items-center gap-1.5">
                            <i class="fas fa-play"></i> Descender
                        </button>
                        <button onclick="step3DOptimizers()" class="bg-blue-600/20 hover:bg-blue-600/40 text-blue-300 border border-blue-500/30 font-bold py-2.5 px-3 rounded-xl transition text-xs" title="Avanzar 1 Paso de Gradiente">
                            +1 Paso
                        </button>
                        <button onclick="init3DBalls()" class="bg-slate-800 hover:bg-slate-700 text-slate-200 py-2.5 px-3 rounded-xl transition text-xs border border-slate-700" title="Reiniciar Trayectoria">
                            <i class="fas fa-undo"></i>
                        </button>
                    </div>
                `;
    renderModuleExplanation();
    state.m5.optimizer = "compare_all";
    state.m5.isRunning = false;
    init3DBalls(-1.65, 1.55);
  } else if (modIndex === 6) {
    titleEl.innerText = "Visión Artificial (CNN)";
    descEl.innerText =
      "Observa cómo una Red Convolucional escanea una imagen con un filtro (Kernel 3×3), extrae bordes y reduce dimensionalidad con Max-Pooling.";
    hintText.innerText =
      "Pinta sobre la Imagen de Entrada (10×10) con el ratón o haz clic en cualquier celda del Mapa 8×8 para inspeccionar su cálculo";

    if (!state.m6.inputGrid.length) {
      loadCNNPreset("digit7");
    } else {
      recomputeCNN();
    }

    let kernelInputsHTML = "";
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        kernelInputsHTML += `
                            <input id="k-cell-${r}-${c}" type="number" step="0.5" value="${state.m6.kernel[r][c]}"
                                onchange="updateKernelCell(${r},${c},this.value)"
                                class="w-full bg-slate-900 border border-slate-700 rounded text-center font-mono text-xs text-amber-300 py-1 focus:border-amber-400 outline-none">
                        `;
      }
    }

    controlsEl.innerHTML = `
                    <div class="flex flex-col gap-1">
                        <label class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Patrón de Entrada (o Dibuja en el Lienzo)</label>
                        <div class="flex gap-1.5">
                            <select onchange="loadCNNPreset(this.value)" class="flex-1 bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 outline-none focus:border-blue-500">
                                <option value="digit7" ${state.m6.presetImg === "digit7" ? "selected" : ""}>🔢 Dígito "7"</option>
                                <option value="cross" ${state.m6.presetImg === "cross" ? "selected" : ""}>➕ Cruz Simétrica</option>
                                <option value="box" ${state.m6.presetImg === "box" ? "selected" : ""}>🔲 Contorno Cuadrado</option>
                                <option value="diag" ${state.m6.presetImg === "diag" ? "selected" : ""}>📏 Línea Diagonal</option>
                            </select>
                            <button onclick="clearCNNCanvas()" class="bg-slate-800 hover:bg-slate-700 text-slate-300 px-2.5 rounded-lg border border-slate-700 text-xs" title="Limpiar lienzo 10x10">
                                <i class="fas fa-eraser"></i>
                            </button>
                        </div>
                    </div>

                    <div class="flex flex-col gap-1">
                        <label class="text-[10px] text-slate-400 font-bold uppercase tracking-wider">Filtro Convolucional (Kernel 3×3)</label>
                        <select onchange="setCNNKernel(this.value)" class="bg-slate-950 border border-slate-700 text-white text-xs rounded-lg p-2 outline-none focus:border-blue-500">
                            <option value="edge_vert" ${state.m6.kernelType === "edge_vert" ? "selected" : ""}>┃ Detector Bordes Verticales (Sobel X)</option>
                            <option value="edge_horiz" ${state.m6.kernelType === "edge_horiz" ? "selected" : ""}>━ Detector Bordes Horizontales (Sobel Y)</option>
                            <option value="laplacian" ${state.m6.kernelType === "laplacian" ? "selected" : ""}>💠 Contorno Omnidireccional (Laplaciano)</option>
                            <option value="sharpen" ${state.m6.kernelType === "sharpen" ? "selected" : ""}>✨ Enfoque y Nitidez (Sharpen)</option>
                            <option value="emboss" ${state.m6.kernelType === "emboss" ? "selected" : ""}>🗿 Relieve Direccional (Emboss)</option>
                        </select>
                    </div>

                    <div class="bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 space-y-2">
                        <div class="flex justify-between items-center">
                            <span class="text-[10px] font-bold uppercase tracking-wider text-amber-400">Matriz de Pesos del Kernel</span>
                            <span class="text-[10px] text-slate-500">Editable</span>
                        </div>
                        <div class="grid grid-cols-3 gap-1.5">
                            ${kernelInputsHTML}
                        </div>
                    </div>

                    <div class="flex flex-col gap-2 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800 text-xs">
                        <label class="flex items-center justify-between cursor-pointer text-slate-300">
                            <span>Aplicar Activación <b>ReLU max(0, z)</b></span>
                            <input type="checkbox" class="accent-blue-500 w-4 h-4 rounded" onchange="state.m6.useRelu = this.checked; recomputeCNN();" ${state.m6.useRelu ? "checked" : ""}>
                        </label>
                        <label class="flex items-center justify-between cursor-pointer text-slate-300">
                            <span>Escaneo Automático de Ventana 3×3</span>
                            <input id="chk-cnn-autoscan" type="checkbox" class="accent-emerald-500 w-4 h-4 rounded" onchange="state.m6.autoScan = this.checked;" ${state.m6.autoScan ? "checked" : ""}>
                        </label>
                    </div>

                    <div class="flex gap-1.5">
                        <button onclick="stepCNNScanner(-1)" class="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-200 py-2 rounded-xl text-xs font-semibold border border-slate-700">
                            <i class="fas fa-chevron-left mr-1"></i> Anterior
                        </button>
                        <button onclick="stepCNNScanner(1)" class="flex-1 bg-blue-600/20 hover:bg-blue-600/40 text-blue-300 py-2 rounded-xl text-xs font-semibold border border-blue-500/30">
                            Siguiente <i class="fas fa-chevron-right ml-1"></i>
                        </button>
                    </div>
                `;
    renderModuleExplanation();
  } else if (modIndex === 7) {
    titleEl.innerText = "Clasificador multiclase";
    descEl.innerText =
      "Experimenta con una red que asigna una probabilidad a cada clase y elige la categoría más probable.";
    hintText.innerText =
      state.m7.interactionMode === "add"
        ? "Modo añadir: pulsa en el mapa para colocar un ejemplo etiquetado"
        : "Modo probar: pulsa en el mapa para cambiar x₁ y x₂";

    if (
      state.m7.weights.length !== state.m7.hiddenLayers + 1 ||
      state.m7.weights.at(-1)?.length !== state.m7.classCount
    ) {
      initializeClassifier();
    }

    controlsEl.innerHTML = `
        <div class="flex flex-col gap-2 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <label for="classifier-class-count" class="text-[10px] font-bold uppercase tracking-wider text-slate-400">Neuronas de salida / clases</label>
            <select id="classifier-class-count" onchange="setClassifierClassCount(this.value)" class="rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white outline-none focus:border-blue-500">
                ${[2, 3, 4, 5]
                  .map(
                    (count) =>
                      `<option value="${count}" ${state.m7.classCount === count ? "selected" : ""}>${count} clases</option>`,
                  )
                  .join("")}
            </select>
            <p class="text-[10px] leading-relaxed text-slate-400">Una neurona de salida aprende a reconocer cada categoría.</p>
            <div class="flex flex-wrap gap-x-3 gap-y-1">
                ${Array.from({ length: state.m7.classCount }, (_, index) => `<span class="flex items-center gap-1 text-[10px] text-slate-300"><span class="h-2 w-2 rounded-full" style="background:${classifierColors[index]}"></span>${classifierLetters[index]}</span>`).join("")}
            </div>
        </div>

        <div class="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <label for="classifier-dataset" class="text-[10px] font-bold uppercase tracking-wider text-amber-300">Dataset y edición de ejemplos</label>
            <select id="classifier-dataset" onchange="setClassifierDataset(this.value)" class="rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white outline-none focus:border-amber-500">
                <option value="blobs" ${state.m7.datasetType === "blobs" ? "selected" : ""}>Nubes (blobs)</option>
                <option value="circles" ${state.m7.datasetType === "circles" ? "selected" : ""}>Círculos concéntricos</option>
                <option value="spirals" ${state.m7.datasetType === "spirals" ? "selected" : ""}>Espirales</option>
                <option value="manual" ${state.m7.datasetType === "manual" ? "selected" : ""}>Datos manuales</option>
            </select>
            <div class="${state.m7.datasetType === "manual" ? "hidden" : "flex"} flex-col gap-1.5" id="classifier-noise-control">
                <div class="flex justify-between text-[11px] text-slate-300">
                    <label for="classifier-noise">Ruido / dispersión</label>
                    <span id="classifier-noise-value" class="font-mono text-amber-300">${state.m7.noise.toFixed(2)}</span>
                </div>
                <input id="classifier-noise" type="range" min="0" max="0.4" step="0.01" value="${state.m7.noise}" oninput="setClassifierNoise(this.value)" class="slider-custom">
            </div>
            <div class="${state.m7.datasetType === "manual" ? "flex" : "hidden"} flex-col gap-2" id="classifier-manual-controls">
                <label for="classifier-manual-class" class="text-[11px] text-slate-300">Etiqueta para los nuevos puntos</label>
                <select id="classifier-manual-class" onchange="setClassifierManualClass(this.value)" class="rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white outline-none">
                    ${Array.from({ length: state.m7.classCount }, (_, index) => `<option value="${index}" ${state.m7.manualClass === index ? "selected" : ""}>Clase ${classifierLetters[index]}</option>`).join("")}
                </select>
                <p id="classifier-map-hint" class="text-[10px] leading-relaxed text-slate-400">${state.m7.interactionMode === "add" ? `Modo añadir: pulsa el mapa para colocar un punto de clase ${classifierLetters[state.m7.manualClass]}.` : "Pulsa el mapa para probar una entrada; cambia a modo añadir para etiquetar ejemplos."}</p>
            </div>
            <div class="flex gap-1.5">
                <button id="classifier-mode-probe" onclick="setClassifierInteractionMode('probe')" class="flex-1 rounded-lg border ${state.m7.interactionMode === "probe" ? "border-blue-500/50 bg-blue-600/20 text-blue-200" : "border-slate-700 bg-slate-800 text-slate-300"} px-2 py-1.5 text-[10px] font-semibold">Probar entrada</button>
                <button id="classifier-mode-add" onclick="setClassifierInteractionMode('add')" class="flex-1 rounded-lg border ${state.m7.interactionMode === "add" ? "border-amber-500/50 bg-amber-600/20 text-amber-200" : "border-slate-700 bg-slate-800 text-slate-300"} px-2 py-1.5 text-[10px] font-semibold">Añadir punto</button>
            </div>
            <button onclick="clearClassifierDataset()" class="rounded-lg border border-slate-700 bg-slate-800 px-2 py-1.5 text-[10px] font-semibold text-slate-300 transition hover:bg-slate-700">
                <i class="fas fa-eraser mr-1"></i> Borrar ejemplos
            </button>
        </div>

        <div class="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <h3 class="text-[10px] font-bold uppercase tracking-wider text-purple-300">Arquitectura de la red</h3>
            <div class="flex items-center justify-between gap-2">
                <label for="classifier-hidden-layers" class="text-[11px] text-slate-300">Capas ocultas</label>
                <select id="classifier-hidden-layers" onchange="setClassifierArchitecture(this.value, document.getElementById('classifier-neurons').value)" class="rounded-lg border border-slate-700 bg-slate-950 p-1.5 text-xs text-white">
                    ${[1, 2, 3].map((count) => `<option value="${count}" ${state.m7.hiddenLayers === count ? "selected" : ""}>${count}</option>`).join("")}
                </select>
            </div>
            <div class="flex items-center justify-between gap-2">
                <label for="classifier-neurons" class="text-[11px] text-slate-300">Neuronas por capa</label>
                <select id="classifier-neurons" onchange="setClassifierArchitecture(document.getElementById('classifier-hidden-layers').value, this.value)" class="rounded-lg border border-slate-700 bg-slate-950 p-1.5 text-xs text-white">
                    ${[4, 8, 12].map((count) => `<option value="${count}" ${state.m7.neuronsPerLayer === count ? "selected" : ""}>${count}</option>`).join("")}
                </select>
            </div>
            <p class="text-[10px] leading-relaxed text-slate-400">ReLU en capas ocultas · Softmax en la salida</p>
        </div>

        <div class="flex flex-col gap-2 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <label for="classifier-optimizer" class="text-[10px] font-bold uppercase tracking-wider text-emerald-300">Algoritmo de entrenamiento</label>
            <select id="classifier-optimizer" onchange="state.m7.optimizer = this.value; renderClassifierResults();" class="rounded-lg border border-slate-700 bg-slate-950 p-2 text-xs text-white outline-none focus:border-emerald-500">
                <option value="batch" ${state.m7.optimizer === "batch" ? "selected" : ""}>Batch GD · dataset completo</option>
                <option value="sgd" ${state.m7.optimizer === "sgd" ? "selected" : ""}>SGD · un ejemplo por actualización</option>
                <option value="minibatch" ${state.m7.optimizer === "minibatch" ? "selected" : ""}>Mini-batch · grupos de 16</option>
            </select>
            <p class="text-[10px] leading-relaxed text-slate-400">El algoritmo cambia el tamaño del lote usado en cada actualización.</p>
        </div>

        <div class="flex flex-col gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <h3 class="text-[10px] font-bold uppercase tracking-wider text-blue-300">Prueba una entrada (x₁, x₂)</h3>
            ${[
              ["x1", "x₁ · característica 1"],
              ["x2", "x₂ · característica 2"],
            ]
              .map(
                ([axis, label]) => `
                    <div class="flex flex-col gap-1">
                        <div class="flex justify-between text-[11px] text-slate-300">
                            <label for="classifier-${axis}">${label}</label>
                            <span id="classifier-${axis}-value" class="font-mono text-blue-300">${state.m7[axis].toFixed(2)}</span>
                        </div>
                        <input id="classifier-${axis}" type="range" min="-1" max="1" step="0.01" value="${state.m7[axis]}" oninput="setClassifierInput('${axis}', this.value)" class="slider-custom">
                    </div>`,
              )
              .join("")}
        </div>

        <div class="flex flex-col gap-2 rounded-xl border border-slate-800 bg-slate-950/60 p-3">
            <div class="flex items-center justify-between">
                <h3 class="text-[10px] font-bold uppercase tracking-wider text-emerald-300">Predicción · Softmax</h3>
                <span class="text-[9px] text-slate-500">Σ probabilidades = 100 %</span>
            </div>
            <p id="classifier-prediction" class="text-sm font-bold text-white"></p>
            <div id="classifier-results" class="flex flex-col gap-2"></div>
        </div>

        <div class="flex flex-col gap-2">
            <p id="classifier-training-status" class="text-[10px] leading-relaxed text-slate-400"></p>
            <button onclick="trainClassifier()" class="flex items-center justify-center gap-2 rounded-xl bg-emerald-600 px-3 py-2.5 text-xs font-bold text-white shadow-lg shadow-emerald-600/20 transition hover:bg-emerald-500">
                <i class="fas fa-play"></i> Entrenar clasificador
            </button>
            <button onclick="resetClassifier()" class="rounded-xl border border-slate-700 bg-slate-800 px-3 py-2 text-xs font-semibold text-slate-300 transition hover:bg-slate-700">
                <i class="fas fa-undo mr-1"></i> Reiniciar pesos
            </button>
        </div>
        <p class="rounded-lg border border-blue-500/20 bg-blue-500/5 p-2.5 text-[10px] leading-relaxed text-slate-400">
            <i class="fas fa-hand-pointer mr-1 text-blue-300"></i> En el mapa, usa <b>Probar entrada</b> para mover el marcador o <b>Añadir punto</b> para crear un ejemplo con la clase elegida.
        </p>
    `;
    renderClassifierResults();
    renderModuleExplanation();
  }
}

function stepCNNScanner(dir) {
  state.m6.autoScan = false;
  const chk = document.getElementById("chk-cnn-autoscan");
  if (chk) chk.checked = false;
  let idx = state.m6.scanY * 8 + state.m6.scanX + dir;
  if (idx < 0) idx = 63;
  if (idx >= 64) idx = 0;
  state.m6.scanY = Math.floor(idx / 8);
  state.m6.scanX = idx % 8;
}

function clearCustomPoints() {
  changeDataset(state.m4.dataset);
}

function stepSingleSandboxEpoch() {
  if (!currentData.length) return;
  currentData.sort(() => Math.random() - 0.5);
  for (let item of currentData) {
    sandboxNet.forward(item.x);
    sandboxNet.backward(item.y);
  }
  state.m4.epoch++;
  evaluateSandboxMetrics();
  state.m4.lossHistory.push(state.m4.loss);
  if (state.m4.lossHistory.length > 60) state.m4.lossHistory.shift();
  updateMetricsUI();
}

function toggleM5Run() {
  state.m5.isRunning = !state.m5.isRunning;
  const btn = document.getElementById("btn-m5-run");
  if (!btn) return;
  if (state.m5.isRunning) {
    btn.innerHTML = '<i class="fas fa-pause"></i> Pausar';
    btn.classList.replace("bg-emerald-600", "bg-amber-600");
    btn.classList.replace("hover:bg-emerald-500", "hover:bg-amber-500");
  } else {
    btn.innerHTML = '<i class="fas fa-play"></i> Descender';
    btn.classList.replace("bg-amber-600", "bg-emerald-600");
    btn.classList.replace("hover:bg-amber-500", "hover:bg-emerald-500");
  }
}

function setAddClassMode(cls) {
  state.m4.addClassMode = cls;
  const b1 = document.getElementById("btn-class-1");
  const b0 = document.getElementById("btn-class-0");
  if (cls === 1) {
    b1.className =
      "px-2.5 py-1 rounded text-[11px] font-bold bg-blue-600 text-white";
    b0.className =
      "px-2.5 py-1 rounded text-[11px] font-bold bg-slate-800 text-rose-400";
  } else {
    b0.className =
      "px-2.5 py-1 rounded text-[11px] font-bold bg-rose-600 text-white";
    b1.className =
      "px-2.5 py-1 rounded text-[11px] font-bold bg-slate-800 text-blue-400";
  }
}

function updateState(mod, prop, val) {
  state[mod][prop] = parseFloat(val);
}

function solveXOR() {
  state.m1.solvedXor = true;
  document.getElementById("explanation-content").innerHTML =
    "<span class='text-emerald-400 font-bold'>¡Red Multicapa Activada (2→2→1)!</span> La neurona oculta <span class='font-mono text-purple-300'>h₁</span> activa la frontera superior-derecha y <span class='font-mono text-purple-300'>h₂</span> la inferior-izquierda. La neurona de salida las combina aislando el canal central azul con un <strong>100% de exactitud (4/4)</strong>.";
}

function changeDataset(type) {
  state.m4.dataset = type;
  currentData = generateData(type);
  evaluateSandboxMetrics();
}

function changeNetwork(structStr) {
  const hidden = structStr.split(",").map(Number);
  state.m4.hidden = hidden;
  sandboxNet = new SimpleMLP(
    [2, ...hidden, 1],
    state.m4.lr,
    state.m4.activation,
    state.m4.regL2,
  );
  state.m4.epoch = 0;
  state.m4.lossHistory = [];
  evaluateSandboxMetrics();
}

function resetSandbox() {
  const structEl = document.getElementById("net-struct");
  changeNetwork(structEl ? structEl.value : "5,4");
}

function toggleTraining() {
  state.m4.isTraining = !state.m4.isTraining;
  const btn = document.getElementById("btn-train");
  const dot = document.getElementById("lbl-status-dot");
  if (state.m4.isTraining) {
    btn.innerHTML = '<i class="fas fa-pause"></i> Pausar';
    btn.classList.replace("bg-emerald-600", "bg-amber-600");
    btn.classList.replace("hover:bg-emerald-500", "hover:bg-amber-500");
    if (dot) dot.className = "w-2 h-2 rounded-full bg-emerald-400 animate-ping";
  } else {
    btn.innerHTML = '<i class="fas fa-play"></i> Continuar';
    btn.classList.replace("bg-amber-600", "bg-emerald-600");
    btn.classList.replace("hover:bg-amber-500", "hover:bg-emerald-500");
    if (dot) dot.className = "w-2 h-2 rounded-full bg-amber-400";
  }
}

function evaluateSandboxMetrics() {
  if (!currentData.length) return;
  let totalLoss = 0;
  let correct = 0;
  for (let item of currentData) {
    const out = sandboxNet.forward(item.x)[0];
    totalLoss += Math.pow(out - item.y[0], 2);
    const pred = out >= 0.5 ? 1 : 0;
    if (pred === item.y[0]) correct++;
  }
  state.m4.loss = totalLoss / currentData.length;
  state.m4.accuracy = Math.round((correct / currentData.length) * 100);
  updateMetricsUI();
}

function updateMetricsUI() {
  const epEl = document.getElementById("lbl-epoch");
  const lsEl = document.getElementById("lbl-loss");
  const acEl = document.getElementById("lbl-acc");
  if (epEl) epEl.innerText = state.m4.epoch;
  if (lsEl) lsEl.innerText = state.m4.loss.toFixed(4);
  if (acEl) acEl.innerText = `${state.m4.accuracy}%`;

  // Dibujar mini gráfico de Loss
  if (lossChartCtx && state.m4.lossHistory.length > 1) {
    const cw = lossChartCanvas.width;
    const ch = lossChartCanvas.height;
    lossChartCtx.clearRect(0, 0, cw, ch);

    const maxLoss = Math.max(0.35, ...state.m4.lossHistory);
    lossChartCtx.beginPath();
    state.m4.lossHistory.forEach((l, idx) => {
      const x = (idx / (state.m4.lossHistory.length - 1)) * cw;
      const y = ch - (l / maxLoss) * (ch - 8) - 4;
      if (idx === 0) lossChartCtx.moveTo(x, y);
      else lossChartCtx.lineTo(x, y);
    });
    lossChartCtx.strokeStyle = "#fb7185";
    lossChartCtx.lineWidth = 2;
    lossChartCtx.stroke();
  }
}

function bpAction(action) {
  const m = state.m3;
  m.waitFrames = 0;

  if (action === "reset") {
    m.runMode = "paused";
    m.phase = 1;
    m.progress = 1;
    m.epoch = 0;
    m.w_i_h = [
      [0.5, -0.3],
      [0.8, 0.2],
    ];
    m.b_h = [0.1, -0.1];
    m.w_h_o = [0.9, -0.6];
    m.b_o = -0.2;
    m.lossHistory = [];
    m3Compute();
    m.lossHistory.push(m.fwd.loss);
  } else if (action === "play-pause") {
    if (m.runMode === "playing") {
      m.runMode = "paused";
    } else {
      if (m.phase === 0) m.phase = 1;
      if (m.progress >= 1) m.progress = 0;
      m.runMode = "playing";
    }
  } else if (action === "step-fwd") {
    m.runMode = "step";
    if (m.phase === 0) {
      m.phase = 1;
      m.progress = 0;
    } else if (m.progress >= 1) {
      m.phase = m.phase === 4 ? 1 : m.phase + 1;
      if (m.phase === 1) m3ApplyWeights();
      m.progress = 0;
    }
  } else if (action === "step-back") {
    m.runMode = "paused";
    if (m.phase > 1) {
      m.phase--;
      m.progress = 1;
    } else {
      m.phase = 1;
      m.progress = 0;
    }
  }
  document.getElementById("btn-play").innerHTML =
    m.runMode === "playing"
      ? '<i class="fas fa-pause text-xs"></i>'
      : '<i class="fas fa-play text-xs"></i>';
}

// ==========================================
// UTILIDADES DE DIBUJO EN CANVAS
// ==========================================
function drawTechnicalBackgroundGrid() {
  const step = 40;
  const topLeft = screenToWorld(0, 0);
  const bottomRight = screenToWorld(width, height);

  const startX = Math.floor(topLeft.x / step) * step;
  const endX = Math.ceil(bottomRight.x / step) * step;
  const startY = Math.floor(topLeft.y / step) * step;
  const endY = Math.ceil(bottomRight.y / step) * step;

  ctx.lineWidth = 1 / camera.zoom;
  ctx.strokeStyle = "rgba(51, 65, 85, 0.22)";
  ctx.beginPath();
  for (let x = startX; x <= endX; x += step) {
    ctx.moveTo(x, startY);
    ctx.lineTo(x, endY);
  }
  for (let y = startY; y <= endY; y += step) {
    ctx.moveTo(startX, y);
    ctx.lineTo(endX, y);
  }
  ctx.stroke();
}

function drawBezierSynapse(
  x1,
  y1,
  x2,
  y2,
  weight,
  animSpeed = 1,
  colorOverride = null,
) {
  const cpOffset = Math.abs(x2 - x1) * 0.45;
  const cp1x = x1 + cpOffset;
  const cp1y = y1;
  const cp2x = x2 - cpOffset;
  const cp2y = y2;

  const isPos = weight >= 0;
  const absW = Math.abs(weight);
  const baseColor = colorOverride || (isPos ? "#10b981" : "#f43f5e");

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, x2, y2);
  ctx.strokeStyle = baseColor;
  ctx.globalAlpha = Math.min(0.9, 0.25 + absW * 0.25);
  ctx.lineWidth = Math.max(1.2, Math.min(7, absW * 2.2));
  ctx.stroke();

  // Pulso animado viajando por la curva Bezier
  if (absW > 0.05 && animSpeed > 0) {
    const t = (state.time * 0.012 * animSpeed) % 1;
    const mt = 1 - t;
    const px =
      mt * mt * mt * x1 +
      3 * mt * mt * t * cp1x +
      3 * mt * t * t * cp2x +
      t * t * t * x2;
    const py =
      mt * mt * mt * y1 +
      3 * mt * mt * t * cp1y +
      3 * mt * t * t * cp2y +
      t * t * t * y2;

    ctx.globalAlpha = 0.95;
    ctx.fillStyle = isPos ? "#6ee7b7" : "#fda4af";
    ctx.beginPath();
    ctx.arc(px, py, 3.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function drawPillLabel(
  text,
  x,
  y,
  bgColor = "rgba(15, 23, 42, 0.9)",
  borderColor = "#334155",
  textColor = "#e2e8f0",
  fontSize = 11,
) {
  ctx.save();
  ctx.font = `600 ${fontSize}px 'JetBrains Mono', monospace`;
  const metrics = ctx.measureText(text);
  const pw = metrics.width + 14;
  const ph = fontSize + 10;
  ctx.fillStyle = bgColor;
  ctx.strokeStyle = borderColor;
  ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.roundRect(x - pw / 2, y - ph / 2, pw, ph, 6);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = textColor;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawModule7() {
  const m = state.m7;
  const plot = getClassifierPlotGeometry();
  const cellSize = plot.size / 30;

  ctx.save();
  ctx.fillStyle = "#0f172a";
  ctx.fillRect(plot.x, plot.y, plot.size, plot.size);
  ctx.globalAlpha = 0.2;
  for (let row = 0; row < 30; row++) {
    for (let col = 0; col < 30; col++) {
      const x1 = ((col + 0.5) / 30) * 2 - 1;
      const x2 = 1 - ((row + 0.5) / 30) * 2;
      const probabilities = classifierForward([x1, x2]).probabilities;
      const classIndex = probabilities.indexOf(Math.max(...probabilities));
      ctx.fillStyle = classifierColors[classIndex];
      ctx.fillRect(
        plot.x + col * cellSize,
        plot.y + row * cellSize,
        cellSize + 0.5,
        cellSize + 0.5,
      );
    }
  }
  ctx.globalAlpha = 1;

  ctx.strokeStyle = "rgba(148, 163, 184, 0.28)";
  ctx.lineWidth = 1;
  for (let tick = 0; tick <= 4; tick++) {
    const offset = (plot.size * tick) / 4;
    ctx.beginPath();
    ctx.moveTo(plot.x + offset, plot.y);
    ctx.lineTo(plot.x + offset, plot.y + plot.size);
    ctx.moveTo(plot.x, plot.y + offset);
    ctx.lineTo(plot.x + plot.size, plot.y + offset);
    ctx.stroke();
  }
  ctx.strokeStyle = "#94a3b8";
  ctx.lineWidth = 1.5;
  ctx.strokeRect(plot.x, plot.y, plot.size, plot.size);

  for (const sample of m.dataset) {
    const x = plot.x + ((sample.x[0] + 1) / 2) * plot.size;
    const y = plot.y + ((1 - sample.x[1]) / 2) * plot.size;
    ctx.fillStyle = classifierColors[sample.label];
    if (sample.manual) {
      const markerSize = Math.max(5, plot.size * 0.014);
      ctx.fillRect(x - markerSize / 2, y - markerSize / 2, markerSize, markerSize);
    } else {
      ctx.beginPath();
      ctx.arc(x, y, Math.max(2, plot.size * 0.007), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.strokeStyle = "rgba(15, 23, 42, 0.7)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  const pointX = plot.x + ((m.x1 + 1) / 2) * plot.size;
  const pointY = plot.y + ((1 - m.x2) / 2) * plot.size;
  const probabilities = getClassifierProbabilities();
  const prediction = probabilities.indexOf(Math.max(...probabilities));
  ctx.beginPath();
  ctx.arc(pointX, pointY, Math.max(8, plot.size * 0.025), 0, Math.PI * 2);
  ctx.fillStyle = "#f8fafc";
  ctx.fill();
  ctx.lineWidth = 3;
  ctx.strokeStyle = classifierColors[prediction];
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(pointX, pointY, Math.max(3, plot.size * 0.01), 0, Math.PI * 2);
  ctx.fillStyle = classifierColors[prediction];
  ctx.fill();
  ctx.restore();

  drawPillLabel(
    m.interactionMode === "add"
      ? `AÑADIR PUNTO · Clase ${classifierLetters[m.manualClass]}`
      : "MAPA DE DECISIÓN · Pulsa para probar una entrada",
    plot.x + plot.size / 2,
    plot.y + 16,
    "rgba(15, 23, 42, 0.95)",
    "#334155",
    "#e2e8f0",
    10,
  );
  ctx.save();
  ctx.font = "10px 'JetBrains Mono', monospace";
  ctx.fillStyle = "#94a3b8";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("x₁", plot.x + plot.size / 2, plot.y + plot.size + 7);
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  ctx.fillText("x₂", plot.x - 8, plot.y + plot.size / 2);
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (let tick = 0; tick <= 4; tick++) {
    const value = (-1 + tick * 0.5).toFixed(1);
    ctx.fillText(value, plot.x + (plot.size * tick) / 4, plot.y + plot.size + 20);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(value, plot.x - 8, plot.y + (plot.size * tick) / 4);
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
  }
  ctx.restore();

  if (width > 850) {
    const outputX = width - 75;
    const firstLayerX = plot.x + plot.size + 72;
    const lastLayerX = outputX - 55;
    const layerGap =
      (lastLayerX - firstLayerX) / Math.max(1, m.hiddenLayers + 1);
    const centerY = height / 2;
    const layerSizes = [
      2,
      ...Array(m.hiddenLayers).fill(m.neuronsPerLayer),
      m.classCount,
    ];
    const activations = classifierForward([m.x1, m.x2]).activations;
    const nodePositions = layerSizes.map((count, layerIndex) => {
      const x =
        layerIndex === 0
          ? firstLayerX
          : layerIndex === layerSizes.length - 1
            ? outputX
            : firstLayerX + layerGap * layerIndex;
      const verticalStep = Math.min(34, (height - 96) / Math.max(1, count));
      return Array.from({ length: count }, (_, nodeIndex) => ({
        x,
        y: centerY + (nodeIndex - (count - 1) / 2) * verticalStep,
      }));
    });

    for (let layerIndex = 0; layerIndex < m.weights.length; layerIndex++) {
      for (
        let fromIndex = 0;
        fromIndex < nodePositions[layerIndex].length;
        fromIndex++
      ) {
        for (
          let toIndex = 0;
          toIndex < nodePositions[layerIndex + 1].length;
          toIndex++
        ) {
          const from = nodePositions[layerIndex][fromIndex];
          const to = nodePositions[layerIndex + 1][toIndex];
          const weight = m.weights[layerIndex][toIndex][fromIndex];
          ctx.beginPath();
          ctx.moveTo(from.x + 9, from.y);
          ctx.lineTo(to.x - 9, to.y);
          ctx.strokeStyle =
            weight >= 0 ? "rgba(52, 211, 153, 0.36)" : "rgba(251, 113, 133, 0.36)";
          ctx.lineWidth = 0.6 + Math.min(1.5, Math.abs(weight) * 0.45);
          ctx.stroke();
        }
      }
    }

    for (let layerIndex = 0; layerIndex < nodePositions.length; layerIndex++) {
      const isOutput = layerIndex === nodePositions.length - 1;
      const isInput = layerIndex === 0;
      const positions = nodePositions[layerIndex];
      const radius =
        positions.length > 8
          ? Math.max(6, Math.min(9, (height - 96) / positions.length / 2.5))
          : 12;
      ctx.fillStyle = "#94a3b8";
      ctx.font = "9px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillText(
        isInput
          ? "ENTRADA"
          : isOutput
            ? "SALIDA · SOFTMAX"
            : `OCULTA ${layerIndex}`,
        positions[0].x,
        Math.max(4, positions[0].y - (positions.length - 1) * 12 - radius - 17),
      );

      for (let nodeIndex = 0; nodeIndex < positions.length; nodeIndex++) {
        const point = positions[nodeIndex];
        const color = isInput
          ? "#60a5fa"
          : isOutput
            ? classifierColors[nodeIndex]
            : "#a78bfa";
        ctx.beginPath();
        ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = "#0f172a";
        ctx.fill();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = color;
        ctx.font = `${radius < 9 ? 7 : 9}px 'JetBrains Mono', monospace`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        const nodeLabel = isInput
          ? nodeIndex === 0
            ? "x₁"
            : "x₂"
          : isOutput
            ? classifierLetters[nodeIndex]
            : `${nodeIndex + 1}`;
        ctx.fillText(nodeLabel, point.x, point.y);

        if (isOutput) {
          ctx.fillStyle = "#cbd5e1";
          ctx.font = "9px Inter, sans-serif";
          ctx.textAlign = "left";
          ctx.fillText(
            `${(activations[layerIndex][nodeIndex] * 100).toFixed(0)}%`,
            point.x + radius + 4,
            point.y,
          );
        }
      }
    }
  }
}

// ==========================================
// RENDERIZADO PRINCIPAL
// ==========================================
function draw() {
  if (!ctx) return;
  ctx.clearRect(0, 0, width, height);

  // Interpolación suave de cámara
  camera.zoom += (camera.targetZoom - camera.zoom) * 0.22;
  camera.x += (camera.targetX - camera.x) * 0.22;
  camera.y += (camera.targetY - camera.y) * 0.22;

  ctx.save();
  const cx = width / 2;
  const cy = height / 2;
  ctx.translate(cx + camera.x, cy + camera.y);
  ctx.scale(camera.zoom, camera.zoom);
  ctx.translate(-cx, -cy);

  // Cuadrícula técnica de fondo sensible al Pan & Zoom
  drawTechnicalBackgroundGrid();

  if (state.module === 1 || state.module === 2) {
    drawModule1_2();
  } else if (state.module === 3) {
    drawModule3();
  } else if (state.module === 4) {
    drawModule4();
  } else if (state.module === 5) {
    drawModule5();
  } else if (state.module === 6) {
    drawModule6();
  } else if (state.module === 7) {
    drawModule7();
  }

  ctx.restore();

  // Dibujar Tooltip del Inspector en espacio de pantalla para que siempre tenga nitidez 1:1
  if (state.module === 1 && state.m1.showHelp && activeHoverTarget) {
    drawScreenInspectorTooltip(activeHoverTarget);
  }
}

let activeHoverTarget = null;

// ==========================================
// MÓDULO 1 Y 2: NEURONA Y DESAFÍO XOR
// ==========================================
function drawModule1_2() {
  const { w1, w2, b, x1, x2, actFn, solvedXor, showHelp } = state.m1;
  const { cx, cy, scale } = getPlotGeometryM1();
  const nx = width * 0.25;
  const ny = height * 0.5;

  activeHoverTarget = null;

  // --- PANEL IZQUIERDO: DIAGRAMA SINÁPTICO ---
  if (state.module === 2 && solvedXor) {
    // Dibujar Red Multicapa 2 -> 2 -> 1 que resuelve XOR
    const in1 = { x: nx - 135, y: ny - 65 };
    const in2 = { x: nx - 135, y: ny + 65 };
    const h1 = { x: nx - 10, y: ny - 65 };
    const h2 = { x: nx - 10, y: ny + 65 };
    const outN = { x: nx + 115, y: ny };

    drawBezierSynapse(in1.x, in1.y, h1.x, h1.y, 2.0, 1);
    drawBezierSynapse(in2.x, in2.y, h1.x, h1.y, 2.0, 1);
    drawBezierSynapse(in1.x, in1.y, h2.x, h2.y, -2.0, 1);
    drawBezierSynapse(in2.x, in2.y, h2.x, h2.y, -2.0, 1);
    drawBezierSynapse(h1.x, h1.y, outN.x, outN.y, 2.2, 1);
    drawBezierSynapse(h2.x, h2.y, outN.x, outN.y, 2.2, 1);

    const drawMiniNode = (pt, label, sub, borderCol) => {
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 24, 0, Math.PI * 2);
      ctx.fillStyle = "#0f172a";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = borderCol;
      ctx.stroke();
      ctx.fillStyle = "#f8fafc";
      ctx.font = "bold 12px Inter";
      ctx.textAlign = "center";
      ctx.fillText(label, pt.x, pt.y - 2);
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px JetBrains Mono";
      ctx.fillText(sub, pt.x, pt.y + 12);
    };

    drawMiniNode(in1, "x₁", "In 1", "#3b82f6");
    drawMiniNode(in2, "x₂", "In 2", "#3b82f6");
    drawMiniNode(h1, "h₁", "OR", "#a855f7");
    drawMiniNode(h2, "h₂", "NAND", "#a855f7");
    drawMiniNode(outN, "y", "AND", "#10b981");

    drawPillLabel(
      "Arquitectura Multicapa 2 → 2 → 1",
      nx - 10,
      ny - 130,
      "rgba(88, 28, 135, 0.4)",
      "#a855f7",
      "#e9d5ff",
      12,
    );
  } else {
    // Dibujar Neurona Unitaria de Alta Definición
    const in1X = nx - 145,
      in1Y = ny - 65;
    const in2X = nx - 145,
      in2Y = ny + 65;
    const biasX = nx,
      biasY = ny + 108;
    const outX = nx + 145,
      outY = ny;

    // Sinapsis Curvas con Pulsos
    drawBezierSynapse(in1X + 26, in1Y, nx - 44, ny - 12, w1, Math.abs(x1));
    drawBezierSynapse(in2X + 26, in2Y, nx - 44, ny + 12, w2, Math.abs(x2));

    // Conexión del Bias
    ctx.beginPath();
    ctx.moveTo(biasX, biasY - 20);
    ctx.lineTo(nx, ny + 46);
    ctx.strokeStyle = "#f59e0b";
    ctx.lineWidth = 2;
    ctx.setLineDash([4, 3]);
    ctx.stroke();
    ctx.setLineDash([]);

    // Cálculo Matemático
    const z = w1 * x1 + w2 * x2 + b;
    const out = evalActivation(z, actFn);

    // Línea de Salida
    drawBezierSynapse(
      nx + 46,
      ny,
      outX - 30,
      outY,
      out > 0 ? 1.8 : -0.8,
      1.2,
      out > 0.5 ? "#3b82f6" : "#64748b",
    );

    // Píldoras de Pesos w1 y w2
    drawPillLabel(
      `w₁ = ${w1.toFixed(2)}`,
      nx - 82,
      ny - 45,
      "rgba(15,23,42,0.92)",
      w1 >= 0 ? "#10b981" : "#f43f5e",
      w1 >= 0 ? "#6ee7b7" : "#fda4af",
      11,
    );
    drawPillLabel(
      `w₂ = ${w2.toFixed(2)}`,
      nx - 82,
      ny + 45,
      "rgba(15,23,42,0.92)",
      w2 >= 0 ? "#10b981" : "#f43f5e",
      w2 >= 0 ? "#6ee7b7" : "#fda4af",
      11,
    );

    // Nodos de Entrada x1 y x2
    const drawInputNode = (ix, iy, lbl, val) => {
      ctx.beginPath();
      ctx.arc(ix, iy, 26, 0, Math.PI * 2);
      ctx.fillStyle = "#0f172a";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#3b82f6";
      ctx.stroke();
      ctx.fillStyle = "#93c5fd";
      ctx.font = "bold 12px Inter";
      ctx.textAlign = "center";
      ctx.fillText(lbl, ix, iy - 4);
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 11px JetBrains Mono";
      ctx.fillText(val.toFixed(2), ix, iy + 12);
    };
    drawInputNode(in1X, in1Y, "x₁", x1);
    drawInputNode(in2X, in2Y, "x₂", x2);

    // Nodo de Bias
    drawPillLabel(
      `Sesgo b = ${b.toFixed(2)}`,
      biasX,
      biasY,
      "rgba(30, 41, 59, 0.95)",
      "#f59e0b",
      "#fcd34d",
      11,
    );

    // Cuerpo Principal de la Neurona (Doble Cámara: Suma Σ | Activación f)
    const nRadius = 46;
    ctx.save();
    ctx.shadowColor =
      out > 0.5 ? "rgba(59, 130, 246, 0.45)" : "rgba(15, 23, 42, 0.8)";
    ctx.shadowBlur = 22;

    // Mitad Izquierda (Suma Ponderada)
    ctx.beginPath();
    ctx.arc(nx, ny, nRadius, Math.PI / 2, Math.PI * 1.5);
    ctx.fillStyle = "#0f172a";
    ctx.fill();

    // Mitad Derecha (Función de Activación)
    ctx.beginPath();
    ctx.arc(nx, ny, nRadius, -Math.PI / 2, Math.PI / 2);
    ctx.fillStyle =
      out > 0.5 ? "rgba(37, 99, 235, 0.28)" : "rgba(30, 41, 59, 0.7)";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(nx, ny, nRadius, 0, Math.PI * 2);
    ctx.lineWidth = 3;
    ctx.strokeStyle = out > 0.5 ? "#60a5fa" : "#475569";
    ctx.stroke();
    ctx.restore();

    // Línea divisoria interna
    ctx.beginPath();
    ctx.moveTo(nx, ny - nRadius);
    ctx.lineTo(nx, ny + nRadius);
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Texto Σ y valor z
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "bold 16px Inter";
    ctx.textAlign = "center";
    ctx.fillText("Σ", nx - 22, ny - 6);
    ctx.fillStyle = "#34d399";
    ctx.font = "bold 10px JetBrains Mono";
    ctx.fillText(`z=${z.toFixed(1)}`, nx - 22, ny + 12);

    // Mini gráfico de la función de activación dentro de la mitad derecha
    ctx.save();
    ctx.strokeStyle = "#93c5fd";
    ctx.lineWidth = 2;
    ctx.beginPath();
    const gx = nx + 21,
      gy = ny;
    if (actFn === "step") {
      ctx.moveTo(gx - 10, gy + 8);
      ctx.lineTo(gx, gy + 8);
      ctx.lineTo(gx, gy - 8);
      ctx.lineTo(gx + 10, gy - 8);
    } else if (actFn === "relu") {
      ctx.moveTo(gx - 10, gy + 6);
      ctx.lineTo(gx, gy + 6);
      ctx.lineTo(gx + 10, gy - 9);
    } else {
      for (let i = -10; i <= 10; i++) {
        const sy = -Math.tanh(i * 0.25) * 9;
        if (i === -10) ctx.moveTo(gx + i, gy + sy);
        else ctx.lineTo(gx + i, gy + sy);
      }
    }
    ctx.stroke();
    ctx.restore();

    // Nodo de Salida y
    ctx.beginPath();
    ctx.arc(outX, outY, 28, 0, Math.PI * 2);
    ctx.fillStyle =
      out >= 0.5 ? "rgba(37, 99, 235, 0.25)" : "rgba(239, 68, 68, 0.2)";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = out >= 0.5 ? "#60a5fa" : "#f87171";
    ctx.stroke();
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 12px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.fillText(`y=${out.toFixed(2)}`, outX, outY + 4);

    // Fórmula en vivo en la parte superior izquierda del diagrama
    drawPillLabel(
      `z = (${w1.toFixed(1)}·${x1.toFixed(1)}) + (${w2.toFixed(1)}·${x2.toFixed(1)}) + (${b.toFixed(1)}) = ${z.toFixed(2)}`,
      nx,
      ny - 125,
      "rgba(15, 23, 42, 0.9)",
      "#334155",
      "#93c5fd",
      11,
    );

    // Detección de Hover para el Inspector Didáctico
    if (state.module === 1 && showHelp) {
      const targets = [
        {
          x: in1X,
          y: in1Y,
          r: 28,
          title: "Entrada x₁",
          desc: "Coordenada horizontal del punto en el plano cartesiano.",
        },
        {
          x: in2X,
          y: in2Y,
          r: 28,
          title: "Entrada x₂",
          desc: "Coordenada vertical del punto en el plano cartesiano.",
        },
        {
          x: nx - 82,
          y: ny - 45,
          r: 26,
          title: "Peso Sináptico w₁",
          desc: "Determina la pendiente horizontal e importancia de x₁.",
        },
        {
          x: nx - 82,
          y: ny + 45,
          r: 26,
          title: "Peso Sináptico w₂",
          desc: "Determina la pendiente vertical e importancia de x₂.",
        },
        {
          x: biasX,
          y: biasY,
          r: 26,
          title: "Sesgo (Bias b)",
          desc: "Umbral que desplaza la línea de decisión sin rotarla.",
        },
        {
          x: nx - 20,
          y: ny,
          r: 30,
          title: "Suma Ponderada (Σ)",
          desc: "Producto escalar z = w₁x₁ + w₂x₂ + b.",
        },
        {
          x: nx + 20,
          y: ny,
          r: 30,
          title: `Activación (${actFn.toUpperCase()})`,
          desc: "Transforma el valor lineal z en la decisión de salida y.",
        },
        {
          x: outX,
          y: outY,
          r: 30,
          title: "Predicción de Salida (y)",
          desc: "Clasificación final: Azul (≥ 0.5) o Rojo (< 0.5).",
        },
      ];
      for (let t of targets) {
        if (Math.hypot(mouseWorldX - t.x, mouseWorldY - t.y) < t.r) {
          activeHoverTarget = t;
          ctx.beginPath();
          ctx.arc(t.x, t.y, t.r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = "#fbbf24";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 4]);
          ctx.stroke();
          ctx.setLineDash([]);
          break;
        }
      }
    }
  }

  // --- PANEL DERECHO: PLANO CARTESIANO 2D INTERACTIVO ---
  const plotRadius = scale * 2.2;
  ctx.save();
  // Marco del gráfico 2D
  ctx.beginPath();
  ctx.roundRect(
    cx - plotRadius,
    cy - plotRadius,
    plotRadius * 2,
    plotRadius * 2,
    12,
  );
  ctx.fillStyle = "#090d16";
  ctx.fill();
  ctx.clip();

  // Cuadrícula interna del plano
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(51, 65, 85, 0.35)";
  for (let g = -2; g <= 2; g += 0.5) {
    ctx.beginPath();
    ctx.moveTo(cx + g * scale, cy - plotRadius);
    ctx.lineTo(cx + g * scale, cy + plotRadius);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - plotRadius, cy - g * scale);
    ctx.lineTo(cx + plotRadius, cy - g * scale);
    ctx.stroke();
  }

  if (state.module === 2 && solvedXor) {
    // Mapa de calor real para la solución XOR con 2 neuronas ocultas
    for (let py = 0; py < heatRes; py++) {
      for (let px = 0; px < heatRes; px++) {
        const wx = (px / (heatRes - 1)) * 4.4 - 2.2;
        const wy = 2.2 - (py / (heatRes - 1)) * 4.4;
        // Combinación de 2 hiperplanos paralelos que aíslan los puntos (0,1) y (1,0)
        const h1 = sigmoid(4.5 * (wx + wy - 0.45));
        const h2 = sigmoid(4.5 * (wx + wy - 1.55));
        const yXor = Math.max(0, Math.min(1, (h1 - h2) * 1.35));

        const idx = (py * heatRes + px) * 4;
        heatImgData.data[idx] = Math.round(239 * (1 - yXor) + 59 * yXor);
        heatImgData.data[idx + 1] = Math.round(68 * (1 - yXor) + 130 * yXor);
        heatImgData.data[idx + 2] = Math.round(68 * (1 - yXor) + 246 * yXor);
        heatImgData.data[idx + 3] = 115;
      }
    }
    heatCtx.putImageData(heatImgData, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(
      heatmapCanvas,
      cx - plotRadius,
      cy - plotRadius,
      plotRadius * 2,
      plotRadius * 2,
    );

    // Dibujar las 2 rectas de la capa oculta h1 y h2
    ctx.strokeStyle = "#c084fc";
    ctx.lineWidth = 2.2;
    ctx.setLineDash([6, 4]);
    // Recta 1: x + y = 0.45
    ctx.beginPath();
    ctx.moveTo(cx + -2 * scale, cy - (0.45 - -2) * scale);
    ctx.lineTo(cx + 2 * scale, cy - (0.45 - 2) * scale);
    ctx.stroke();
    // Recta 2: x + y = 1.55
    ctx.beginPath();
    ctx.moveTo(cx + -2 * scale, cy - (1.55 - -2) * scale);
    ctx.lineTo(cx + 2 * scale, cy - (1.55 - 2) * scale);
    ctx.stroke();
    ctx.setLineDash([]);
  } else {
    // Sombreado de regiones para una sola neurona
    for (let py = 0; py < heatRes; py++) {
      for (let px = 0; px < heatRes; px++) {
        const wx = (px / (heatRes - 1)) * 4.4 - 2.2;
        const wy = 2.2 - (py / (heatRes - 1)) * 4.4;
        const zVal = w1 * wx + w2 * wy + b;
        const prob =
          actFn === "step" ? (zVal >= 0 ? 0.85 : 0.15) : sigmoid(zVal * 1.5);

        const idx = (py * heatRes + px) * 4;
        heatImgData.data[idx] = Math.round(239 * (1 - prob) + 59 * prob);
        heatImgData.data[idx + 1] = Math.round(68 * (1 - prob) + 130 * prob);
        heatImgData.data[idx + 2] = Math.round(68 * (1 - prob) + 246 * prob);
        heatImgData.data[idx + 3] = 75;
      }
    }
    heatCtx.putImageData(heatImgData, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(
      heatmapCanvas,
      cx - plotRadius,
      cy - plotRadius,
      plotRadius * 2,
      plotRadius * 2,
    );

    // Dibujar Línea de Frontera de Decisión: w1*x + w2*y + b = 0
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#f8fafc";
    ctx.beginPath();
    if (Math.abs(w2) > 0.001) {
      const xA = -2.5,
        yA = (-w1 * xA - b) / w2;
      const xB = 2.5,
        yB = (-w1 * xB - b) / w2;
      ctx.moveTo(cx + xA * scale, cy - yA * scale);
      ctx.lineTo(cx + xB * scale, cy - yB * scale);
    } else if (Math.abs(w1) > 0.001) {
      const xVert = -b / w1;
      ctx.moveTo(cx + xVert * scale, cy - plotRadius);
      ctx.lineTo(cx + xVert * scale, cy + plotRadius);
    }
    ctx.stroke();

    // Dibujar Vector Normal de Pesos w = (w1, w2) desde el origen
    if (Math.hypot(w1, w2) > 0.1) {
      const normLen = Math.min(1.4, Math.hypot(w1, w2) * 0.35);
      const angle = Math.atan2(w2, w1);
      const vx = cx + Math.cos(angle) * normLen * scale;
      const vy = cy - Math.sin(angle) * normLen * scale;

      ctx.strokeStyle = "#34d399";
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(vx, vy);
      ctx.stroke();

      // Punta de flecha
      ctx.fillStyle = "#34d399";
      ctx.beginPath();
      ctx.moveTo(vx, vy);
      ctx.lineTo(
        vx - 8 * Math.cos(angle - 0.4),
        vy + 8 * Math.sin(angle - 0.4),
      );
      ctx.lineTo(
        vx - 8 * Math.cos(angle + 0.4),
        vy + 8 * Math.sin(angle + 0.4),
      );
      ctx.closePath();
      ctx.fill();
      ctx.font = "bold 10px JetBrains Mono";
      ctx.fillText("w⃗", vx + 10 * Math.cos(angle), vy - 10 * Math.sin(angle));
    }
  }

  // Ejes principales X1 y X2
  ctx.lineWidth = 1.5;
  ctx.strokeStyle = "#64748b";
  ctx.beginPath();
  ctx.moveTo(cx - plotRadius, cy);
  ctx.lineTo(cx + plotRadius, cy);
  ctx.moveTo(cx, cy - plotRadius);
  ctx.lineTo(cx, cy + plotRadius);
  ctx.stroke();

  // Numeración de los ejes
  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px JetBrains Mono";
  ctx.textAlign = "center";
  for (let t = -2; t <= 2; t++) {
    if (t === 0) continue;
    ctx.fillText(t.toString(), cx + t * scale, cy + 14);
    ctx.fillText(t.toString(), cx - 12, cy - t * scale + 3);
  }
  ctx.fillStyle = "#cbd5e1";
  ctx.font = "bold 11px Inter";
  ctx.fillText("x₁", cx + plotRadius - 14, cy - 8);
  ctx.fillText("x₂", cx + 14, cy - plotRadius + 16);

  // Dibujar Puntos en el Plano Cartesiano
  if (state.module === 2) {
    const xorPoints = [
      { x: 0, y: 0, target: 0 },
      { x: 1, y: 1, target: 0 },
      { x: 0, y: 1, target: 1 },
      { x: 1, y: 0, target: 1 },
    ];
    let hits = 0;
    xorPoints.forEach((pt) => {
      const px = cx + pt.x * scale;
      const py = cy - pt.y * scale;
      let pred = 0;
      if (solvedXor) {
        pred = pt.target;
      } else {
        pred = w1 * pt.x + w2 * pt.y + b >= 0 ? 1 : 0;
      }
      const isCorrect = pred === pt.target;
      if (isCorrect) hits++;

      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = pt.target === 1 ? "#3b82f6" : "#ef4444";
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = isCorrect ? "#ffffff" : "#fbbf24";
      ctx.stroke();

      // Indicador ✓ o ✗
      ctx.fillStyle = isCorrect ? "#34d399" : "#fbbf24";
      ctx.font = "bold 11px Inter";
      ctx.fillText(isCorrect ? "✓" : "✗", px, py - 15);
    });

    const badge = document.getElementById("xor-score-badge");
    if (badge) {
      badge.innerText = `${hits}/4 Aciertos (${hits * 25}%)`;
      badge.className =
        hits === 4
          ? "px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[11px] border border-emerald-500/40"
          : "px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 font-mono text-[11px] border border-amber-500/40";
    }
  } else {
    // Punto interactivo arrastrable (x1, x2) en el Módulo 1
    const px = cx + x1 * scale;
    const py = cy - x2 * scale;
    const z = w1 * x1 + w2 * x2 + b;
    const out = evalActivation(z, actFn);

    // Líneas guía punteadas hacia los ejes
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.5)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, cy);
    ctx.moveTo(px, py);
    ctx.lineTo(cx, py);
    ctx.stroke();
    ctx.setLineDash([]);

    // Halo exterior pulsante
    ctx.beginPath();
    ctx.arc(px, py, 15 + Math.sin(state.time * 0.08) * 2, 0, Math.PI * 2);
    ctx.fillStyle =
      out >= 0.5 ? "rgba(59, 130, 246, 0.25)" : "rgba(239, 68, 68, 0.25)";
    ctx.fill();

    // Punto central
    ctx.beginPath();
    ctx.arc(px, py, 9, 0, Math.PI * 2);
    ctx.fillStyle = out >= 0.5 ? "#3b82f6" : "#ef4444";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();

    drawPillLabel(
      `(${x1.toFixed(1)}, ${x2.toFixed(1)})`,
      px,
      py - 22,
      "rgba(15, 23, 42, 0.9)",
      "#475569",
      "#f8fafc",
      10,
    );
  }

  ctx.restore();

  // Borde exterior del plano 2D
  ctx.beginPath();
  ctx.roundRect(
    cx - plotRadius,
    cy - plotRadius,
    plotRadius * 2,
    plotRadius * 2,
    12,
  );
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#334155";
  ctx.stroke();

  // Etiqueta de ecuación debajo del plano 2D
  drawPillLabel(
    state.module === 2 && solvedXor
      ? `Solución No Lineal: h₁(x₁+x₂-0.45) - h₂(x₁+x₂-1.55)`
      : `Frontera: ${w1.toFixed(1)}x₁ + ${w2.toFixed(1)}x₂ + (${b.toFixed(1)}) = 0`,
    cx,
    cy + plotRadius + 22,
    "rgba(15, 23, 42, 0.9)",
    "#334155",
    "#cbd5e1",
    11,
  );
}

function drawScreenInspectorTooltip(target) {
  const screenPt = worldToScreen(target.x, target.y);
  const boxW = 250;
  const boxH = 58;
  let bx = screenPt.x - boxW / 2;
  let by = screenPt.y - target.r * camera.zoom - boxH - 14;

  if (by < 12) by = screenPt.y + target.r * camera.zoom + 14;
  if (bx < 12) bx = 12;
  if (bx + boxW > width - 12) bx = width - boxW - 12;

  ctx.save();
  ctx.fillStyle = "rgba(15, 23, 42, 0.96)";
  ctx.strokeStyle = "#fbbf24";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(bx, by, boxW, boxH, 8);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#fbbf24";
  ctx.font = "bold 12px Inter";
  ctx.textAlign = "center";
  ctx.fillText(target.title, bx + boxW / 2, by + 20);

  ctx.fillStyle = "#e2e8f0";
  ctx.font = "11px Inter";
  ctx.fillText(target.desc, bx + boxW / 2, by + 40);
  ctx.restore();
}

// ==========================================
// MÓDULO 3: RETROPROPAGACIÓN PASO A PASO
// ==========================================
function drawModule3() {
  const m = state.m3;
  if (!m.fwd) return;

  const cx = width * 0.46;
  const cy = height * 0.46;
  const p = m.progress;

  const layI = cx - 220;
  const layH = cx;
  const layO = cx + 220;
  const dy = 88;

  const nodes = [
    { id: "x₁", x: layI, y: cy - dy, val: m.x[0], type: "in" },
    { id: "x₂", x: layI, y: cy + dy, val: m.x[1], type: "in" },
    {
      id: "h₁",
      x: layH,
      y: cy - dy,
      val: m.fwd.a_h[0],
      z: m.fwd.z_h[0],
      delta: m.bwd.delta_h[0],
      type: "hid",
    },
    {
      id: "h₂",
      x: layH,
      y: cy + dy,
      val: m.fwd.a_h[1],
      z: m.fwd.z_h[1],
      delta: m.bwd.delta_h[1],
      type: "hid",
    },
    {
      id: "ŷ",
      x: layO,
      y: cy,
      val: m.fwd.a_o,
      z: m.fwd.z_o,
      delta: m.bwd.delta_o,
      type: "out",
    },
  ];

  const edges = [
    {
      f: nodes[0],
      t: nodes[2],
      w: m.w_i_h[0][0],
      dw: m.bwd.dw_i_h[0][0],
      lbl: "w₁₁",
    },
    {
      f: nodes[0],
      t: nodes[3],
      w: m.w_i_h[0][1],
      dw: m.bwd.dw_i_h[0][1],
      lbl: "w₁₂",
    },
    {
      f: nodes[1],
      t: nodes[2],
      w: m.w_i_h[1][0],
      dw: m.bwd.dw_i_h[1][0],
      lbl: "w₂₁",
    },
    {
      f: nodes[1],
      t: nodes[3],
      w: m.w_i_h[1][1],
      dw: m.bwd.dw_i_h[1][1],
      lbl: "w₂₂",
    },
    { f: nodes[2], t: nodes[4], w: m.w_h_o[0], dw: m.bwd.dw_h_o[0], lbl: "v₁" },
    { f: nodes[3], t: nodes[4], w: m.w_h_o[1], dw: m.bwd.dw_h_o[1], lbl: "v₂" },
  ];

  // Dibujar conexiones sinápticas y partículas de fase
  edges.forEach((e, idx) => {
    const isUpdate = m.phase === 4;
    const col = isUpdate ? "#10b981" : e.w >= 0 ? "#3b82f6" : "#f43f5e";
    drawBezierSynapse(e.f.x + 28, e.f.y, e.t.x - 28, e.t.y, e.w, 0, col);

    const mx = (e.f.x + e.t.x) / 2;
    const my = (e.f.y + e.t.y) / 2 + (idx % 2 === 0 ? -14 : 14);

    drawPillLabel(
      `${e.lbl}:${e.w.toFixed(2)}`,
      mx,
      my,
      "rgba(15,23,42,0.92)",
      "#334155",
      "#cbd5e1",
      10,
    );

    if (m.phase === 1 && p > 0) {
      const px = e.f.x + (e.t.x - e.f.x) * p;
      const py = e.f.y + (e.t.y - e.f.y) * p;
      ctx.beginPath();
      ctx.arc(px, py, 5.5, 0, Math.PI * 2);
      ctx.fillStyle = "#60a5fa";
      ctx.fill();
    } else if (m.phase === 3 && p > 0) {
      const tBack = 1 - p;
      const px = e.f.x + (e.t.x - e.f.x) * tBack;
      const py = e.f.y + (e.t.y - e.f.y) * tBack;
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = "#fb7185";
      ctx.fill();
      drawPillLabel(
        `∂L/∂w=${e.dw.toFixed(3)}`,
        mx,
        my + 18,
        "rgba(69,10,10,0.9)",
        "#f43f5e",
        "#fda4af",
        9,
      );
    } else if (m.phase === 4) {
      drawPillLabel(
        `Δw=${(-m.lr * e.dw).toFixed(3)}`,
        mx,
        my + 18,
        "rgba(6,78,59,0.9)",
        "#10b981",
        "#6ee7b7",
        9,
      );
    }
  });

  // Dibujar Nodos con barra de activación interna
  nodes.forEach((n) => {
    ctx.save();
    ctx.beginPath();
    ctx.arc(n.x, n.y, 30, 0, Math.PI * 2);
    ctx.fillStyle = "#0f172a";
    ctx.fill();
    ctx.lineWidth = 2.5;
    ctx.strokeStyle =
      n.type === "in" ? "#3b82f6" : n.type === "hid" ? "#a855f7" : "#10b981";
    ctx.stroke();

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 12px Inter";
    ctx.textAlign = "center";
    ctx.fillText(`${n.id} = ${n.val.toFixed(2)}`, n.x, n.y - 3);

    if (n.type !== "in") {
      ctx.fillStyle = "#94a3b8";
      ctx.font = "10px JetBrains Mono";
      ctx.fillText(`z=${n.z.toFixed(2)}`, n.x, n.y + 13);
    }
    ctx.restore();
  });

  // Panel derecho de Error (Loss) e Historial de Convergencia
  const cardX = layO + 65;
  const cardY = cy - 75;
  ctx.fillStyle = "rgba(15, 23, 42, 0.92)";
  ctx.strokeStyle = "#334155";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(cardX, cardY, 175, 150, 10);
  ctx.fill();
  ctx.stroke();

  ctx.fillStyle = "#e2e8f0";
  ctx.font = "bold 11px Inter";
  ctx.textAlign = "left";
  ctx.fillText(`Época #${m.epoch}`, cardX + 12, cardY + 20);
  ctx.fillStyle = "#34d399";
  ctx.font = "11px JetBrains Mono";
  ctx.fillText(`Target y = ${m.target.toFixed(2)}`, cardX + 12, cardY + 38);
  ctx.fillStyle = "#fb7185";
  ctx.fillText(`Loss L   = ${m.fwd.loss.toFixed(4)}`, cardX + 12, cardY + 56);

  // Mini gráfico de evolución del Loss
  const gx = cardX + 12,
    gy = cardY + 70,
    gw = 150,
    gh = 65;
  ctx.fillStyle = "#090d16";
  ctx.fillRect(gx, gy, gw, gh);
  ctx.strokeStyle = "#1e293b";
  ctx.strokeRect(gx, gy, gw, gh);

  if (m.lossHistory.length > 1) {
    const maxL = Math.max(0.25, ...m.lossHistory);
    ctx.beginPath();
    m.lossHistory.forEach((lv, i) => {
      const lx = gx + (i / (m.lossHistory.length - 1)) * gw;
      const ly = gy + gh - (lv / maxL) * (gh - 10) - 5;
      if (i === 0) ctx.moveTo(lx, ly);
      else ctx.lineTo(lx, ly);
    });
    ctx.strokeStyle = "#fb7185";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  // Etiqueta de estado inferior
  const phaseNames = [
    "Listo para iniciar",
    "Fase 1: Propagación Hacia Adelante (Forward Pass)",
    "Fase 2: Evaluación de la Función de Coste L = ½(y - ŷ)²",
    "Fase 3: Retropropagación del Error (Regla de la Cadena ∂L/∂w)",
    "Fase 4: Actualización de Pesos Sinápticos w ← w - η·∂L/∂w",
  ];
  drawPillLabel(
    phaseNames[m.phase] || phaseNames[1],
    width / 2,
    height - 32,
    "rgba(15, 23, 42, 0.95)",
    "#3b82f6",
    "#93c5fd",
    12,
  );
}

// ==========================================
// MÓDULO 4: SANDBOX MULTICAPA (VISTA DUAL)
// ==========================================
function drawModule4() {
  const { mapX, mapY, mapSize } = getSandboxHeatmapRect();

  // --- 1. GRAFO DE ARQUITECTURA EN VIVO (IZQUIERDA) ---
  const netAreaW = mapX - 40;
  const layers = sandboxNet.layers;
  const layerSpacing = Math.min(
    115,
    (netAreaW - 60) / Math.max(1, layers.length - 1),
  );
  const startX = 45;
  const nodeCoords = [];

  for (let l = 0; l < layers.length; l++) {
    const count = layers[l];
    const col = [];
    const totalH = Math.min(height * 0.72, count * 52);
    const stepY = count > 1 ? totalH / (count - 1) : 0;
    const startY = height / 2 - (count > 1 ? totalH / 2 : 0);
    for (let n = 0; n < count; n++) {
      col.push({ x: startX + l * layerSpacing, y: startY + n * stepY });
    }
    nodeCoords.push(col);
  }

  // Conexiones sinápticas de la red
  for (let l = 0; l < sandboxNet.weights.length; l++) {
    for (let i = 0; i < layers[l]; i++) {
      for (let j = 0; j < layers[l + 1]; j++) {
        const w = sandboxNet.weights[l][i][j];
        const p1 = nodeCoords[l][i];
        const p2 = nodeCoords[l + 1][j];
        drawBezierSynapse(
          p1.x,
          p1.y,
          p2.x,
          p2.y,
          w,
          state.m4.isTraining && state.m4.showWeights ? 1.5 : 0,
        );
      }
    }
  }

  // Nodos de cada capa
  for (let l = 0; l < nodeCoords.length; l++) {
    for (let n = 0; n < nodeCoords[l].length; n++) {
      const pt = nodeCoords[l][n];
      const actVal =
        sandboxNet.activations[l] && sandboxNet.activations[l][n] !== undefined
          ? sandboxNet.activations[l][n]
          : 0;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, 13, 0, Math.PI * 2);
      ctx.fillStyle = "#0f172a";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle =
        l === 0
          ? "#3b82f6"
          : l === nodeCoords.length - 1
            ? "#10b981"
            : "#a855f7";
      ctx.stroke();

      ctx.fillStyle = "#e2e8f0";
      ctx.font = "bold 9px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(actVal.toFixed(1), pt.x, pt.y);
    }
  }

  drawPillLabel(
    `Arquitectura [${layers.join(" → ")}] · Activación: ${state.m4.activation.toUpperCase()}`,
    startX + ((layers.length - 1) * layerSpacing) / 2,
    height - 26,
    "rgba(15,23,42,0.9)",
    "#334155",
    "#cbd5e1",
    10,
  );

  // --- 2. MAPA DE CALOR Y FRONTERA DE DECISIÓN (DERECHA) ---
  for (let j = 0; j < heatRes; j++) {
    for (let i = 0; i < heatRes; i++) {
      const x = i / (heatRes - 1);
      const y = j / (heatRes - 1);
      const out = sandboxNet.forward([x, y])[0];

      const isContour = state.m4.showContour && Math.abs(out - 0.5) < 0.032;
      const idx = (j * heatRes + i) * 4;

      if (isContour) {
        heatImgData.data[idx] = 255;
        heatImgData.data[idx + 1] = 255;
        heatImgData.data[idx + 2] = 255;
        heatImgData.data[idx + 3] = 230;
      } else {
        heatImgData.data[idx] = Math.round(239 * (1 - out) + 59 * out);
        heatImgData.data[idx + 1] = Math.round(68 * (1 - out) + 130 * out);
        heatImgData.data[idx + 2] = Math.round(68 * (1 - out) + 246 * out);
        heatImgData.data[idx + 3] = 165;
      }
    }
  }
  heatCtx.putImageData(heatImgData, 0, 0);

  ctx.save();
  ctx.beginPath();
  ctx.roundRect(mapX, mapY, mapSize, mapSize, 12);
  ctx.clip();
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(heatmapCanvas, mapX, mapY, mapSize, mapSize);

  // Puntos del dataset
  for (let pt of currentData) {
    const px = mapX + pt.x[0] * mapSize;
    const py = mapY + pt.x[1] * mapSize;
    ctx.beginPath();
    ctx.arc(px, py, 5.5, 0, Math.PI * 2);
    ctx.fillStyle = pt.y[0] > 0.5 ? "#3b82f6" : "#ef4444";
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();
  }
  ctx.restore();

  ctx.beginPath();
  ctx.roundRect(mapX, mapY, mapSize, mapSize, 12);
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#334155";
  ctx.stroke();

  drawPillLabel(
    "Espacio de Decisión 2D (Clic para insertar datos)",
    mapX + mapSize / 2,
    mapY - 16,
    "rgba(15,23,42,0.9)",
    "#3b82f6",
    "#93c5fd",
    11,
  );
}

// ==========================================
// MÓDULO 5: DESCENSO DEL GRADIENTE EN 3D
// ==========================================
function drawModule5() {
  const gridN = 24;
  const minW = -2.1,
    maxW = 2.1;
  const stepW = (maxW - minW) / gridN;

  // 1. Dibujar malla de la superficie 3D ordenada por profundidad (Painter's Algorithm)
  const quads = [];
  for (let i = 0; i < gridN; i++) {
    for (let j = 0; j < gridN; j++) {
      const w1a = minW + i * stepW;
      const w1b = w1a + stepW;
      const w2a = minW + j * stepW;
      const w2b = w2a + stepW;

      const l00 = evalLossSurface3D(w1a, w2a);
      const l10 = evalLossSurface3D(w1b, w2a);
      const l11 = evalLossSurface3D(w1b, w2b);
      const l01 = evalLossSurface3D(w1a, w2b);

      const p00 = project3DPoint(w1a, w2a, l00);
      const p10 = project3DPoint(w1b, w2a, l10);
      const p11 = project3DPoint(w1b, w2b, l11);
      const p01 = project3DPoint(w1a, w2b, l01);

      const avgDepth = (p00.depth + p10.depth + p11.depth + p01.depth) / 4;
      const avgLoss = (l00 + l10 + l11 + l01) / 4;
      quads.push({ p00, p10, p11, p01, avgDepth, avgLoss });
    }
  }

  quads.sort((a, b) => a.avgDepth - b.avgDepth);

  quads.forEach((q) => {
    const normL = Math.max(0, Math.min(1, q.avgLoss / 2.2));
    const r = Math.round(30 + normL * 190);
    const g = Math.round(90 + (1 - Math.abs(normL - 0.5) * 2) * 95);
    const b = Math.round(235 - normL * 165);

    ctx.beginPath();
    ctx.moveTo(q.p00.x, q.p00.y);
    ctx.lineTo(q.p10.x, q.p10.y);
    ctx.lineTo(q.p11.x, q.p11.y);
    ctx.lineTo(q.p01.x, q.p01.y);
    ctx.closePath();
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.62)`;
    ctx.fill();
    ctx.strokeStyle = "rgba(148, 163, 184, 0.22)";
    ctx.lineWidth = 0.8;
    ctx.stroke();
  });

  // 2. Dibujar trayectorias y esferas de cada optimizador
  state.m5.balls.forEach((ball) => {
    if (ball.trail.length > 1) {
      ctx.beginPath();
      ball.trail.forEach((pt, idx) => {
        const proj = project3DPoint(pt.w1, pt.w2, pt.loss);
        if (idx === 0) ctx.moveTo(proj.x, proj.y);
        else ctx.lineTo(proj.x, proj.y);
      });
      ctx.strokeStyle = ball.color;
      ctx.lineWidth = 2.8;
      ctx.stroke();
    }

    const curLoss = evalLossSurface3D(ball.w1, ball.w2);
    const curProj = project3DPoint(ball.w1, ball.w2, curLoss);

    // Vector de máximo descenso (-∇L)
    if (state.m5.showGradientVec) {
      const { dw1, dw2 } = evalSurfaceGradient3D(ball.w1, ball.w2);
      const nextW1 = ball.w1 - dw1 * 0.28;
      const nextW2 = ball.w2 - dw2 * 0.28;
      const nextLoss = evalLossSurface3D(nextW1, nextW2);
      const vecProj = project3DPoint(nextW1, nextW2, nextLoss);

      ctx.beginPath();
      ctx.moveTo(curProj.x, curProj.y);
      ctx.lineTo(vecProj.x, vecProj.y);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    ctx.beginPath();
    ctx.arc(curProj.x, curProj.y, 7.5, 0, Math.PI * 2);
    ctx.fillStyle = ball.color;
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#ffffff";
    ctx.stroke();

    drawPillLabel(
      `${ball.label}: L=${curLoss.toFixed(2)}`,
      curProj.x,
      curProj.y - 18,
      "rgba(15,23,42,0.9)",
      ball.color,
      "#ffffff",
      10,
    );
  });
}

// ==========================================
// MÓDULO 6: VISIÓN ARTIFICIAL Y REDES CONVOLUCIONALES (CNN)
// ==========================================
function drawModule6() {
  const m = state.m6;
  if (!m.inputGrid.length) return;

  const { cellIn, cellConv, cellPool, inX, inY, convX, convY, poolX, poolY } =
    getCNNLayoutGeometry();
  const sx = m.scanX;
  const sy = m.scanY;

  // --- 1. DIBUJAR IMAGEN DE ENTRADA (10x10) ---
  drawPillLabel(
    "1. Imagen de Entrada (10×10) · [Dibuja aquí]",
    inX + (10 * cellIn) / 2,
    inY - 22,
    "rgba(15, 23, 42, 0.92)",
    "#3b82f6",
    "#93c5fd",
    11,
  );

  for (let r = 0; r < 10; r++) {
    for (let c = 0; c < 10; c++) {
      const val = m.inputGrid[r][c];
      const x = inX + c * cellIn;
      const y = inY + r * cellIn;

      const lum = Math.round(15 + val * 225);
      ctx.fillStyle = `rgb(${lum}, ${lum}, ${Math.min(255, lum + 15)})`;
      ctx.fillRect(x, y, cellIn, cellIn);

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellIn, cellIn);
    }
  }

  // Resaltar la ventana 3x3 activa del Kernel sobre la imagen de entrada
  const winX = inX + sx * cellIn;
  const winY = inY + sy * cellIn;
  ctx.save();
  ctx.strokeStyle = "#fbbf24";
  ctx.lineWidth = 2.8;
  ctx.shadowColor = "#fbbf24";
  ctx.shadowBlur = 10;
  ctx.strokeRect(winX, winY, cellIn * 3, cellIn * 3);
  ctx.restore();

  // Mostrar los pesos del Kernel superpuestos en pequeño dentro de la ventana 3x3 activa
  for (let kr = 0; kr < 3; kr++) {
    for (let kc = 0; kc < 3; kc++) {
      const kw = m.kernel[kr][kc];
      const kx = winX + kc * cellIn + cellIn / 2;
      const ky = winY + kr * cellIn + cellIn / 2;
      ctx.fillStyle = kw > 0 ? "#34d399" : kw < 0 ? "#fb7185" : "#94a3b8";
      ctx.font = "bold 9px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(kw.toString(), kx, ky);
    }
  }

  // --- 2. DIBUJAR MAPA DE CARACTERÍSTICAS CONVOLUCIONADO (8x8) ---
  const maxConv = Math.max(1, ...m.convGrid.flat().map(Math.abs));
  drawPillLabel(
    `2. Mapa Convolucional (8×8) ${m.useRelu ? "+ ReLU" : "(Crudo)"}`,
    convX + (8 * cellConv) / 2,
    convY - 22,
    "rgba(15, 23, 42, 0.92)",
    "#a855f7",
    "#e9d5ff",
    11,
  );

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const val = m.convGrid[r][c];
      const x = convX + c * cellConv;
      const y = convY + r * cellConv;

      const norm = Math.min(1, Math.abs(val) / maxConv);
      if (val >= 0) {
        ctx.fillStyle = `rgba(56, 189, 248, ${0.12 + norm * 0.88})`;
      } else {
        ctx.fillStyle = `rgba(244, 63, 94, ${0.12 + norm * 0.88})`;
      }
      ctx.fillRect(x, y, cellConv, cellConv);

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1;
      ctx.strokeRect(x, y, cellConv, cellConv);

      if (cellConv >= 22) {
        ctx.fillStyle = norm > 0.5 ? "#090d16" : "#cbd5e1";
        ctx.font = "bold 9px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(val.toFixed(0), x + cellConv / 2, y + cellConv / 2);
      }
    }
  }

  // Resaltar el bloque 2x2 de Pooling correspondiente y la celda (sx, sy) activa en el Mapa 8x8
  const poolBlockX = Math.floor(sx / 2);
  const poolBlockY = Math.floor(sy / 2);
  ctx.strokeStyle = "rgba(16, 185, 129, 0.75)";
  ctx.lineWidth = 2;
  ctx.setLineDash([3, 3]);
  ctx.strokeRect(
    convX + poolBlockX * 2 * cellConv,
    convY + poolBlockY * 2 * cellConv,
    cellConv * 2,
    cellConv * 2,
  );
  ctx.setLineDash([]);

  const targetConvX = convX + sx * cellConv;
  const targetConvY = convY + sy * cellConv;
  ctx.save();
  ctx.strokeStyle = "#fbbf24";
  ctx.lineWidth = 2.8;
  ctx.strokeRect(targetConvX, targetConvY, cellConv, cellConv);
  ctx.restore();

  // Haz de proyección que une la ventana 3x3 de entrada con la celda 1x1 de salida
  ctx.save();
  ctx.strokeStyle = "rgba(251, 191, 36, 0.45)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(winX + cellIn * 3, winY);
  ctx.lineTo(targetConvX, targetConvY);
  ctx.moveTo(winX + cellIn * 3, winY + cellIn * 3);
  ctx.lineTo(targetConvX, targetConvY + cellConv);
  ctx.stroke();
  ctx.restore();

  // --- 3. DIBUJAR SALIDA DE MAX-POOLING 2x2 (4x4) ---
  drawPillLabel(
    "3. Max-Pooling 2×2 (4×4)",
    poolX + (4 * cellPool) / 2,
    poolY - 22,
    "rgba(15, 23, 42, 0.92)",
    "#10b981",
    "#6ee7b7",
    11,
  );

  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      const val = m.poolGrid[r][c];
      const x = poolX + c * cellPool;
      const y = poolY + r * cellPool;

      const norm = Math.min(1, Math.abs(val) / maxConv);
      ctx.fillStyle = `rgba(16, 185, 129, ${0.15 + norm * 0.85})`;
      ctx.fillRect(x, y, cellPool, cellPool);

      ctx.strokeStyle = "#1e293b";
      ctx.lineWidth = 1.2;
      ctx.strokeRect(x, y, cellPool, cellPool);

      ctx.fillStyle = norm > 0.55 ? "#052e16" : "#ecfdf5";
      ctx.font = "bold 11px JetBrains Mono";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(val.toFixed(0), x + cellPool / 2, y + cellPool / 2);
    }
  }

  // Resaltar celda activa en el Max-Pool 4x4 y conectar el bloque 2x2 con ella
  const targetPoolX = poolX + poolBlockX * cellPool;
  const targetPoolY = poolY + poolBlockY * cellPool;
  ctx.strokeStyle = "#10b981";
  ctx.lineWidth = 2.8;
  ctx.strokeRect(targetPoolX, targetPoolY, cellPool, cellPool);

  ctx.strokeStyle = "rgba(16, 185, 129, 0.45)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(
    convX + (poolBlockX * 2 + 2) * cellConv,
    convY + poolBlockY * 2 * cellConv,
  );
  ctx.lineTo(targetPoolX, targetPoolY);
  ctx.moveTo(
    convX + (poolBlockX * 2 + 2) * cellConv,
    convY + (poolBlockY * 2 + 2) * cellConv,
  );
  ctx.lineTo(targetPoolX, targetPoolY + cellPool);
  ctx.stroke();

  // --- 4. BARRA DE DESGLOSE MATEMÁTICO EN TIEMPO REAL DE LA CELDA ACTIVA ---
  let rawSum = 0;
  const terms = [];
  for (let kr = 0; kr < 3; kr++) {
    for (let kc = 0; kc < 3; kc++) {
      const pxVal = m.inputGrid[sy + kr][sx + kc];
      const kwVal = m.kernel[kr][kc];
      rawSum += pxVal * kwVal;
      if (pxVal !== 0 && kwVal !== 0) {
        terms.push(`(${pxVal}×${kwVal})`);
      }
    }
  }
  const finalVal = m.useRelu ? Math.max(0, rawSum) : rawSum;
  const eqText =
    terms.length > 0
      ? terms.slice(0, 5).join(" + ") + (terms.length > 5 ? " + ..." : "")
      : "0";

  drawPillLabel(
    `Celda (${sx},${sy}): Σ(I·K) = ${eqText} = ${rawSum.toFixed(1)}  ➔  ${m.useRelu ? `ReLU(${rawSum.toFixed(1)}) = ${finalVal.toFixed(1)}` : `Salida = ${finalVal.toFixed(1)}`}`,
    width / 2,
    height - 30,
    "rgba(15, 23, 42, 0.95)",
    "#fbbf24",
    "#fef08a",
    11,
  );
}

// ==========================================
// BUCLE PRINCIPAL DE ANIMACIÓN Y ENTRENAMIENTO
// ==========================================
window.onload = () => {
  applyTheme(localStorage.getItem("neurosim-theme") || "dark", false);
  resize();
  setModule(1);

  function mainLoop() {
    state.time++;

    // Animación del Módulo 3 (Backpropagation)
    if (state.module === 3) {
      const m = state.m3;
      if (m.runMode === "playing" || m.runMode === "step") {
        m.progress += 0.022;
        if (m.progress >= 1) {
          m.progress = 1;
          if (m.runMode === "step") {
            m.runMode = "paused";
            document.getElementById("btn-play").innerHTML =
              '<i class="fas fa-play text-xs"></i>';
          } else if (m.runMode === "playing") {
            m.runMode = "paused";
            m.waitFrames = 45;
          }
        }
      } else if (m.runMode === "paused" && m.waitFrames > 0) {
        m.waitFrames--;
        if (m.waitFrames === 0) {
          m.phase = m.phase === 4 ? 1 : m.phase + 1;
          if (m.phase === 1) m3ApplyWeights();
          m.progress = 0;
          m.runMode = "playing";
        }
      }
    }

    // Entrenamiento en Vivo del Módulo 4 (Sandbox)
    if (state.module === 4 && state.m4.isTraining && currentData.length > 0) {
      const epochsPerFrame = Math.max(2, Math.round(state.m4.batchSize / 2));
      for (let e = 0; e < epochsPerFrame; e++) {
        currentData.sort(() => Math.random() - 0.5);
        for (let item of currentData) {
          sandboxNet.forward(item.x);
          sandboxNet.backward(item.y);
        }
        state.m4.epoch++;
      }
      evaluateSandboxMetrics();
      state.m4.lossHistory.push(state.m4.loss);
      if (state.m4.lossHistory.length > 60) state.m4.lossHistory.shift();
      updateMetricsUI();
    }

    // Animación en Vivo del Módulo 5 (Descenso del Gradiente 3D)
    if (state.module === 5 && state.m5.isRunning) {
      if (state.time % 2 === 0) {
        step3DOptimizers();
      }
    }

    // Escaneo Automático del Kernel 3x3 en el Módulo 6 (Visión CNN)
    if (state.module === 6 && state.m6.autoScan) {
      if (state.time % 18 === 0) {
        let nextIdx = state.m6.scanY * 8 + state.m6.scanX + 1;
        if (nextIdx >= 64) nextIdx = 0;
        state.m6.scanY = Math.floor(nextIdx / 8);
        state.m6.scanX = nextIdx % 8;
      }
    }

    draw();
    requestAnimationFrame(mainLoop);
  }

  requestAnimationFrame(mainLoop);
};
