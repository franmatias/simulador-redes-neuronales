const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const appSource = fs.readFileSync(
  path.join(__dirname, "..", "js", "app.js"),
  "utf8",
);

function sourceBetween(startMarker, endMarker) {
  const start = appSource.indexOf(startMarker);
  const end = appSource.indexOf(endMarker, start);

  assert.notEqual(start, -1, `No se encontró el inicio: ${startMarker}`);
  assert.notEqual(end, -1, `No se encontró el final: ${endMarker}`);
  return appSource.slice(start, end);
}

function createSimulationContext() {
  const sections = [
    sourceBetween("const state = {", "// SISTEMA DE EXPLICACIONES"),
    sourceBetween(
      "// FUNCIONES MATEMÁTICAS Y RED MULTICAPA",
      "// INTERACCIÓN DE RATÓN Y RUEDA",
    ),
    sourceBetween(
      "// GENERACIÓN DE DATASETS (MÓDULO 4)",
      "// MATEMÁTICA Y OPTIMIZADORES DEL MÓDULO 5 (3D)",
    ),
    sourceBetween(
      "// MATEMÁTICA Y OPTIMIZADORES DEL MÓDULO 5 (3D)",
      "// LÓGICA MATEMÁTICA MÓDULO 3 (BACKPROP REAL)",
    ),
    sourceBetween(
      "// LÓGICA MATEMÁTICA MÓDULO 3 (BACKPROP REAL)",
      "// CLASIFICADOR MULTICLASE (MÓDULO 7)",
    ),
    sourceBetween("function recomputeCNN() {", "function getCNNLayoutGeometry()"),
    sourceBetween(
      "// CLASIFICADOR MULTICLASE (MÓDULO 7)",
      "function getClassifierPlotGeometry()",
    ),
    `
      globalThis.simulations = {
        state,
        evalActivation,
        stepFn,
        SimpleMLP,
        generateData,
        evalLossSurface3D,
        evalSurfaceGradient3D,
        init3DBalls,
        step3DOptimizers,
        m3Compute,
        m3ApplyWeights,
        recomputeCNN,
        classifierForward,
        generateClassifierDataset,
        initializeClassifier,
        classifierLoss,
        trainClassifier,
      };
    `,
  ];

  const context = vm.createContext({
    document: { getElementById: () => null },
  });
  vm.runInContext(sections.join("\n"), context, {
    filename: "js/app.js",
  });
  return context.simulations;
}

function assertFinite(value) {
  assert.equal(Number.isFinite(value), true, `Se esperaba un número finito: ${value}`);
}

test("Módulo 1: activaciones con valores y límites esperados", () => {
  const { evalActivation } = createSimulationContext();

  assert.equal(evalActivation(0, "step"), 1);
  assert.equal(evalActivation(-0.01, "step"), 0);
  assert.equal(evalActivation(0, "sigmoid"), 0.5);
  assert.equal(evalActivation(0, "tanh"), 0);
  assert.equal(evalActivation(-2, "relu"), 0);
  assert.equal(evalActivation(-2, "leaky_relu"), -0.2);
  assert.equal(evalActivation(0, "sin"), 0);
  assert.equal(evalActivation(0, "desconocida"), 1);
  assert.ok(evalActivation(1000, "sigmoid") <= 1);
  assert.ok(evalActivation(-1000, "sigmoid") >= 0);
});

test("Módulo 2: una neurona no puede clasificar perfectamente XOR", () => {
  const { stepFn } = createSimulationContext();
  const samples = [
    { x1: 0, x2: 0, target: 0 },
    { x1: 0, x2: 1, target: 1 },
    { x1: 1, x2: 0, target: 1 },
    { x1: 1, x2: 1, target: 0 },
  ];
  let bestCorrect = 0;

  for (let w1 = -2; w1 <= 2; w1 += 0.5) {
    for (let w2 = -2; w2 <= 2; w2 += 0.5) {
      for (let bias = -2; bias <= 2; bias += 0.5) {
        const correct = samples.filter(
          ({ x1, x2, target }) =>
            stepFn(w1 * x1 + w2 * x2 + bias) === target,
        ).length;
        bestCorrect = Math.max(bestCorrect, correct);
      }
    }
  }

  assert.equal(bestCorrect, 3);
});

test("Módulo 3: backprop calcula gradientes y reduce el error", () => {
  const { state, m3Compute, m3ApplyWeights } = createSimulationContext();
  const model = state.m3;
  m3Compute();

  const epsilon = 1e-5;
  const analyticGradient = model.bwd.dw_h_o[0];
  const originalWeight = model.w_h_o[0];
  model.w_h_o[0] = originalWeight + epsilon;
  m3Compute();
  const lossAbove = model.fwd.loss;
  model.w_h_o[0] = originalWeight - epsilon;
  m3Compute();
  const lossBelow = model.fwd.loss;
  model.w_h_o[0] = originalWeight;

  const numericalGradient = (lossAbove - lossBelow) / (2 * epsilon);
  assert.ok(Math.abs(analyticGradient - numericalGradient) < 1e-6);

  m3Compute();
  const initialLoss = model.fwd.loss;
  for (let epoch = 0; epoch < 15; epoch++) {
    m3Compute();
    m3ApplyWeights();
  }
  assert.ok(model.fwd.loss < initialLoss);
  assert.equal(model.epoch, 15);
  assert.ok(model.lossHistory.every(Number.isFinite));
});

test("Módulo 4: todos los datasets producen muestras binarias válidas", () => {
  const { generateData, state } = createSimulationContext();
  const expectedSizes = {
    xor: 180,
    circles: 200,
    moons: 200,
    spiral: 220,
    blobs: 160,
    checkerboard: 240,
    rings: 240,
    sinwave: 210,
    fourclusters: 200,
  };

  for (const [type, expectedSize] of Object.entries(expectedSizes)) {
    const data = generateData(type);
    assert.equal(data.length, expectedSize, `Tamaño incorrecto para ${type}`);
    assert.ok(
      data.every(
        (sample) =>
          sample.x.length === 2 &&
          sample.y.length === 1 &&
          sample.y[0] >= 0 &&
          sample.y[0] <= 1 &&
          sample.x.every(Number.isFinite),
      ),
      `Muestras inválidas en ${type}`,
    );
  }
  assert.equal(state.m4.noise, 0.08);
});

test("Módulo 4: la retropropagación oculta coincide con el gradiente numérico", () => {
  const { SimpleMLP } = createSimulationContext();
  const network = new SimpleMLP([2, 2, 1], 0.1, "tanh");
  network.weights = [
    [
      [0.2, -0.1],
      [0.4, 0.3],
    ],
    [[0.5], [-0.2]],
  ];
  network.biases = [[0.1, -0.2], [0.05]];

  const input = [0.4, -0.6];
  const target = [0.7];
  const loss = () => 0.5 * (network.forward(input)[0] - target[0]) ** 2;
  const initialLoss = loss();
  const epsilon = 1e-5;
  const originalWeight = network.weights[0][0][0];
  network.weights[0][0][0] = originalWeight + epsilon;
  const lossAbove = loss();
  network.weights[0][0][0] = originalWeight - epsilon;
  const lossBelow = loss();
  network.weights[0][0][0] = originalWeight;

  const numericalGradient = (lossAbove - lossBelow) / (2 * epsilon);
  network.forward(input);
  network.backward(target);
  const analyticGradient =
    (originalWeight - network.weights[0][0][0]) / network.lr;
  assert.ok(Math.abs(analyticGradient - numericalGradient) < 1e-6);

  for (let epoch = 0; epoch < 100; epoch++) {
    network.forward(input);
    network.backward(target);
  }

  assert.ok(loss() < initialLoss);
  assert.ok(network.weights.flat(2).every(Number.isFinite));
});

test("Módulo 5: superficies, gradientes y optimizadores son finitos", () => {
  const {
    state,
    evalLossSurface3D,
    evalSurfaceGradient3D,
    init3DBalls,
    step3DOptimizers,
  } = createSimulationContext();
  const surfaces = ["bowl", "saddle", "local_minima", "ravine", "rosenbrock"];

  for (const surface of surfaces) {
    const loss = evalLossSurface3D(0.7, -0.4, surface);
    const gradient = evalSurfaceGradient3D(0.7, -0.4, surface);
    assertFinite(loss);
    assertFinite(gradient.dw1);
    assertFinite(gradient.dw2);
  }

  const bowlGradient = evalSurfaceGradient3D(0.7, -0.4, "bowl");
  assert.ok(Math.abs(bowlGradient.dw1 - 0.532) < 1e-6);
  assert.ok(Math.abs(bowlGradient.dw2 + 0.304) < 1e-6);

  for (const optimizer of ["sgd", "momentum", "adam", "compare_all"]) {
    state.m5.surface = "bowl";
    state.m5.optimizer = optimizer;
    state.m5.lr = 0.04;
    init3DBalls(-1.5, 1.25);
    const initialLoss = evalLossSurface3D(-1.5, 1.25);

    for (let step = 0; step < 20; step++) step3DOptimizers();

    assert.equal(state.m5.stepCount, 20);
    assert.ok(
      state.m5.balls.every(
        (ball) =>
          Number.isFinite(ball.w1) &&
          Number.isFinite(ball.w2) &&
          Number.isFinite(ball.trail.at(-1).loss) &&
          ball.trail.at(-1).loss < initialLoss,
      ),
      `El optimizador ${optimizer} no descendió`,
    );
  }
});

test("Módulo 6: convolución y max-pooling mantienen dimensiones y valores", () => {
  const { state, recomputeCNN } = createSimulationContext();
  const model = state.m6;
  model.inputGrid = Array.from({ length: 10 }, () => Array(10).fill(1));
  model.kernel = Array.from({ length: 3 }, () => Array(3).fill(1));
  model.useRelu = true;

  recomputeCNN();
  assert.equal(model.convGrid.length, 8);
  assert.ok(model.convGrid.every((row) => row.length === 8));
  assert.equal(model.poolGrid.length, 4);
  assert.ok(model.poolGrid.every((row) => row.length === 4));
  assert.ok(model.convGrid.flat().every((value) => value === 9));
  assert.ok(model.poolGrid.flat().every((value) => value === 9));

  model.kernel = Array.from({ length: 3 }, () => Array(3).fill(-1));
  recomputeCNN();
  assert.ok(model.convGrid.flat().every((value) => value === 0));

  model.useRelu = false;
  recomputeCNN();
  assert.ok(model.convGrid.flat().every((value) => value === -9));
  assert.ok(model.poolGrid.flat().every((value) => value === -9));
});

test("Módulo 7: Softmax suma uno incluso con puntuaciones extremas", () => {
  const { state, classifierForward } = createSimulationContext();
  state.m7.weights = [
    [
      [10000, 0],
      [0, 10000],
      [-10000, -10000],
    ],
  ];
  state.m7.networkBiases = [[0, 0, 0]];

  const probabilities = classifierForward([1, 1]).probabilities;
  assert.equal(probabilities.length, 3);
  assert.ok(probabilities.every((value) => value >= 0 && value <= 1));
  assert.ok(Math.abs(probabilities.reduce((sum, value) => sum + value, 0) - 1) < 1e-12);
});

test("Módulo 7: datasets multiclase respetan etiquetas y límites", () => {
  const { state, generateClassifierDataset } = createSimulationContext();

  for (const datasetType of ["blobs", "circles", "spirals"]) {
    for (const classCount of [2, 5]) {
      state.m7.datasetType = datasetType;
      state.m7.classCount = classCount;
      generateClassifierDataset();

      assert.equal(state.m7.dataset.length, classCount * 36);
      assert.ok(
        state.m7.dataset.every(
          ({ x, label }) =>
            x.length === 2 &&
            x.every((value) => Number.isFinite(value) && value >= -1 && value <= 1) &&
            Number.isInteger(label) &&
            label >= 0 &&
            label < classCount,
        ),
      );
    }
  }
});

test("Módulo 7: cada optimizador de entrenamiento reduce la entropía cruzada", () => {
  const { state, classifierLoss, trainClassifier } = createSimulationContext();

  for (const optimizer of ["batch", "sgd", "minibatch"]) {
    state.m7.classCount = 2;
    state.m7.hiddenLayers = 0;
    state.m7.datasetType = "manual";
    state.m7.optimizer = optimizer;
    state.m7.dataset = Array.from({ length: 8 }, (_, index) => {
      const positive = index % 2 === 0;
      return { x: [positive ? 0.8 : -0.8, 0], label: positive ? 0 : 1 };
    });
    state.m7.weights = [[[1, 0], [-1, 0]]];
    state.m7.networkBiases = [[0, 0]];

    const initialLoss = classifierLoss();
    trainClassifier();

    assert.ok(
      Number.isFinite(state.m7.loss) && state.m7.loss < initialLoss,
      `${optimizer} no redujo la pérdida`,
    );
    assert.equal(state.m7.trained, true);
  }
});

test("La interfaz declara los siete módulos y carga la aplicación", () => {
  const html = fs.readFileSync(path.join(__dirname, "..", "index.html"), "utf8");

  for (let module = 1; module <= 7; module++) {
    assert.match(html, new RegExp(`setModule\\(${module}\\)`));
  }
  assert.match(html, /src="js\/app\.js"/);
});
