import {
  Activity,
  BarChart3,
  Box,
  BrainCircuit,
  GitBranch,
  LineChart,
  Network,
  Sigma,
} from "lucide-react";

import denseBackwardTrace from "./traces/dense_backward.json";
import optimizerPathsTrace from "./traces/optimizer_paths.json";
import perceptronTrace from "./traces/perceptron_or.json";

export type ModuleKey =
  | "perceptron"
  | "neuron"
  | "dense"
  | "backprop"
  | "optimizer"
  | "training";

export type ModuleSpec = {
  key: ModuleKey;
  title: string;
  file: string;
  metric: string;
  accent: string;
  icon: typeof BrainCircuit;
};

export const modules: ModuleSpec[] = [
  {
    key: "perceptron",
    title: "Perceptron",
    file: "src/ai_lab/perceptron.py",
    metric: "binary update",
    accent: "text-weight",
    icon: GitBranch,
  },
  {
    key: "neuron",
    title: "Neuron",
    file: "neural_core/neuron.py",
    metric: "differentiable",
    accent: "text-current",
    icon: BrainCircuit,
  },
  {
    key: "dense",
    title: "DenseLayer",
    file: "neural_core/layers.py",
    metric: "matrix core",
    accent: "text-signal",
    icon: Network,
  },
  {
    key: "backprop",
    title: "Backprop",
    file: "neural_core/network.py",
    metric: "gradient flow",
    accent: "text-error",
    icon: Sigma,
  },
  {
    key: "optimizer",
    title: "Optimizers",
    file: "neural_core/optimizers.py",
    metric: "stateful steps",
    accent: "text-amber-500",
    icon: Activity,
  },
  {
    key: "training",
    title: "Training Runs",
    file: "artifacts/plots",
    metric: "verified curves",
    accent: "text-cyan-500",
    icon: BarChart3,
  },
];

export const xorPoints = [
  { x: 0, y: 0, label: 0 },
  { x: 0, y: 1, label: 1 },
  { x: 1, y: 0, label: 1 },
  { x: 1, y: 1, label: 0 },
];

export const perceptronSteps = perceptronTrace.steps;

export const denseTrace = denseBackwardTrace;

export const backpropFlow = [
  { name: "loss", value: 1.0 },
  { name: "dA", value: 0.74 },
  { name: "dZ", value: 0.58 },
  { name: "dW", value: 0.42 },
  { name: "db", value: 0.36 },
  { name: "dX", value: 0.24 },
];

const optimizerPathColors = {
  SGD: "#38bdf8",
  Momentum: "#22c55e",
  Adam: "#f59e0b",
} as const;

export const optimizerPaths = Object.entries(optimizerPathsTrace.paths).map(([name, points]) => ({
  name,
  color: optimizerPathColors[name as keyof typeof optimizerPathColors],
  points,
}));

export const trainingPlots = [
  { name: "XOR", src: "./plots/xor_network_history.png", icon: Box },
  { name: "Optimizers", src: "./plots/optimizer_regression_comparison.png", icon: LineChart },
  { name: "MNIST", src: "./plots/mnist_training_history.png", icon: BarChart3 },
  { name: "Predictions", src: "./plots/mnist_sample_predictions.png", icon: BrainCircuit },
];
