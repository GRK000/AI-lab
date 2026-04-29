import {
  ArrowRight,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Cpu,
  Database,
  ExternalLink,
  Gauge,
  Layers3,
  MousePointer2,
  Pause,
  Play,
  ShieldCheck,
  Sparkles,
} from "lucide-react";
import type { ReactNode } from "react";
import { useMemo, useState } from "react";

import {
  backpropFlow,
  denseTrace,
  ModuleKey,
  modules,
  optimizerPaths,
  perceptronSteps,
  trainingPlots,
  xorPoints,
} from "./data";

type DenseMatrixValues = Pick<typeof denseTrace, "input" | "weights" | "bias" | "output">;

const githubRepo = "https://github.com/";
const denseUnits = 2;

function makeMatrix(rows: number, cols: number, offset = 0) {
  return Array.from({ length: rows }, (_, row) =>
    Array.from({ length: cols }, (_, col) => {
      const value = ((row + 1) * 0.17 - (col + 1) * 0.09 + offset).toFixed(2);
      return Number(value);
    }),
  );
}

function App() {
  const [active, setActive] = useState<ModuleKey>("perceptron");
  const [step, setStep] = useState(0);
  const activeModule = modules.find((item) => item.key === active) ?? modules[0];
  const activeIndex = modules.findIndex((item) => item.key === active);
  const currentStep = perceptronSteps[step];

  const navTo = (offset: number) => {
    const next = (activeIndex + offset + modules.length) % modules.length;
    setActive(modules[next].key);
  };

  return (
    <div className="h-screen overflow-hidden bg-graphite-50 text-graphite-950 transition-colors">
      <header className="sticky top-0 z-20 border-b border-graphite-200/80 bg-graphite-50/90 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl items-center justify-between gap-4 px-4 py-3 sm:px-6 lg:px-8">
          <a href="#core" className="flex items-center gap-3 rounded-md focus:outline-none focus:ring-2 focus:ring-current">
            <span className="grid h-10 w-10 place-items-center rounded-md bg-graphite-950 text-white">
              <Cpu aria-hidden="true" size={21} />
            </span>
            <span>
              <span className="block text-sm font-semibold leading-5">AI Lab Visual Core</span>
              <span className="block text-xs text-graphite-600">NumPy neural network internals</span>
            </span>
          </a>
          <nav className="hidden items-center gap-1 lg:flex" aria-label="Core modules">
            {modules.map((item) => (
              <button
                key={item.key}
                type="button"
                onClick={() => setActive(item.key)}
                className={`rounded-md px-3 py-2 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-current ${
                  active === item.key
                    ? "bg-graphite-950 text-white"
                    : "text-graphite-700 hover:bg-white hover:text-graphite-950"
                }`}
              >
                {item.title}
              </button>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <a
              href={githubRepo}
              aria-label="Open repository"
              className="grid h-10 w-10 place-items-center rounded-md border border-graphite-300 bg-white text-graphite-700 shadow-sm transition hover:border-graphite-500 hover:text-graphite-950 focus:outline-none focus:ring-2 focus:ring-current"
            >
              <ExternalLink aria-hidden="true" size={18} />
            </a>
          </div>
        </div>
      </header>

      <main id="core" className="mx-auto h-[calc(100vh-73px)] max-w-7xl overflow-hidden px-4 py-5 sm:px-6 lg:px-8">
        <section className="grid h-full min-h-0 gap-5 lg:grid-cols-[280px_minmax(0,1fr)]">
          <aside className="order-2 min-h-0 lg:order-1">
            <div className="grid gap-2 lg:sticky lg:top-20">
              {modules.map((item) => {
                const Icon = item.icon;
                const selected = active === item.key;
                return (
                  <button
                    key={item.key}
                    type="button"
                    onClick={() => setActive(item.key)}
                    className={`group grid min-h-16 grid-cols-[40px_1fr_auto] items-center gap-3 rounded-md border px-3 text-left transition focus:outline-none focus:ring-2 focus:ring-current ${
                      selected
                        ? "border-graphite-950 bg-white shadow-panel"
                        : "border-graphite-200 bg-white/65 hover:border-graphite-400 hover:bg-white"
                    }`}
                  >
                    <span
                      className={`grid h-10 w-10 place-items-center rounded-md ${
                        selected
                          ? "bg-graphite-950 text-white"
                          : "bg-graphite-100 text-graphite-700"
                      }`}
                    >
                      <Icon aria-hidden="true" size={19} />
                    </span>
                    <span>
                      <span className="block text-sm font-semibold">{item.title}</span>
                      <span className="block text-xs text-graphite-600">{item.metric}</span>
                    </span>
                    <ArrowRight
                      aria-hidden="true"
                      size={16}
                      className={selected ? "opacity-100" : "opacity-0 group-hover:opacity-70"}
                    />
                  </button>
                );
              })}
            </div>
          </aside>

          <section className="order-1 min-h-0 lg:order-2">
            <div className="grid h-full min-h-0 gap-5 xl:grid-cols-[minmax(0,1.15fr)_360px]">
              <div className="flex min-h-0 min-w-0 flex-col overflow-hidden rounded-md border border-graphite-200 bg-white shadow-panel">
                <div className="flex flex-wrap items-start justify-between gap-4 border-b border-graphite-200 px-5 py-4">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite-500">
                      {activeModule.file}
                    </p>
                    <h1 className="mt-1 text-2xl font-semibold leading-tight text-graphite-950 sm:text-3xl">
                      {activeModule.title}
                    </h1>
                  </div>
                  <div className="flex items-center gap-2">
                    <IconButton label="Previous module" onClick={() => navTo(-1)}>
                      <ChevronLeft aria-hidden="true" size={18} />
                    </IconButton>
                    <IconButton label="Next module" onClick={() => navTo(1)}>
                      <ChevronRight aria-hidden="true" size={18} />
                    </IconButton>
                  </div>
                </div>
                <div className="min-h-0 min-w-0 flex-1 px-4 py-5 sm:px-5">
                  {active === "perceptron" && (
                    <PerceptronPanel step={step} setStep={setStep} currentStep={currentStep} />
                  )}
                  {active === "neuron" && <NeuronPanel step={step} />}
                  {active === "dense" && <DenseLayerPanel />}
                  {active === "backprop" && <BackpropPanel />}
                  {active === "optimizer" && <OptimizerPanel />}
                  {active === "training" && <TrainingPanel />}
                </div>
              </div>

              <Diagnostics active={active} />
            </div>
          </section>
        </section>
      </main>
    </div>
  );
}

function IconButton({
  label,
  onClick,
  children,
}: {
  label: string;
  onClick: () => void;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      aria-label={label}
      title={label}
      onClick={onClick}
      className="grid h-10 w-10 place-items-center rounded-md border border-graphite-300 bg-white text-graphite-700 transition hover:border-graphite-500 hover:text-graphite-950 focus:outline-none focus:ring-2 focus:ring-current"
    >
      {children}
    </button>
  );
}

function SegmentedControl<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (value: T) => void;
}) {
  return (
    <div className="rounded-md border border-graphite-200 bg-white p-1">
      <div className="sr-only">{label}</div>
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}>
        {options.map((option) => (
          <button
            key={option}
            type="button"
            aria-pressed={value === option}
            onClick={() => onChange(option)}
            className={`rounded px-3 py-2 text-center text-sm font-semibold transition focus:outline-none focus:ring-2 focus:ring-current ${
              value === option
                ? "bg-graphite-950 text-white"
                : "text-graphite-600 hover:bg-graphite-100 hover:text-graphite-950"
            }`}
          >
            {option}
          </button>
        ))}
      </div>
    </div>
  );
}

function PerceptronPanel({
  step,
  setStep,
  currentStep,
}: {
  step: number;
  setStep: (step: number) => void;
  currentStep: (typeof perceptronSteps)[number];
}) {
  const wrong = currentStep.target !== currentStep.prediction;
  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
      <div className="rounded-md border border-graphite-200 bg-graphite-50 p-4">
        <DecisionPlane step={step} />
      </div>
      <div className="grid content-start gap-4">
        <Stepper step={step} setStep={setStep} />
        <div className="rounded-md border border-graphite-200 bg-white p-4">
          <p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite-500">
            active sample
          </p>
          <div className="mt-3 grid grid-cols-2 gap-3">
            <Metric label="x" value={`[${currentStep.sample.join(", ")}]`} />
            <Metric label="target" value={String(currentStep.target)} />
            <Metric label="prediction" value={String(currentStep.prediction)} />
            <Metric label="status" value={wrong ? "update" : "stable"} tone={wrong ? "error" : "ok"} />
          </div>
        </div>
        <WeightsVector weights={currentStep.weights} bias={currentStep.bias} />
      </div>
    </div>
  );
}

function Stepper({ step, setStep }: { step: number; setStep: (step: number) => void }) {
  const [playing, setPlaying] = useState(false);
  return (
    <div className="rounded-md border border-graphite-200 bg-white p-4">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-semibold">Epoch sample {step + 1}</p>
        <button
          type="button"
          aria-label={playing ? "Pause sample playback" : "Play sample playback"}
          title={playing ? "Pause" : "Play"}
          onClick={() => {
            setPlaying((value) => !value);
            setStep((step + 1) % perceptronSteps.length);
          }}
          className="grid h-9 w-9 place-items-center rounded-md bg-graphite-950 text-white focus:outline-none focus:ring-2 focus:ring-current"
        >
          {playing ? <Pause aria-hidden="true" size={16} /> : <Play aria-hidden="true" size={16} />}
        </button>
      </div>
      <input
        aria-label="Perceptron sample step"
        className="mt-4 w-full accent-graphite-950"
        type="range"
        min={0}
        max={perceptronSteps.length - 1}
        value={step}
        onChange={(event) => setStep(Number(event.target.value))}
      />
    </div>
  );
}

function DecisionPlane({ step }: { step: number }) {
  const selected = perceptronSteps[step];
  return (
    <svg viewBox="0 0 520 390" role="img" aria-label="Perceptron decision plane" className="h-full min-h-[390px] w-full">
      <rect width="520" height="390" rx="8" className="fill-graphite-50" />
      <g className="stroke-graphite-300" strokeWidth="1">
        {[80, 170, 260, 350, 440].map((x) => (
          <line key={`v-${x}`} x1={x} y1="42" x2={x} y2="330" />
        ))}
        {[64, 128, 192, 256, 320].map((y) => (
          <line key={`h-${y}`} x1="58" y1={y} x2="468" y2={y} />
        ))}
      </g>
      <line x1="62" y1="310" x2="472" y2="86" className="stroke-graphite-900" strokeWidth="3" />
      <line x1="62" y1="280" x2="472" y2="56" stroke="#38bdf8" strokeWidth="2" strokeDasharray="8 9" />
      {xorPoints.map((point) => {
        const cx = 112 + point.x * 300;
        const cy = 278 - point.y * 200;
        const selectedPoint = selected.sample[0] === point.x && selected.sample[1] === point.y;
        return (
          <g key={`${point.x}-${point.y}`}>
            <circle
              cx={cx}
              cy={cy}
              r={selectedPoint ? 18 : 13}
              fill={point.label === 1 ? "#22c55e" : "#f43f5e"}
              stroke={selectedPoint ? "#121715" : "#ffffff"}
              strokeWidth={selectedPoint ? 4 : 3}
            />
            <text x={cx} y={cy + 5} textAnchor="middle" fontSize="12" fontWeight="700" fill="#fff">
              {point.label}
            </text>
          </g>
        );
      })}
      <text x="64" y="360" className="fill-graphite-600" fontSize="13">
        feature 1
      </text>
      <text x="18" y="78" className="fill-graphite-600" fontSize="13" transform="rotate(-90 18 78)">
        feature 2
      </text>
    </svg>
  );
}

function WeightsVector({ weights, bias }: { weights: number[]; bias: number }) {
  return (
    <div className="rounded-md border border-graphite-200 bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite-500">
        parameter vector
      </p>
      <div className="mt-3 grid gap-2">
        {[...weights.map((value, index) => [`w${index + 1}`, value] as const), ["b", bias] as const].map(
          ([label, value]) => (
            <div key={label} className="grid grid-cols-[34px_1fr_54px] items-center gap-3">
              <span className="font-mono text-xs text-graphite-500">{label}</span>
              <span className="h-2 overflow-hidden rounded-full bg-graphite-100">
                <span
                  className="block h-full rounded-full bg-weight"
                  style={{ width: `${Math.max(12, Math.abs(Number(value)) * 180)}%` }}
                />
              </span>
              <span className="text-right font-mono text-xs">{Number(value).toFixed(2)}</span>
            </div>
          ),
        )}
      </div>
    </div>
  );
}

function NeuronPanel({ step }: { step: number }) {
  const [zOverride, setZOverride] = useState(0);
  const [mode, setMode] = useState<"sample" | "manual">("sample");
  const sampleZ = perceptronSteps[step].weights.reduce(
    (total, weight, index) => total + weight * perceptronSteps[step].sample[index],
    perceptronSteps[step].bias,
  );
  const weightedSum = mode === "sample" ? sampleZ : zOverride;
  const sigmoid = 1 / (1 + Math.exp(-weightedSum));
  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_300px]">
      <div className="rounded-md border border-graphite-200 bg-graphite-50 p-4">
        <ActivationCurve value={weightedSum} />
      </div>
      <div className="grid content-start gap-4">
        <SegmentedControl
          label="Neuron input mode"
          value={mode}
          options={["sample", "manual"]}
          onChange={setMode}
        />
        <div className="rounded-md border border-graphite-200 bg-white p-4">
          <div className="flex items-center justify-between gap-3">
            <span className="text-sm font-semibold">z input</span>
            <span className="font-mono text-sm">{weightedSum.toFixed(2)}</span>
          </div>
          <input
            aria-label="Manual neuron z input"
            type="range"
            min="-4"
            max="4"
            step="0.1"
            value={mode === "sample" ? sampleZ : zOverride}
            disabled={mode === "sample"}
            onChange={(event) => setZOverride(Number(event.target.value))}
            className="mt-4 w-full accent-graphite-950 disabled:opacity-45"
          />
        </div>
        <MetricBlock
          items={[
            ["linear z", weightedSum.toFixed(3)],
            ["sigmoid", sigmoid.toFixed(3)],
            ["threshold", sigmoid >= 0.5 ? "positive" : "negative"],
          ]}
        />
        <SignalStack values={[0.2, 0.44, sigmoid, 0.76]} />
      </div>
    </div>
  );
}

function ActivationCurve({ value }: { value: number }) {
  const x = 260 + value * 160;
  const y = 278 - (1 / (1 + Math.exp(-value))) * 220;
  return (
    <svg viewBox="0 0 520 390" role="img" aria-label="Sigmoid activation curve" className="h-full min-h-[390px] w-full">
      <rect width="520" height="390" rx="8" className="fill-graphite-50" />
      <line x1="50" y1="300" x2="470" y2="300" className="stroke-graphite-300" />
      <line x1="260" y1="54" x2="260" y2="326" className="stroke-graphite-300" />
      <path
        d="M 58 292 C 136 286, 184 246, 228 205 S 318 110, 462 78"
        fill="none"
        stroke="#38bdf8"
        strokeWidth="5"
        strokeLinecap="round"
      />
      <circle cx={x} cy={y} r="15" fill="#22c55e" stroke="#121715" strokeWidth="4" />
      <text x="60" y="48" className="fill-graphite-900" fontSize="16" fontWeight="700">
        sigmoid(z)
      </text>
      <text x={Math.min(410, Math.max(80, x - 34))} y={y - 24} className="fill-graphite-900" fontSize="13">
        z={value.toFixed(2)}
      </text>
    </svg>
  );
}

function DenseLayerPanel() {
  const [activeTerm, setActiveTerm] = useState<"X" | "W" | "b" | "Z">("W");
  const [inputRows, setInputRows] = useState(2);
  const [inputCols, setInputCols] = useState(3);
  const denseValues = useMemo(
    () => ({
      input: makeMatrix(inputRows, inputCols, 0.12),
      weights: makeMatrix(inputCols, denseUnits, 0.31),
      bias: [makeMatrix(1, denseUnits, -0.08)[0]],
      output: makeMatrix(inputRows, denseUnits, 0.04),
    }),
    [inputRows, inputCols],
  );
  const descriptions = {
    X: "Batch input cached during forward pass.",
    W: "Trainable weights updated by the optimizer.",
    b: "Bias row broadcast across the batch.",
    Z: "Linear output before the activation derivative.",
  };
  return (
    <div className="grid min-w-0 gap-5">
      <SegmentedControl
        label="Dense layer term"
        value={activeTerm}
        options={["X", "W", "b", "Z"]}
        onChange={setActiveTerm}
      />
      <MatrixEquation
        activeTerm={activeTerm}
        setActiveTerm={setActiveTerm}
        values={denseValues}
      />
      <div className="grid gap-4 md:grid-cols-3">
        <ShapeInput
          rows={inputRows}
          cols={inputCols}
          setRows={setInputRows}
          setCols={setInputCols}
        />
        <Metric label="weights shape" value={`${inputCols} x ${denseUnits}`} />
        <Metric label="output shape" value={`${inputRows} x ${denseUnits}`} tone="ok" />
      </div>
      <div className="rounded-md border border-graphite-200 bg-white p-4">
        <p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite-500">
          selected tensor
        </p>
        <p className="mt-2 text-sm leading-6 text-graphite-700">
          <span className="font-mono font-semibold text-graphite-950">{activeTerm}</span>{" "}
          {descriptions[activeTerm]}
        </p>
      </div>
    </div>
  );
}

function MatrixEquation({
  activeTerm,
  setActiveTerm,
  values,
}: {
  activeTerm: "X" | "W" | "b" | "Z";
  setActiveTerm: (term: "X" | "W" | "b" | "Z") => void;
  values: DenseMatrixValues;
}) {
  return (
    <div className="min-w-0 rounded-md border border-graphite-200 bg-graphite-50 p-3 sm:p-4">
      <div className="grid min-w-0 items-stretch gap-3 lg:grid-cols-[minmax(0,1fr)_20px_minmax(0,1fr)_20px_minmax(0,.75fr)_20px_minmax(0,1fr)]">
      <Matrix title="X" values={values.input} active={activeTerm === "X"} onSelect={() => setActiveTerm("X")} />
      <Operator symbol="x" />
      <Matrix title="W" values={values.weights} tone="weight" active={activeTerm === "W"} onSelect={() => setActiveTerm("W")} />
      <Operator symbol="+" />
      <Matrix title="b" values={values.bias} tone="current" active={activeTerm === "b"} onSelect={() => setActiveTerm("b")} />
      <Operator symbol="=" />
      <Matrix title="Z" values={values.output} tone="signal" active={activeTerm === "Z"} onSelect={() => setActiveTerm("Z")} />
      </div>
    </div>
  );
}

function Matrix({
  title,
  values,
  tone = "default",
  active = false,
  onSelect,
}: {
  title: string;
  values: number[][];
  tone?: string;
  active?: boolean;
  onSelect?: () => void;
}) {
  const toneClass =
    tone === "weight"
      ? "border-weight/40 bg-amber-50"
      : tone === "signal"
        ? "border-signal/40 bg-emerald-50"
        : tone === "current"
          ? "border-current/40 bg-sky-50"
          : "border-graphite-200 bg-white";
  return (
    <button
      type="button"
      onClick={onSelect}
      className={`min-w-0 rounded-md p-1 text-left transition focus:outline-none focus:ring-2 focus:ring-current ${
        active ? "bg-graphite-950/5 ring-2 ring-graphite-950" : ""
      }`}
    >
      <p className="mb-2 text-center font-mono text-sm font-semibold">{title}</p>
      <div className={`grid max-w-full gap-1 rounded-md border p-1.5 ${toneClass}`}>
        {values.map((row, rowIndex) => (
          <div
            key={rowIndex}
            className="grid min-w-0 gap-1"
            style={{ gridTemplateColumns: `repeat(${row.length}, minmax(0, 1fr))` }}
          >
            {row.map((value, colIndex) => (
              <span
                key={`${rowIndex}-${colIndex}`}
                className="min-w-0 rounded bg-white px-1 py-2 text-center font-mono text-[10px] leading-none shadow-sm"
              >
                {value.toFixed(2)}
              </span>
            ))}
          </div>
        ))}
      </div>
    </button>
  );
}

function Operator({ symbol }: { symbol: string }) {
  return <span className="text-center text-xl font-semibold text-graphite-500">{symbol}</span>;
}

function ShapeInput({
  rows,
  cols,
  setRows,
  setCols,
}: {
  rows: number;
  cols: number;
  setRows: (value: number) => void;
  setCols: (value: number) => void;
}) {
  const clamp = (value: number) => Math.min(4, Math.max(1, value));
  return (
    <div className="rounded-md border border-graphite-200 bg-white p-3">
      <p className="text-xs uppercase tracking-[0.12em] text-graphite-500">
        input shape
      </p>
      <div className="mt-2 grid grid-cols-[1fr_auto_1fr] items-center gap-2">
        <label className="sr-only" htmlFor="dense-rows">
          Dense input rows
        </label>
        <input
          id="dense-rows"
          type="number"
          min={1}
          max={4}
          value={rows}
          onChange={(event) => setRows(clamp(Number(event.target.value)))}
          className="h-9 min-w-0 rounded border border-graphite-300 bg-white px-2 text-center font-mono text-sm font-semibold text-graphite-950 focus:outline-none focus:ring-2 focus:ring-current"
        />
        <span className="font-mono text-sm font-semibold text-graphite-500">
          x
        </span>
        <label className="sr-only" htmlFor="dense-cols">
          Dense input columns
        </label>
        <input
          id="dense-cols"
          type="number"
          min={1}
          max={4}
          value={cols}
          onChange={(event) => setCols(clamp(Number(event.target.value)))}
          className="h-9 min-w-0 rounded border border-graphite-300 bg-white px-2 text-center font-mono text-sm font-semibold text-graphite-950 focus:outline-none focus:ring-2 focus:ring-current"
        />
      </div>
    </div>
  );
}

function BackpropPanel() {
  const [selected, setSelected] = useState(0);
  return (
    <div className="grid gap-4">
    <div className="rounded-md border border-graphite-200 bg-graphite-50 p-4">
      <svg viewBox="0 0 760 390" role="img" aria-label="Backpropagation gradient flow" className="h-full min-h-[390px] w-full">
        <rect width="760" height="390" rx="8" className="fill-graphite-50" />
        {backpropFlow.map((item, index) => {
          const x = 80 + index * 118;
          const height = 230 * item.value;
          const active = index === selected;
          return (
            <g key={item.name} opacity={index <= selected ? 1 : 0.36}>
              {index < backpropFlow.length - 1 && (
                <line x1={x + 38} y1="184" x2={x + 108} y2="184" className="stroke-graphite-400" strokeWidth="3" markerEnd="url(#arrow)" />
              )}
              <rect x={x - 28} y={278 - height} width="56" height={height} rx="6" fill={index === 0 ? "#f43f5e" : active ? "#22c55e" : "#38bdf8"} />
              {active && <rect x={x - 36} y={270 - height} width="72" height={height + 16} rx="8" fill="none" className="stroke-graphite-950" strokeWidth="3" />}
              <text x={x} y="316" textAnchor="middle" fontSize="13" fontWeight="700" className="fill-graphite-900">
                {item.name}
              </text>
              <text x={x} y={258 - height} textAnchor="middle" fontSize="12" className="fill-graphite-600">
                {item.value.toFixed(2)}
              </text>
            </g>
          );
        })}
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
            <path d="M 0 0 L 8 4 L 0 8 z" className="fill-graphite-400" />
          </marker>
        </defs>
      </svg>
    </div>
      <div className="rounded-md border border-graphite-200 bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <p className="text-sm font-semibold">Selected gradient: {backpropFlow[selected].name}</p>
          <p className="font-mono text-sm text-graphite-600">
            {backpropFlow[selected].value.toFixed(2)}
          </p>
        </div>
        <input
          aria-label="Backpropagation gradient step"
          type="range"
          min={0}
          max={backpropFlow.length - 1}
          value={selected}
          onChange={(event) => setSelected(Number(event.target.value))}
          className="mt-4 w-full accent-graphite-950"
        />
      </div>
    </div>
  );
}

function OptimizerPanel() {
  const [activeOptimizer, setActiveOptimizer] = useState("Adam");
  return (
    <div className="grid gap-5">
      <div className="rounded-md border border-graphite-200 bg-graphite-50 p-4">
        <svg viewBox="0 0 520 320" role="img" aria-label="Optimizer descent paths" className="h-full min-h-[320px] w-full">
          <rect width="520" height="320" rx="8" className="fill-graphite-50" />
          <ellipse cx="420" cy="104" rx="44" ry="18" fill="none" className="stroke-graphite-300" />
          <ellipse cx="420" cy="104" rx="94" ry="42" fill="none" className="stroke-graphite-200" />
          <ellipse cx="420" cy="104" rx="154" ry="72" fill="none" className="stroke-graphite-100" />
          {optimizerPaths.map((path) => (
            <polyline
              key={path.name}
              points={path.points.map(([x, y]) => `${x + 180},${y + 50}`).join(" ")}
              fill="none"
              stroke={path.color}
              strokeWidth={path.name === activeOptimizer ? 7 : 4}
              opacity={path.name === activeOptimizer ? 1 : 0.24}
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          ))}
          <circle cx="420" cy="104" r="8" className="fill-graphite-950" />
        </svg>
      </div>
      <div className="grid gap-3 md:grid-cols-3">
        {optimizerPaths.map((path) => (
          <button
            type="button"
            onClick={() => setActiveOptimizer(path.name)}
            key={path.name}
            className={`rounded-md border p-4 text-left transition focus:outline-none focus:ring-2 focus:ring-current ${
              activeOptimizer === path.name
                ? "border-graphite-950 bg-white shadow-panel"
                : "border-graphite-200 bg-white hover:border-graphite-400"
            }`}
          >
            <span className="block h-2 w-12 rounded-full" style={{ backgroundColor: path.color }} />
            <p className="mt-3 text-sm font-semibold">{path.name}</p>
            <p className="mt-1 text-xs leading-5 text-graphite-600">state-aware parameter movement</p>
          </button>
        ))}
      </div>
    </div>
  );
}

function TrainingPanel() {
  const [selectedPlot, setSelectedPlot] = useState(trainingPlots[0]);
  const SelectedIcon = selectedPlot.icon;
  return (
    <div className="grid gap-4">
      <figure className="overflow-hidden rounded-md border border-graphite-200 bg-white">
        <div className="flex items-center gap-2 border-b border-graphite-200 px-4 py-3">
          <SelectedIcon aria-hidden="true" size={17} />
          <figcaption className="text-sm font-semibold">{selectedPlot.name}</figcaption>
        </div>
        <img
          src={selectedPlot.src}
          alt={`${selectedPlot.name} training artifact`}
          className="max-h-[420px] w-full object-contain p-3"
        />
      </figure>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
      {trainingPlots.map((plot) => {
        const Icon = plot.icon;
        return (
          <button
            key={plot.name}
            type="button"
            onClick={() => setSelectedPlot(plot)}
            className={`overflow-hidden rounded-md border text-left transition focus:outline-none focus:ring-2 focus:ring-current ${
              selectedPlot.name === plot.name
                ? "border-graphite-950 bg-white shadow-panel"
                : "border-graphite-200 bg-white hover:border-graphite-400"
            }`}
          >
            <div className="flex items-center gap-2 border-b border-graphite-200 px-4 py-3">
              <Icon aria-hidden="true" size={17} />
              <span className="text-sm font-semibold">{plot.name}</span>
            </div>
            <img src={plot.src} alt={`${plot.name} training artifact`} className="aspect-[4/3] w-full object-contain p-3" />
          </button>
        );
      })}
      </div>
    </div>
  );
}

function Diagnostics({ active }: { active: ModuleKey }) {
  const items = useMemo(
    () => [
      { icon: ShieldCheck, label: "test suite", value: "42 passing" },
      { icon: Gauge, label: "benchmarks", value: "reproducible" },
      { icon: Database, label: "trace mode", value: active },
      { icon: Layers3, label: "package", value: "GitHub Pages ready" },
    ],
    [active],
  );
  return (
    <aside className="grid content-start gap-5">
      <div className="rounded-md border border-graphite-200 bg-graphite-950 p-5 text-white shadow-panel">
        <div className="flex items-center gap-2 text-current">
          <Sparkles aria-hidden="true" size={18} />
          <p className="text-sm font-semibold">Runtime surface</p>
        </div>
        <div className="mt-5 grid gap-3">
          {items.map((item) => {
            const Icon = item.icon;
            return (
              <div key={item.label} className="grid grid-cols-[34px_1fr] items-center gap-3">
                <span className="grid h-8 w-8 place-items-center rounded-md bg-white/10">
                  <Icon aria-hidden="true" size={16} />
                </span>
                <span>
                  <span className="block text-xs uppercase tracking-[0.12em] text-white/55">{item.label}</span>
                  <span className="block text-sm font-semibold">{item.value}</span>
                </span>
              </div>
            );
          })}
        </div>
      </div>
      <div className="rounded-md border border-graphite-200 bg-white p-5">
        <div className="flex items-center gap-2 text-graphite-950">
          <CheckCircle2 aria-hidden="true" size={18} className="text-signal" />
          <p className="text-sm font-semibold">Implementation map</p>
        </div>
        <div className="mt-4 grid gap-2 text-sm text-graphite-700">
          <CodeLine name="forward" value="layer.forward(...)" />
          <CodeLine name="backward" value="layer.backward(...)" />
          <CodeLine name="update" value="optimizer.update(...)" />
          <CodeLine name="history" value="TrainingSnapshot" />
        </div>
      </div>
      <div className="rounded-md border border-graphite-200 bg-white p-5">
        <div className="flex items-center gap-2">
          <MousePointer2 aria-hidden="true" size={18} />
          <p className="text-sm font-semibold">Accessibility</p>
        </div>
        <ul className="mt-4 grid gap-2 text-sm text-graphite-700">
          <li>Keyboard-visible focus states</li>
          <li>Named controls and SVG regions</li>
          <li>Responsive layout without overlap</li>
          <li>High contrast text and controls</li>
        </ul>
      </div>
    </aside>
  );
}

function CodeLine({ name, value }: { name: string; value: string }) {
  return (
    <div className="grid grid-cols-[76px_1fr] gap-2 rounded bg-graphite-50 px-3 py-2">
      <span className="font-mono text-xs text-graphite-500">{name}</span>
      <span className="truncate font-mono text-xs text-graphite-900">{value}</span>
    </div>
  );
}

function MetricBlock({ items }: { items: [string, string][] }) {
  return (
    <div className="grid gap-3 rounded-md border border-graphite-200 bg-white p-4">
      {items.map(([label, value]) => (
        <Metric key={label} label={label} value={value} />
      ))}
    </div>
  );
}

function Metric({ label, value, tone }: { label: string; value: string; tone?: "ok" | "error" }) {
  return (
    <div className="rounded-md border border-graphite-200 bg-white p-3">
      <p className="text-xs uppercase tracking-[0.12em] text-graphite-500">{label}</p>
      <p className={`mt-1 font-mono text-sm font-semibold ${tone === "ok" ? "text-signal" : tone === "error" ? "text-error" : "text-graphite-950"}`}>
        {value}
      </p>
    </div>
  );
}

function SignalStack({ values }: { values: number[] }) {
  return (
    <div className="rounded-md border border-graphite-200 bg-white p-4">
      <p className="text-xs font-semibold uppercase tracking-[0.14em] text-graphite-500">
        activation stack
      </p>
      <div className="mt-4 grid gap-3">
        {values.map((value, index) => (
          <div key={index} className="grid grid-cols-[34px_1fr_46px] items-center gap-3">
            <span className="font-mono text-xs text-graphite-500">a{index + 1}</span>
            <span className="h-3 rounded-full bg-graphite-100">
              <span className="block h-3 rounded-full bg-current" style={{ width: `${value * 100}%` }} />
            </span>
            <span className="text-right font-mono text-xs">{value.toFixed(2)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
