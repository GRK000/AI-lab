/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        graphite: {
          50: "#f7f8f8",
          100: "#ebeeee",
          200: "#d5dddd",
          300: "#b2c0bf",
          400: "#899b9a",
          500: "#687c7a",
          600: "#526462",
          700: "#43524f",
          800: "#394542",
          900: "#202725",
          950: "#121715",
        },
        signal: "#22c55e",
        current: "#38bdf8",
        weight: "#f59e0b",
        error: "#f43f5e",
      },
      boxShadow: {
        panel: "0 18px 60px rgba(18, 23, 21, 0.14)",
        focus: "0 0 0 3px rgba(56, 189, 248, 0.38)",
      },
      fontFamily: {
        sans: [
          "Inter",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "Segoe UI",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};
