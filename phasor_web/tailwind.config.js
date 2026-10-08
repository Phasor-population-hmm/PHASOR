/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: "class",
  content: ["./app/**/*.{js,jsx}", "./components/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["var(--font-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-mono)", "ui-monospace", "monospace"],
      },
      keyframes: {
        blink: { "0%,100%": { opacity: 1 }, "50%": { opacity: 0 } },
        rise: { from: { opacity: 0, transform: "translateY(8px)" }, to: { opacity: 1, transform: "translateY(0)" } },
        dash: { to: { strokeDashoffset: 0 } },
      },
      animation: {
        blink: "blink 1s step-start infinite",
        rise: "rise .35s cubic-bezier(.16,1,.3,1)",
        dash: "dash 1.2s ease-out forwards",
      },
    },
  },
  plugins: [],
};
