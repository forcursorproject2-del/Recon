module.exports = {
  content: ["./web/index.html", "./web/**/*.js"],
  theme: {
    extend: {
      colors: {
        cyan: '#00d4ff',
        purple: '#9d4edd',
        'dark-bg': '#0f0f1a',
        'dark-card': '#1a1a2e'
      },
      fontFamily: {
        sans: ['Segoe UI Variable', 'system-ui', 'sans-serif']
      }
    }
  },
  plugins: []
}
