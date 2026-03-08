module.exports = {
  run: [{
    method: "shell.run",
    params: {
      message: "git pull"
    }
  }, {
    method: "shell.run",
    params: {
      path: "app",
      message: "if [ -d .git ]; then git pull; else echo \"app not found or not a git repo; skipping app update.\"; fi"
    }
  }]
}
