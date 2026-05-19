module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTHONUNBUFFERED: "1",
          PYTHONIOENCODING: "utf-8",
          GRADIO_SERVER_NAME: "127.0.0.1"
        },
        path: "app",
        message: [
          "python -u app.py"
        ],
        on: [{
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
