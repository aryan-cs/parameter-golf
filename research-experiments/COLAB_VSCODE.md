# Colab VS Code Workflow

This repo now has a Colab-VS-Code-native path for real training.

## What The Official Extension Supports

The official `googlecolab/colab-vscode` user guide says:

- open a notebook, click `Select Kernel`, and select `Colab`
- choose either `Auto Connect` or `New Colab Server`
- once connected, the Colab activity bar exposes the server's `/content` directory
- you can right click files or folders in VS Code and use `Upload to Colab`
- `Mount Google Drive to Server...` is available from the command palette or the notebook toolbar
- `Server Mounting` and `Colab Terminal` exist, but they are experimental features enabled in settings

## Recommended Setup For This Repo

1. Open `research-experiments/colab_vscode_bootstrap.ipynb` in VS Code.
2. Use `Select Kernel` > `Colab` > `New Colab Server`.
3. In VS Code settings, enable the experimental Colab features if you want:
   - `Server Mounting`
   - `Terminal`
4. Run the notebook cells in order.

The notebook will:

- clone or update this repo into `/content/parameter-golf`
- install runtime dependencies
- optionally download the published challenge data
- run record preflight
- offer a smaller Colab-friendly pilot run
- show how to start the autonomous loop from the Colab runtime itself

## Record Runs vs Colab Pilot Runs

- The staged record manifest still targets `8` GPUs because that is the real record-track intent.
- A standard Colab runtime is more useful for pilot runs and smaller validation jobs unless you have access to a sufficiently large multi-GPU runtime.
- For pilot training on a single Colab GPU, use the provided `colab_pilot_env.json` overrides with `--nproc-per-node 1 --required-cuda-devices 1`.
