# auto-catan
Academic technical demonstration of agentic RL implementation of game AI playing `Catan`. It is aimed to showcase a prototype pipeline starting from data acquisition, model training, and model inference

## Installation

> *NOTE: Currently, the installation of the cluster using defined IAC only supports Linux (Tested on Ubuntu 24.04 Server) 

1. Create Local Executable Directory

```bash
$ mkdir -p $HOME/.local/bin
```

2. Run the scripts in `iac` to bootstrap cluster

```bash
$ ./1-install-kubectl.sh
$ ./2-install-k3s.sh
...
...
```

## Dev Dependencies
In order to assist in reducing reproducability issues, a list of the following dependencies used to develop the project is provided:

- [Ubuntu Server 24.04](https://ubuntu.com/download/server) *(Kernel: 6.8.0-45-generic)* 
- [Docker](https://www.docker.com/) *(Version 24.0.5, Build ced0996)*
