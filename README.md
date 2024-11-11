# auto-catan
Academic technical demonstration of agentic RL implementation of game AI playing `Catan`. It is aimed to showcase a prototype pipeline starting from data acquisition, model training, and model inference


## Installation

> *NOTE: Currently, the installation of the cluster using defined IAC only supports Linux (Tested on Ubuntu 24.04 Server). YES YOU WILL NEED ROOT ACCESS TO YOUR MACHINE* 

1. Create Local Executable Directory

```bash
$ mkdir -p $HOME/.local/bin
```

2. Add the above directory to path

```bash
$ echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc
$ source $HOME/.bashrc
```
3. Run the scripts in `iac` to bootstrap cluster

```bash
$ ./1-install-kubectl.sh
$ ./2-install-k3s.sh
...
...
```

4. Refresh your terminal to realize system changes

```bash
$ source $HOME/.bashrc
```

## Dev Dependencies
In order to assist in reducing reproducability issues, a list of the following dependencies used to develop the project is provided:

- [Ubuntu Server 24.04](https://ubuntu.com/download/server) *(Kernel: 6.8.0-45-generic)* 
- [Docker](https://www.docker.com/) *(Version 24.0.5, Build ced0996)*
- [Helm](https://helm.sh/docs/intro/install/)
- [Bash]() *(Used in IaC scripting)*

### Some Comments Regarding the Repo
- If you use this, and you are experiencing issues, you are largerly on your own. If you email me I can possibly try to help out (islamwasif3@gmail.com)
- Yes, the steps to get it up is bad/redundant/buggy. I apologize as the project was made in a pinch with some classmates
