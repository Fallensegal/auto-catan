# auto-catan
Academic technical demonstration of agentic RL implementation of game AI playing `Catan`. It is aimed to showcase a prototype pipeline starting from data acquisition, model training, and model inference

## Table of Content
1. [Installation](#install)
    - [Pre-requisites](#prereq)
    - [Install and Start Cluster](#cluster-start)
    - [Install Application Helm Chart](#app-helm)
    - [Running Parallel Trainings (*NO UI*)](#curl)
    - [Alternative Execution](#exec)

2. [Installation Troubleshooting](#bugs)
3. [Dev Dependencies](#deps)

## <a name="install"></a>Installation

> *NOTE: Currently, the installation of the cluster using defined IAC only supports Linux (Tested on Ubuntu 24.04 Server). YES YOU WILL NEED ROOT ACCESS TO YOUR MACHINE*

### <a name="prereq"></a> Pre-requisites
For effective reproduction, please setup a `linux/amd64` VM with `NAT` networking and with `Ubuntu 24.04` Server. Perform a minimal install with SSH server (no additional packages selected from snap) and install the following packages:

```bash
apt-get install git build-essential curl ca-certificates
```

Following that, clone the `main` branch onto the home directory of your VM

### <a name="cluster-start"></a> Install and Start Cluster

1. Create Local Executable Directory

```bash
mkdir -p $HOME/.local/bin
```

2. Add the above directory to path

```bash
echo "export PATH=$PATH:$HOME/.local/bin" >> $HOME/.bashrc
source $HOME/.bashrc
```
3. Run the scripts in `iac` to bootstrap cluster

```bash
./1-install-kubectl.sh
./2-install-k3s.sh
...
...
```

4. Refresh your terminal to realize system changes

```bash
source $HOME/.bashrc
```

5. Check status of Cluster: *Cluster status should be active*

```bash
sudo systemctl status k3s
```

The more comprehensive way of checking if the cluster is healthy is to check deployment and pod status under the `kube-system` namespace. For K3s, `coredns`, `local-path-provisioner`, `metrics-server`, and `traefik` should be all running and ready.

```bash
kubectl get all -A
```
We want to see the **non-installer** pods to be `Running` and `Ready`. The **installer** pods should not be running, and should be shown as `Completed`.

6. Create the `catan` namespace for the deployment, and set it as the default namespace

```bash
kubectl create namespace catan
kubectl kubectl config set-context --current --namespace=catan
```

### <a name="app-helm"></a> Install Application Helm Chart

1. Once the cluster is running and cluster health can be verified, we can start deploying our application. Start by changing directories to root of the project and executing the following commands

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add kedacore https://kedacore.github.io/charts
helm dependency build ./deploy
```

This will install our custom helm chart that contain RL specific applications and its dependencies.

2. Once the helm dependencies are installed you can bring up the deployment using `tilt`

```bash
tilt up --host 0.0.0.0
```  
3. You can check if the deployment is properly working by looking at the `Tilt` status and navigating to `Minio`

<img src="./docs/tilt_screenshot.png" alt="Minio" width=1600 height=400>

> *Tilt showing all deployments are operational*

<img src="./docs/minio_screenshot.png" alt="Minio" width=600 height=400>

> *Splash Screen for Minio*

### <a name="curl"></a>Running Parallel Trainings (*NO UI*)

Unfortunately, (*as of the current release*), we do not have asynchronous handling of client requests to the web server or a UI. As a result, the requests can be sent to the backend using `curl` or the swagger UI at `<ip-address>:8251/docs`. The `config.md` document contain valid hyper-parameters to submit to the backend.

> *Example `curl` command*:

```bash
curl -X POST <vm-ip-address>:8251/inf/run_experiment \
-H "Content-Type: application/json" \
-d <json-object-from-config.md>
```

If using the Swagger UI, then multiple tabs will need to be opened in order to submit multiple jobs, since asynchronous handling of requests, or a client side application capable of handling asynchronous requests are not implemented yet.

### <a name="exec"></a>Alternative Execution
If you do not feel like going through all of the steps and troubleshooting and just want to run the model training experiment, go to the `rl_catan` folder, and just run `main.py`. If you would like to change the hyper-parameters, change the constant definitions under `Configurations.py`. In order to run, execute the following commands:

1. Download the `uv` package manager:
<br></br>
> For Linux and MacOS
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

> For Windows
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. Execute `main.py`

```bash
cd rl-catan
uv sync
source .venv/bin/activate
python3 main.py
```

## <a name="bugs"></a>Installation Troubleshooting

1. CoreDNS `CrashBackOffLoop`

If you are running into pods in `coredns` experiencing `CrashBackOffLoop` due to the control plane not being able to `curl` the health probe, it is most likely because you do not have a DNS server configured with your host machine in `/etc/resolv.conf` 

Execute the following command:

```bash
$ kubectl edit configmap coredns -n kube-system
```

Remove the following field and its assocaited predicates (if there are any)

```bash
 forward . /etc/resolv.conf
```

Add the following in its place

```bash
forward . 8.8.8.8 8.8.4.4
```

After the edit is complete, restart the `coredns` deployment

```bash
$ kubectl rollout restart deployment.apps/coredns -n kube-system
$ kubectl rollout restart deployment.apps/traefik -n kube-system
$ kubectl rollout restart deployment.apps/metrics-server -n kube-system
``` 
2. Docker Failing to Build Images (`deb.debian.org` not found):

If you are developing `Dockerfiles` that do not have an image already built, and you are in an Ubuntu VM, and chose to install `docker` from the OS installation menu, there is a good chance `docker` was installed using `snap`.

If `docker` is having a hard time resolving QDN's you will have to manually add google's DNS server to the docker config in the following location

```bash
/var/snap/docker/current/config
```
Edit the file `daemon.json` (note: you will have to be sudo to run this) and add the following lines:

```json
{
    "dns": ["8.8.8.8", "8.8.4.4"]
}
```

This should now allow docker to properly build docker files when before it might have been having issues resolving hosts.

## <a name="deps"></a> Dev Dependencies
In order to assist in reducing reproducability issues, a list of the following dependencies used to develop the project is provided:

- [Ubuntu Server 24.04](https://ubuntu.com/download/server) *(Kernel: 6.8.0-45-generic)* 
- [Docker](https://www.docker.com/) *(Version 24.0.5, Build ced0996)*
- [Helm](https://helm.sh/docs/intro/install/)
- [Bash](https://www.gnu.org/software/bash/) *(Used in IaC scripting)*
- [Tilt](https://tilt.dev/)

### Some Comments Regarding the Repo
- If you use this, and you are experiencing issues, you are largerly on your own. If you email me I can possibly try to help out (islamwasif3@gmail.com)
- Yes, the steps to get it up is bad/redundant/buggy. I apologize as the project was made in a pinch with some classmates

## Authors:

1. Wasif Islam
    - islam25@purdue.edu
    - w.islam@obscuritylabs.com
    - islamwasif3@gmail.com

2. John Michael Van Treek
    - jvantree@purdue.edu

3. Diege Reyes Olmos
    - dreyesol@purdue.edu