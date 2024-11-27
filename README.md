# auto-catan
Academic technical demonstration of agentic RL implementation of game AI playing `Catan`. It is aimed to showcase a prototype pipeline starting from data acquisition, model training, and model inference

## Table of Content
    1. [Installation](#install)
        - [Install and Start Cluster](#cluster-start)
        - [Install Application Helm Chart](#app-helm)
        - [Health Check](#health)
 
    2. [Installation Troubleshooting](#bugs)
    3. [Dev Dependencies](#deps)

## <a name="install"></a> Installation

> *NOTE: Currently, the installation of the cluster using defined IAC only supports Linux (Tested on Ubuntu 24.04 Server). YES YOU WILL NEED ROOT ACCESS TO YOUR MACHINE* 

### <a name="cluster-start"></a> Install and Start Cluster

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

5. Check status of Cluster

```bash
$ sudo systemctl status k3s
```

*Sample Expected Output:*

```bash
● k3s.service - Lightweight Kubernetes
     Loaded: loaded (/etc/systemd/system/k3s.service; enabled; preset: enabled)
     Active: active (running) since Tue 2024-11-26 02:04:43 UTC; 1h 52min ago
       Docs: https://k3s.io
    Process: 43662 ExecStartPre=/bin/sh -xc ! /usr/bin/systemctl is-enabled --quiet nm-cloud-setup.service 2>/dev/null (code=exited, status=0/SUCCESS)
    Process: 43665 ExecStartPre=/sbin/modprobe br_netfilter (code=exited, status=0/SUCCESS)
    Process: 43666 ExecStartPre=/sbin/modprobe overlay (code=exited, status=0/SUCCESS)
   Main PID: 43668 (k3s-server)
      Tasks: 129
     Memory: 1.3G (peak: 1.3G)
        CPU: 13min 55.545s
```

The more comprehensive way of checking if the cluster is healthy is to check deployment and pod status under the `kube-system` namespace.

```bash
$ kubectl get all -A
```

*Sample Expected Output:*

```bash
NAMESPACE     NAME                                          READY   STATUS      RESTARTS        AGE
kube-system   pod/coredns-6667d8d5d4-g9ddh                  1/1     Running     0               120m
kube-system   pod/helm-install-traefik-crd-2z7bg            0/1     Completed   0               14d
kube-system   pod/helm-install-traefik-rvfxv                0/1     Completed   1               14d
kube-system   pod/local-path-provisioner-595dcfc56f-jq7wn   1/1     Running     0               14d
kube-system   pod/metrics-server-cdcc87586-6b8gw            1/1     Running     24 (123m ago)   14d
kube-system   pod/svclb-traefik-c0b84e80-wwm9v              2/2     Running     0               14d
kube-system   pod/traefik-d7c9c5778-nsmnr                   1/1     Running     25 (124m ago)   14d

NAMESPACE     NAME                     TYPE           CLUSTER-IP     EXTERNAL-IP      PORT(S)                      AGE
default       service/kubernetes       ClusterIP      10.43.0.1      <none>           443/TCP                      14d
kube-system   service/kube-dns         ClusterIP      10.43.0.10     <none>           53/UDP,53/TCP,9153/TCP       14d
kube-system   service/metrics-server   ClusterIP      10.43.125.25   <none>           443/TCP                      14d
kube-system   service/traefik          LoadBalancer   10.43.18.122   192.168.10.137   80:32470/TCP,443:31233/TCP   14d                                                                                                                                                                                                                                                                                                                                                                  NAMESPACE     NAME                                    DESIRED   CURRENT   READY   UP-TO-DATE   AVAILABLE   NODE SELECTOR   AGE
kube-system   daemonset.apps/svclb-traefik-c0b84e80   1         1         1       1            1           <none>          14d                                                                                                              
NAMESPACE     NAME                                     READY   UP-TO-DATE   AVAILABLE   AGE
kube-system   deployment.apps/coredns                  1/1     1            1           14d
kube-system   deployment.apps/local-path-provisioner   1/1     1            1           14d
kube-system   deployment.apps/metrics-server           1/1     1            1           14d
kube-system   deployment.apps/traefik                  1/1     1            1           14d

NAMESPACE     NAME                                                DESIRED   CURRENT   READY   AGE
kube-system   replicaset.apps/coredns-6667d8d5d4                  1         1         1       120m
kube-system   replicaset.apps/local-path-provisioner-595dcfc56f   1         1         1       14d
kube-system   replicaset.apps/metrics-server-cdcc87586            1         1         1       14d
kube-system   replicaset.apps/traefik-d7c9c5778                   1         1         1       14d

NAMESPACE     NAME                                 STATUS     COMPLETIONS   DURATION   AGE
kube-system   job.batch/helm-install-traefik       Complete   1/1           11s        14d
kube-system   job.batch/helm-install-traefik-crd   Complete   1/1           8s         14d
``` 

We want to see the **non-installer** pods to be `Running` and `Ready`. The **installer** pods should not be running, and should be shown as `Completed`.

### <a name="app-helm"></a> Install Application Helm Chart

1. Once the cluster is running and cluster health can be verified, we can start deploying our application. Start by changing directories to root of the project and executing the following commands

```bash
$ helm dependency build ./deploy
```

This will install our custom helm chart that contain RL specific applications and its dependencies.

2. Once the helm dependencies are installed you can bring up the deployment using `tilt`

```bash
$ tilt up --host 0.0.0.0
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

After the edit is complete, restart the `coredns` deployment

```bash
$ kubectl rollout restart deployment.apps/coredns -n kube-system
$ kubectl rollout restart deployment.apps/traefik -n kube-system
$ kubectl rollout restart deployment.apps/metrics-server -n kube-system
``` 

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
