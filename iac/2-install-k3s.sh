# Check to See if Kubectl Properly Installed
if kubectl > /dev/null 2>&1; then
    echo "Dependency Check: PASS"
else
    echo "ERROR: Dependency Check Fail before Install"
    exit 1
fi

# Check to See if K3s is Already Installed
if sudo systemctl status k3s --no-pager > /dev/null 2>&1; then
    echo "K3s Exists..... Why are you installing again????"
    exit 1
fi

# Install K3s Using Quick-Start
curl -sfL https://get.k3s.io | sh -

# Set KubeConfig Variable for Kubectl and Helm (Installed Later)
if ! grep -q 'export KUBECONFIG=$HOME/.kube/config' $HOME/.bashrc; then
    echo 'export KUBECONFIG=$HOME/.kube/config' >> $HOME/.bashrc
    echo 'Added KubeConfig to bashrc'
else
    echo "KUBECONFIG export path already set..."
fi

source $HOME/bashrc

# Print K3s Status
sudo systemctl status k3s --no-pager
