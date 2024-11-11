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

# Set File Permission for Kubeconfig
sudo chmod 644 /etc/rancher/k3s/k3s.yaml 

# Set KubeConfig Variable for Kubectl and Helm (Installed Later)
echo "export KUBECONFIG=/etc/rancher/k3s/k3s.yaml" >> $HOME/.bashrc
source $HOME/bashrc

# Print K3s Status
sudo systemctl status k3s --no-pager
