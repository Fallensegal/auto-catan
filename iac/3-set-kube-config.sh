# Set Kubeconfig to point to local user

RANCHER_CONFIG="/etc/rancher/k3s/k3s.yaml"
USER_CONFIG="$HOME/.kube/config"

# Check if Config Already Exists
if [ -f $USER_CONFIG ]; then
  owner=$(stat -c '%U' "$USER_CONFIG")

  if [ "$USER" == "$owner" ]; then
    echo "KUBECONFIG file already set"
    exit 0
  fi
fi

mkdir -p $HOME/.kube/

if [ -f $RANCHER_CONFIG ]; then
  echo "Rancher Config Found...."
  sudo cp $RANCHER_CONFIG $HOME/.kube/config
  sudo chown $USER:$USER $HOME/.kube/config
  sudo chmod 644 $HOME/.kube/config
else
  echo "K3s kubeconfig not found, is the cluster up?"
  echo "Check status with 'sudo systemctl status k3s'"
  exit 1
fi

