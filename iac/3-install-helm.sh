# Script to Install Helm

if helm > /dev/null 2>&1; then
    echo "Helm: Its Already Installed...."
    echo
    exit 0
fi

curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

if helm > /dev/null 2>&1; then
    echo "Helm: Installation Successful"
    echo
else
    echo "ERROR: Helm Installation was not successful"
    echo "Check if the helm installation directory is in PATH"
    exit 1
fi

