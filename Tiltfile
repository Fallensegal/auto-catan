# Allow 'Default' Context
allow_k8s_contexts('default')

# Build Dockerfile
docker_build('islam25/mlflow-catan', '.', dockerfile='./gitops/mlflow-catan/Dockerfile')

# Deploy: Point Tilt to Charts.yaml
k8s_yaml(helm('deploy'), allow_duplicates=True)

# Expose Minio Port
k8s_resource('chart-minio', port_forwards=9001)
k8s_resource('mlflow-catan', port_forwards=8250)
