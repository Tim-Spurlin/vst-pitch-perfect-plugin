#!/bin/bash
# Google Cloud setup script for VST Pitch Perfect Plugin
# Sets up all required GCP resources for deployment

# Load color codes for pretty output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set default values
PROJECT_ID="vocal-transformation-ai"
REGION="us-central1"
ZONE="us-central1-a"
CLUSTER_NAME="vocal-transformation-cluster"
NODE_COUNT=2
MACHINE_TYPE="n1-standard-4"
GPU_TYPE="nvidia-tesla-t4"
GPU_COUNT=1

# Function to display usage
function show_usage {
    echo -e "${BLUE}VST Pitch Perfect Plugin - GCP Setup Script${NC}"
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p, --project PROJECT_ID    GCP project ID (default: $PROJECT_ID)"
    echo "  -r, --region REGION         GCP region (default: $REGION)"
    echo "  -z, --zone ZONE             GCP zone (default: $ZONE)"
    echo "  -c, --cluster CLUSTER_NAME  GKE cluster name (default: $CLUSTER_NAME)"
    echo "  -n, --nodes NODE_COUNT      Number of nodes (default: $NODE_COUNT)"
    echo "  -m, --machine MACHINE_TYPE  Machine type (default: $MACHINE_TYPE)"
    echo "  -g, --gpu-type GPU_TYPE     GPU type (default: $GPU_TYPE)"
    echo "  -gc, --gpu-count GPU_COUNT  GPU count (default: $GPU_COUNT)"
    echo "  -h, --help                  Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--project)
        PROJECT_ID="$2"
        shift 2
        ;;
        -r|--region)
        REGION="$2"
        shift 2
        ;;
        -z|--zone)
        ZONE="$2"
        shift 2
        ;;
        -c|--cluster)
        CLUSTER_NAME="$2"
        shift 2
        ;;
        -n|--nodes)
        NODE_COUNT="$2"
        shift 2
        ;;
        -m|--machine)
        MACHINE_TYPE="$2"
        shift 2
        ;;
        -g|--gpu-type)
        GPU_TYPE="$2"
        shift 2
        ;;
        -gc|--gpu-count)
        GPU_COUNT="$2"
        shift 2
        ;;
        -h|--help)
        show_usage
        exit 0
        ;;
        *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_usage
        exit 1
        ;;
    esac
done

# Function to check if a command exists
function command_exists {
    command -v "$1" &> /dev/null
}

# Check if gcloud is installed
if ! command_exists gcloud; then
    echo -e "${RED}Error: Google Cloud SDK (gcloud) is not installed.${NC}"
    echo "Please install it from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo -e "${YELLOW}You are not logged in to Google Cloud SDK.${NC}"
    echo "Please run: gcloud auth login"
    exit 1
fi

# Print the setup details
echo -e "${BLUE}=== VST Pitch Perfect Plugin - GCP Setup ===${NC}"
echo -e "Project ID:     ${GREEN}$PROJECT_ID${NC}"
echo -e "Region:         ${GREEN}$REGION${NC}"
echo -e "Zone:           ${GREEN}$ZONE${NC}"
echo -e "Cluster Name:   ${GREEN}$CLUSTER_NAME${NC}"
echo -e "Node Count:     ${GREEN}$NODE_COUNT${NC}"
echo -e "Machine Type:   ${GREEN}$MACHINE_TYPE${NC}"
echo -e "GPU Type:       ${GREEN}$GPU_TYPE${NC}"
echo -e "GPU Count:      ${GREEN}$GPU_COUNT${NC}"
echo

# Ask for confirmation
read -p "Do you want to proceed with this setup? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Setup aborted.${NC}"
    exit 0
fi

# Create project if it doesn't exist
echo -e "\n${BLUE}=== Creating/Setting Project ===${NC}"
if ! gcloud projects describe "$PROJECT_ID" &> /dev/null; then
    echo -e "Creating project ${GREEN}$PROJECT_ID${NC}..."
    gcloud projects create "$PROJECT_ID"
fi

# Set the current project
echo -e "Setting current project to ${GREEN}$PROJECT_ID${NC}..."
gcloud config set project "$PROJECT_ID"

# Enable required APIs
echo -e "\n${BLUE}=== Enabling Required APIs ===${NC}"
APIS=(
    "compute.googleapis.com"
    "container.googleapis.com"
    "artifactregistry.googleapis.

    "artifactregistry.googleapis.com"
    "containerregistry.googleapis.com"
    "cloudbuild.googleapis.com"
    "iam.googleapis.com"
    "ml.googleapis.com"
    "storage.googleapis.com"
    "monitoring.googleapis.com"
    "logging.googleapis.com"
)

echo "Enabling required APIs..."
for api in "${APIS[@]}"; do
    echo -e "Enabling ${GREEN}$api${NC}..."
    gcloud services enable "$api"
done

# Create service account
echo -e "\n${BLUE}=== Creating Service Account ===${NC}"
SA_NAME="vocal-ai-service"
SA_EMAIL="$SA_NAME@$PROJECT_ID.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe "$SA_EMAIL" &> /dev/null; then
    echo -e "Creating service account ${GREEN}$SA_NAME${NC}..."
    gcloud iam service-accounts create "$SA_NAME" \
        --display-name="VST Pitch Perfect Service Account"
fi

# Grant necessary permissions
echo -e "Granting permissions to service account..."
ROLES=(
    "roles/storage.admin"
    "roles/container.admin"
    "roles/iam.serviceAccountUser"
    "roles/ml.admin"
)

for role in "${ROLES[@]}"; do
    gcloud projects add-iam-policy-binding "$PROJECT_ID" \
        --member="serviceAccount:$SA_EMAIL" \
        --role="$role"
done

# Create storage buckets
echo -e "\n${BLUE}=== Creating Storage Buckets ===${NC}"
BUCKETS=(
    "$PROJECT_ID-models"
    "$PROJECT_ID-data"
    "$PROJECT_ID-training"
)

for bucket in "${BUCKETS[@]}"; do
    if ! gsutil ls -b "gs://$bucket" &> /dev/null; then
        echo -e "Creating bucket ${GREEN}gs://$bucket${NC}..."
        gsutil mb -l "$REGION" "gs://$bucket"
    else
        echo -e "Bucket ${GREEN}gs://$bucket${NC} already exists."
    fi
done

# Create GKE cluster
echo -e "\n${BLUE}=== Creating GKE Cluster ===${NC}"
if ! gcloud container clusters describe "$CLUSTER_NAME" --zone="$ZONE" &> /dev/null; then
    echo -e "Creating GKE cluster ${GREEN}$CLUSTER_NAME${NC}..."
    
    gcloud container clusters create "$CLUSTER_NAME" \
        --zone="$ZONE" \
        --num-nodes="$NODE_COUNT" \
        --machine-type="$MACHINE_TYPE" \
        --disk-size=100 \
        --scopes=storage-full,cloud-platform \
        --enable-autoscaling \
        --min-nodes=2 \
        --max-nodes=5 \
        --enable-autorepair \
        --enable-vertical-pod-autoscaling
else
    echo -e "GKE cluster ${GREEN}$CLUSTER_NAME${NC} already exists."
fi

# Add GPU node pool if requested
if [ "$GPU_COUNT" -gt 0 ]; then
    echo -e "\n${BLUE}=== Adding GPU Node Pool ===${NC}"
    GPU_POOL_NAME="gpu-pool"
    
    if ! gcloud container node-pools describe "$GPU_POOL_NAME" --cluster="$CLUSTER_NAME" --zone="$ZONE" &> /dev/null; then
        echo -e "Creating GPU node pool ${GREEN}$GPU_POOL_NAME${NC}..."
        
        gcloud container node-pools create "$GPU_POOL_NAME" \
            --cluster="$CLUSTER_NAME" \
            --zone="$ZONE" \
            --num-nodes=1 \
            --machine-type="n1-standard-8" \
            --accelerator="type=$GPU_TYPE,count=$GPU_COUNT" \
            --disk-size=100 \
            --enable-autoscaling \
            --min-nodes=1 \
            --max-nodes=3 \
            --scopes=storage-full,cloud-platform
    else
        echo -e "GPU node pool ${GREEN}$GPU_POOL_NAME${NC} already exists."
    fi
    
    # Install NVIDIA drivers
    echo -e "Installing NVIDIA drivers on GPU nodes..."
    kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
fi

# Get cluster credentials
echo -e "\n${BLUE}=== Configuring kubectl ===${NC}"
gcloud container clusters get-credentials "$CLUSTER_NAME" --zone="$ZONE"

# Create and configure Artifact Registry repository
echo -e "\n${BLUE}=== Creating Artifact Registry Repository ===${NC}"
REPO_NAME="vocal-transformation"

if ! gcloud artifacts repositories describe "$REPO_NAME" --location="$REGION" &> /dev/null; then
    echo -e "Creating Artifact Registry repository ${GREEN}$REPO_NAME${NC}..."
    gcloud artifacts repositories create "$REPO_NAME" \
        --repository-format=docker \
        --location="$REGION" \
        --description="VST Pitch Perfect Docker images"
else
    echo -e "Artifact Registry repository ${GREEN}$REPO_NAME${NC} already exists."
fi

# Configure Docker to use Artifact Registry
echo -e "Configuring Docker for Artifact Registry..."
gcloud auth configure-docker "$REGION-docker.pkg.dev"

# Set up monitoring
echo -e "\n${BLUE}=== Setting Up Monitoring ===${NC}"
kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.7.0/aio/deploy/recommended.yaml

# Generate service account key
echo -e "\n${BLUE}=== Generating Service Account Key ===${NC}"
KEY_PATH="$HOME/vocal-ai-key.json"
gcloud iam service-accounts keys create "$KEY_PATH" \
    --iam-account="$SA_EMAIL"

echo -e "Service account key saved to ${GREEN}$KEY_PATH${NC}"
echo -e "Add this to your environment: export GOOGLE_APPLICATION_CREDENTIALS=\"$KEY_PATH\""

# Summary
echo -e "\n${BLUE}=== Setup Complete! ===${NC}"
echo -e "GCP Project:               ${GREEN}$PROJECT_ID${NC}"
echo -e "GKE Cluster:               ${GREEN}$CLUSTER_NAME${NC}"
echo -e "Service Account:           ${GREEN}$SA_EMAIL${NC}"
echo -e "Storage Buckets:           ${GREEN}gs://$PROJECT_ID-models, gs://$PROJECT_ID-data, gs://$PROJECT_ID-training${NC}"
echo -e "Artifact Registry:         ${GREEN}$REGION-docker.pkg.dev/$PROJECT_ID/$REPO_NAME${NC}"
echo -e "Service Account Key:       ${GREEN}$KEY_PATH${NC}"

echo
echo -e "${GREEN}Your GCP environment is now ready for VST Pitch Perfect Plugin deployment!${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Build and push Docker images"
echo "2. Deploy to Kubernetes"
echo "3. Train your vocal transformation model"
echo
echo "Run './build-and-push.sh' to build and push Docker images."