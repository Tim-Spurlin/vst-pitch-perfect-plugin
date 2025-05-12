#!/bin/bash
# Upload preprocessed data to Google Cloud Storage

# Default values
PROJECT_ID="vocal-transformation-ai"
BUCKET_NAME="${PROJECT_ID}-data"
LOCAL_DATA_DIR="../processed_features"

# Function to display usage
function show_usage {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -p, --project PROJECT_ID    GCP project ID (default: $PROJECT_ID)"
    echo "  -b, --bucket BUCKET_NAME    GCP bucket name (default: $BUCKET_NAME)"
    echo "  -d, --dir LOCAL_DATA_DIR    Local data directory (default: $LOCAL_DATA_DIR)"
    echo "  -h, --help                  Show this help message"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -p|--project)
        PROJECT_ID="$2"
        BUCKET_NAME="${PROJECT_ID}-data"
        shift 2
        ;;
        -b|--bucket)
        BUCKET_NAME="$2"
        shift 2
        ;;
        -d|--dir)
        LOCAL_DATA_DIR="$2"
        shift 2
        ;;
        -h|--help)
        show_usage
        exit 0
        ;;
        *)
        echo "Unknown option: $1"
        show_usage
        exit 1
        ;;
    esac
done

# Ensure Google Cloud SDK is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: Google Cloud SDK (gcloud) not found"
    echo "Please install it from https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if user is logged in
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "You are not logged in to Google Cloud SDK"
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set current project
echo "Setting current project to: $PROJECT_ID"
gcloud config set project $PROJECT_ID

# Check if bucket exists
if ! gsutil ls -b gs://$BUCKET_NAME &> /dev/null; then
    echo "Bucket gs://$BUCKET_NAME does not exist"
    echo "Creating bucket..."
    gsutil mb -l us-central1 gs://$BUCKET_NAME
fi

# Check if data directory exists
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "Error: Local data directory $LOCAL_DATA_DIR does not exist"
    exit 1
fi

# Count files to upload
FILE_COUNT=$(find "$LOCAL_DATA_DIR" -type f | wc -l)
if [ $FILE_COUNT -eq 0 ]; then
    echo "Error: No files found in $LOCAL_DATA_DIR"
    exit 1
fi

echo "Found $FILE_COUNT files to upload"
echo "Uploading data to gs://$BUCKET_NAME..."

# Upload data with progress
gsutil -m cp -r "$LOCAL_DATA_DIR"/* gs://$BUCKET_NAME/

# Verify upload
UPLOADED_COUNT=$(gsutil ls -r gs://$BUCKET_NAME/** | wc -l)
echo "Upload complete! $UPLOADED_COUNT objects in bucket"

# Create a README file with information about the dataset
README_CONTENT="Vocal Transformation AI Dataset
Uploaded on: $(date)
Source: $LOCAL_DATA_DIR
File count: $FILE_COUNT
"

echo "$README_CONTENT" > /tmp/dataset_readme.txt
gsutil cp /tmp/dataset_readme.txt gs://$BUCKET_NAME/README.txt
rm /tmp/dataset_readme.txt

# Display bucket information
echo "Bucket information:"
gsutil du -sh gs://$BUCKET_NAME

echo "Upload completed successfully!"