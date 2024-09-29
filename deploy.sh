#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define variables
DOCKER_IMAGE_NAME="your-docker-image-name"  # Replace with your Docker image name
DOCKER_COMPOSE_FILE="docker-compose.yml"      # Name of the Docker Compose file
ENV_FILE=".env"                                # Optional: specify an environment variables file

# Build Docker images
echo "Building Docker images..."
docker-compose -f $DOCKER_COMPOSE_FILE build

# Run Docker containers
echo "Starting services..."
docker-compose -f $DOCKER_COMPOSE_FILE up -d  # Start in detached mode

# Optional: Run database migrations or seed data if necessary
# echo "Running migrations..."
# docker-compose -f $DOCKER_COMPOSE_FILE exec db bash -c "your_migration_command"

# Optional: Deploy to a cloud service (if applicable)
# echo "Deploying to AWS..."
# aws ecs update-service --cluster your-cluster --service your-service --force-new-deployment

# Check the status of the services
echo "Checking the status of services..."
docker-compose -f $DOCKER_COMPOSE_FILE ps

echo "Deployment completed successfully!"
