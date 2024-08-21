# Stop and remove all containers
docker stop $(docker ps -q)
docker rm $(docker ps -a -q)

# Remove all images
docker rmi $(docker images -q)

# Remove all volumes
docker volume rm $(docker volume ls -q)

# Remove all custom networks
docker network rm $(docker network ls -q)

# Clean up all unused data
docker system prune -a --volumes
