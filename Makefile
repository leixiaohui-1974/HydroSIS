# HydroSIS Docker Makefile
# Provides convenient commands for building and running HydroSIS with Docker

.PHONY: help build run stop clean logs test lint format

# Default target
help:
	@echo "HydroSIS Docker Commands:"
	@echo "  build     - Build the HydroSIS Docker image"
	@echo "  run       - Run HydroSIS in a Docker container"
	@echo "  stop      - Stop all running containers"
	@echo "  clean     - Remove Docker images and containers"
	@echo "  logs      - Show logs from running containers"
	@echo "  test      - Run tests in Docker"
	@echo "  lint      - Run linting in Docker"
	@echo "  format    - Format code in Docker"
	@echo "  shell     - Open a shell in the running container"

# Build the Docker image
build:
	docker build -t hydrosis:latest .

# Run HydroSIS with docker-compose
run:
	docker-compose up -d

# Stop all services
stop:
	docker-compose down

# Remove Docker images and containers
clean:
	docker-compose down -v --rmi all
	docker system prune -f

# Show logs from running containers
logs:
	docker-compose logs -f

# Run tests in Docker
test:
	docker-compose run --rm hydrosis-app python -m pytest tests/ -v --cov=hydrosis

# Run linting in Docker
lint:
	docker-compose run --rm hydrosis-app flake8 hydrosis/
	docker-compose run --rm hydrosis-app mypy hydrosis/

# Format code in Docker
format:
	docker-compose run --rm hydrosis-app black hydrosis/
	docker-compose run --rm hydrosis-app isort hydrosis/

# Open a shell in the running container
shell:
	docker-compose exec hydrosis-app /bin/bash

# Development target - build and run
dev: build run

# Production target - build and run with production settings
prod:
	docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build

# Backup data
backup:
	docker-compose exec postgres pg_dump -U hydrosis hydrosis > backup_$(shell date +%Y%m%d_%H%M%S).sql

# Restore data from backup
restore:
	@echo "Usage: make restore BACKUP_FILE=<backup_file>"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Error: BACKUP_FILE not specified"; exit 1; fi
	docker-compose exec -T postgres psql -U hydrosis hydrosis < $(BACKUP_FILE)

# Monitor resources
monitor:
	docker stats

# Update dependencies
update:
	docker-compose run --rm hydrosis-app pip install --upgrade -r requirements.txt

# Security scan
security:
	docker-compose run --rm hydrosis-app safety check

# Performance profiling
profile:
	docker-compose run --rm hydrosis-app python -m cProfile -o profile.stats examples/run_sample_workflow.py

# Generate documentation
docs:
	docker-compose run --rm hydrosis-app sphinx-build -b html docs/ docs/_build/

# Database migrations
migrate:
	docker-compose run --rm hydrosis-app alembic upgrade head

# Create database schema
schema:
	docker-compose run --rm hydrosis-app python -c "from hydrosis.portal.storage import create_sqlalchemy_state; create_sqlalchemy_state('postgresql://hydrosis:password@postgres:5432/hydrosis')"
