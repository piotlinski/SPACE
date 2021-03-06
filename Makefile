build: ## Build docker image
	docker build -f Dockerfile -t space:latest .

docker_args ?= --gpus all  --volume $(shell pwd):/app --volume $(shell pwd)/data:/app/data
shell: ## Run shell
	docker run -it --rm $(docker_args) --entrypoint /bin/bash space:latest

space_args ?=
run: ## Run model
	docker run --rm $(docker_args) space:latest $(space_args)
