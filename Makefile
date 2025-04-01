BASE_NAME=abbfn2
USER_NAME=miguel-1

GCP_PROJECT=int-research-tpu
GCP_ZONE=us-central2-b
ACCELERATOR_TYPE=v4-8
RUNTIME_VERSION=tpu-vm-v4-base

SSH_FLAG="-A"
SSH_KEY_FILE="~/.ssh/id_ed25519"

GIT_REPO=instadeepai/AbBFN2
GIT_BRANCH := $(shell git rev-parse --abbrev-ref HEAD)
CHECKOUT_DIR=abbfn2

PATH_TO_GOOGLE_APPLICATION_CREDENTIALS=/Users/m.braganca/Documents/ProtBFN/core/docker/int-research-tpu-db-access-service-account.json

PORT=8891
DOCKER_VARS_TO_PASS = --env-file $(PATH_TO_ENV_FILE)

DOCKER_IMAGE_NAME = abbfn2/abbfn

PATH_TO_PARAMS_FILE=./params.pkl

#######
# TPU #
#######

# Shared set-up.
BASE_CMD=gcloud alpha compute tpus tpu-vm
BASE_CMD_Q=gcloud alpha compute tpus queued-resources

NAME=$(BASE_NAME)-$(ACCELERATOR_TYPE)-$(USER_NAME)

PATH_TO_ENV_FILE=./.env

SSH_ARGS = --ssh-flag=$(SSH_FLAG) --ssh-key-file=$(SSH_KEY_FILE)

WORKER=all

# Create Queue

.PHONY: create_q

create_q:
	$(BASE_CMD_Q) create \
 		$(NAME)-q --zone $(GCP_ZONE) \
 		--node-id $(NAME) \
		--project $(GCP_PROJECT) \
		--accelerator-type $(ACCELERATOR_TYPE) \
		--runtime-version $(RUNTIME_VERSION) \
		--best-effort


# Prepare VM

checkout_on_vm:
	$(BASE_CMD) ssh --zone $(GCP_ZONE) $(NAME) $(SSH_ARGS) \
		--project $(GCP_PROJECT) \
		--worker=$(WORKER) \
		--command="git clone -b ${GIT_BRANCH} https://${GITHUB_USER_TOKEN}:${GITHUB_ACCESS_TOKEN}@github.com/${GIT_REPO}.git ${CHECKOUT_DIR}"

send_env:
	$(BASE_CMD) scp --zone=$(GCP_ZONE) --project $(GCP_PROJECT) --worker=$(WORKER) \
		$(PATH_TO_ENV_FILE) $(NAME):~/$(CHECKOUT_DIR)/$(PATH_TO_ENV_FILE)

send_params:
	$(BASE_CMD) scp --zone=$(GCP_ZONE) --project $(GCP_PROJECT) --worker=$(WORKER) \
		$(PATH_TO_PARAMS_FILE) $(NAME):~/$(CHECKOUT_DIR)/$(PATH_TO_PARAMS_FILE)

send_credentials:
	$(BASE_CMD) scp --zone=$(GCP_ZONE) --project $(GCP_PROJECT) --worker=$(WORKER) \
		$(PATH_TO_GOOGLE_APPLICATION_CREDENTIALS) $(NAME):~/$(CHECKOUT_DIR)/google_application_credentials.json

prepare_vm: checkout_on_vm send_env send_params send_credentials

.PHONY: checkout_on_vm send_project_configuration send_env send_credentials prepare_vm



.PHONY: run pull

run:
	$(BASE_CMD) ssh --zone $(GCP_ZONE) $(NAME) --project $(GCP_PROJECT) --worker=$(WORKER) --command="$(command)"

pull:
	# echo "Pulling latest version of the ${GIT_REPO}/${GIT_BRANCH} on TPU ${NAME}"
	make run command="cd ${CHECKOUT_DIR} && git fetch && git pull"


###############
# Remote work #
###############

ssh_tpu:
	$(BASE_CMD) ssh --zone=$(GCP_ZONE) $(NAME) --project $(GCP_PROJECT)

set_ssh_agent:
	eval "$(ssh-agent -s)"
	ssh-add ~/.ssh/id_ed25519
	ssh-add ~/.ssh/google_compute_engine

mount_docker:
	sudo docker run -it --rm --privileged -p 8891:8891 --network host --name abbfn2 --env-file $(PATH_TO_ENV_FILE) -v /home/m.braganca/abbfn2:/app abbfn2 /bin/bash

kill_tpu_container:
	$(BASE_CMD) ssh --zone=$(GCP_ZONE) $(NAME) --project $(GCP_PROJECT) --worker=$(WORKER) --command="sudo docker kill abbfn2"


LOCAL_DEST = ./tpu_results

define GET_LATEST_FOLDER1
	$(BASE_CMD) ssh --zone=$(GCP_ZONE) $(NAME) --project $(GCP_PROJECT) --worker=$(WORKER) --command \
	"ls -d abbfn2/outputs/*/ 2>/dev/null | grep '$\' | sort | tail -n 1 | xargs -I {} basename {}"
endef

define GET_LATEST_FOLDER2
	$(BASE_CMD) ssh --zone=$(GCP_ZONE) $(NAME) --project $(GCP_PROJECT) --worker=$(WORKER) --command \
	"ls -d abbfn2/outputs/$(1)/*/ 2>/dev/null | grep '$\' | sort | tail -n 1 | xargs -I {} basename {}"
endef

.PHONY: scp_last_folder
scp_last_folder:
	@echo "Fetching latest folders from remote..."
	$(eval LAST_FOLDER1 := $(shell $(GET_LATEST_FOLDER1)))
	$(eval LAST_FOLDER2 := $(shell $(call GET_LATEST_FOLDER2,$(LAST_FOLDER1))))
	$(eval REMOTE_FULL_PATH := $(NAME):abbfn2/outputs/$(LAST_FOLDER1)/$(LAST_FOLDER2)/validation/inpaint/)

	@if [ -z "$(LAST_FOLDER2)" ]; then \
		echo "No valid folders found on remote server."; \
		exit 1; \
	fi

	@echo "Copying from remote: $(REMOTE_FULL_PATH)"
	$(BASE_CMD) scp --zone $(GCP_ZONE) --project $(GCP_PROJECT) --recurse $(REMOTE_FULL_PATH) $(LOCAL_DEST)
	@echo "Successfully copied $(REMOTE_FULL_PATH) to $(LOCAL_DEST)"


########################
#################
# CPU, TPU or GPU. If you enter an incorrect option, it will default to CPU-only

ACCELERATOR = TPU

# variables
WORK_DIR = $(PWD)
USER_ID = $$(id -u)
GROUP_ID = $$(id -g)

DOCKER_BUILD_FLAGS = --build-arg USER_ID=$(USER_ID) \
	--build-arg GROUP_ID=$(GROUP_ID)

ifeq ($(ACCELERATOR), GPU)
	DOCKER_BUILD_FLAGS += --build-arg BASE_IMAGE="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04"
endif

ifeq ($(ACCELERATOR), TPU)
	DOCKER_BUILD_FLAGS +=  --build-arg BASE_IMAGE="ubuntu:20.04"
endif

ifeq ($(ACCELERATOR), CPU)
	DOCKER_BUILD_FLAGS +=  --build-arg BASE_IMAGE="ubuntu:20.04"
endif

DOCKER_RUN_FLAGS_CPU = --rm \
	--shm-size=1024m \
	-v $(WORK_DIR):/app

DOCKER_RUN_FLAGS_GPU = ${DOCKER_RUN_FLAGS_CPU} --gpus all 

DOCKER_RUN_FLAGS_TPU = --rm --user root --privileged \
	-v $(WORK_DIR):/app

# Select appropriate run flags based on accelerator
ifeq ($(ACCELERATOR), CPU)
	DOCKER_RUN_FLAGS = $(DOCKER_RUN_FLAGS_CPU)
else ifeq ($(ACCELERATOR), GPU)
	DOCKER_RUN_FLAGS = $(DOCKER_RUN_FLAGS_GPU)
else ifeq ($(ACCELERATOR), TPU)
	DOCKER_RUN_FLAGS = $(DOCKER_RUN_FLAGS_TPU)
endif

# image + container name
DOCKER_IMAGE_NAME = abbfn2
DOCKER_CONTAINER_NAME = abbfn2_container


.PHONY: build
build:
	sudo docker build -t $(DOCKER_IMAGE_NAME) -f Dockerfile . \
		$(DOCKER_BUILD_FLAGS) --build-arg ACCELERATOR=$(ACCELERATOR)

.PHONY: unconditional
unconditional:
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_NAME) python experiments/unconditional.py $(RUN_ARGS)

.PHONY: inpaint
inpaint:
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_NAME) python experiments/inpaint.py $(RUN_ARGS)

.PHONY: humanization
humanization:
	sudo docker run $(DOCKER_RUN_FLAGS) $(DOCKER_IMAGE_NAME) python experiments/humanization.py $(RUN_ARGS)