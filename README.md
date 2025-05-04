Containerized Workflow for Multi-site Autism Classification on ABIDE
====================================================================

This repository reproduces the study [1], which proposes a second-order functional connectivity measure called Tangent Pearson describing the ‘tangent correlation of correlation’ of brain region activity and explores the use of domain adaptation for integrating multi-site neuroimaging data.

It aims to simplify the original code from [1] using PyKale [2] and Nilearn [3], while also demonstrating the use of containers (e.g., Docker and Apptainer) to improve reproducibility and reusability of machine learning experiments.

## Instructions

We assume that the user has has Docker/Apptainer already installed on their machine/computing cluster. For simplicity, we omitted the intructions on how to build the container image locally and focuses on how to pull an existing image from registries like [Docker Hub](https://hub.docker.com) or [GitHub Container Registry (GHCR)](https://ghcr.io).

### Docker
---

1. First, we need to pull the container image from the registry. In our case, the image is stored in GHCR and we can pull the image by using command:
   ```
   docker pull ghcr.io/zarizk7/abide-demo:master
   ```
   > Optionally, we may check if the image is already stored locally with `docker images` command. GHCR images may sometimes not appear on the list. A workaround to check is to use `docker inspect ghcr.io/zarizk7/abide-demo:master`

2. Next, we need to ensure that the image is properly deployable. It uses a Python script as the entrypoint to execute the code within the container. We can first check the arguments available in the image with:
   ```
   docker run ghcr.io/zarizk7/abide-demo:master -h
   ```
   We can observe that there are various arguments available when running the container, with `--input-dir` and `--output-dir` being the "required" ones that must be specified.

3. Finally, we can run the container by executing:
   ```
   docker run --rm \
        -v $INPUT_SOURCE_DIRECTORY:$INPUT_MOUNT_DIRECTORY:ro \
        -v $OUTPUT_SOURCE_DIRECTORY:$OUTPUT_MOUNT_DIRECTORY \
        ghcr.io/zarizk7/abide-demo:master \
        --input-dir $INPUT_MOUNT_DIRECTORY \
        --output-dir $OUTPUT_MOUNT_DIRECTORY \
        --random-state 0
   ```
   The flags and arguments does several things with the container runtime:

   - `-rm`: Removes the container after it finishes executing the code.
   - `-v $INPUT_SOURCE_DIRECTORY:$INPUT_MOUNT_DIRECTORY:ro`: Mounts the `INPUT_SOURCE_DIRECTORY` from local file system and load it into the container as `INPUT_MOUNT_DIRECTORY` with read-only access by `ro`.
   - `-v $OUTPUT_SOURCE_DIRECTORY:$OUTPUT_MOUNT_DIRECTORY:ro`: Mounts the `OUTPUT_SOURCE_DIRECTORY` from local file system and load it into the container as `OUTPUT_MOUNT_DIRECTORY`. Unlike the input directory, we omit `ro`, allowing the container to save its output to the output directory.
   - Both input `--input-dir` and `--output-dir` can then be specified as the `INPUT_MOUNT_DIRECTORY` and `OUTPUT_MOUNT_DIRECTORY` respectively.
   - `--random-state`: A positive integer used to control the randomness of the algorithms and evaluation, `0` in our case. While optional, we highly recommend to set the `--random-state` flag to ensure reproducible results.

4. The container will continue to be deployed until the internal code finished running. In the `OUTPUT_SOURCE_DIRECTORY`, there will be various files generated, including:
   - `args.yaml`: All of the arguments defined during the container's deployment time.
   - `cv_results.csv`: Cross-validation runtime, prediction scores, and hyperparameters.
   - `inputs.npz`: Features extracted from the data used to train the model.
   - `model.joblib`: A trained model using the optimal hyperparameter settings identified during the tuning process.
   - `phenotypes.csv`: Preprocessed phenotypic information of the subjects used for domain adaptation.

Congratulations! You have sucessfully to train and evaluate an Autism classication model with a Docker container.

### Apptainer
---

While Docker is the most widely used containerization platform, high performance computing (HPC) clusters in many cases won't have it installed for security purposes due to requiring root privileges to set up and deploy its containers. There are multiple alternatives to Docker that doesn't require root privileges like **Apptainer** or **Podman**. For this instruction, we will use Apptainer and assume that the users have logged in to their clusters and deployed a worker node in an interactive session (e.g., `srun`) to run the container. Like Docker's instruction, we omitted the instructions for building a container image locally.

1. While Apptainer has a different file format for its images, we can conveniently pull the container image from registries like Docker Hub or GHCR with command:
   ```
   apptainer pull $IMAGE.sif docker://ghcr.io/zarizk7/abide-demo:master
   ```
    > `IMAGE` can also be a directory along with the specified filename.

2. Similarly to Docker, we can check the image's entrypoint available flag and argument, calling:
   ```
   apptainer run --pwd / $IMAGE.sif -h
   ```
    > Unlike Docker's example, we have to set the container’s working directory to the root `/` using `--pwd /`. This is a workaround for a common issue when using containers pulled from Docker registries.
    
    > By default, Apptainer sets the container’s working directory to the current host directory, which may not match the working directory defined in Docker-based images. To avoid unexpected behavior, it’s best to explicitly set the working directory to match the one originally defined by the container with `--pwd $ORIGINAL_WORKDIR` or use `/` if it isn't specified in the Docker image. See Apptainer issue #2573 for details.

3. To run deploy a container in a computing cluster in Apptainer is slightly different to Docker's with:
   ```
   apptainer run \
        -B $INPUT_SOURCE_DIRECTORY:$INPUT_MOUNT_DIRECTORY:ro \
        -B $OUTPUT_SOURCE_DIRECTORY:$OUTPUT_MOUNT_DIRECTORY \
        --pwd / \
        $IMAGE.sif \
        --input-dir $INPUT_MOUNT_DIRECTORY \
        --output-dir $OUTPUT_MOUNT_DIRECTORY \
        --random-state 0
   ```
   Unlike Docker, which uses `-v` to mount directories, Apptainer uses `-B` (or `--bind`) to achieve the same functionality.

   > On some clusters, certain directories may already be automatically mounted from the host operating system into the container environment. In such cases, you may not need to explicitly bind those directories using the `-B` flag.

4. The Apptainer container will run similarly as Docker's and is expected to produce the same output.

Congratulations! You have sucessfully to train and evaluate an Autism classication model with an Apptainer container.

## References

[1] *Kunda, Mwiza, Shuo Zhou, Gaolang Gong, and Haiping Lu*. **Improving Multi-Site Autism Classification via Site-Dependence Minimization and Second-Order Functional Connectivity**. IEEE Transactions on Medical Imaging 42, no. 1 (January 2023): 55–65. https://doi.org/10.1109/TMI.2022.3203899.

[2] *Lu, Haiping, Xianyuan Liu, Shuo Zhou, Robert Turner, Peizhen Bai, Raivo E. Koot, Mustafa Chasmai, Lawrence Schobs, and Hao Xu*. **PyKale**. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. New York, NY, USA: ACM, 2022. https://doi.org/10.1145/3511808.3557676.

[3] *Nilearn contributors, Ahmad Chamma, Aina Frau-Pascual, Alex Rothberg, Alexandre Abadie, Alexandre Abraham, Alexandre Gramfort, et al*. **Nilearn**. Zenodo, 2025. https://doi.org/10.5281/ZENODO.8397156.