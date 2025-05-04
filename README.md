Containerized Workflow for Multi-site Autism Classification on ABIDE
====================================================================

This repository reproduces the study [1], which proposes a second-order functional connectivity measure called Tangent Pearson describing the ‘tangent correlation of correlation’ of brain region activity and explores the use of domain adaptation for integrating multi-site neuroimaging data evaluated on the ABIDE [2] dataset for autism classification.

It aims to simplify the original code from [1] using PyKale [3] and Nilearn [4], while also demonstrating the use of containers (e.g., Docker and Apptainer) to improve reproducibility and reusability of machine learning experiments.

# Instructions

We assume that the user has has Docker/Apptainer already installed on their machine/computing cluster. For simplicity, we omitted the intructions on how to build the container image locally and focuses on how to pull an existing image from registries like [Docker Hub](https://hub.docker.com) or [GitHub Container Registry (GHCR)](https://ghcr.io).

## Docker

1. First, we need to pull the container image from the registry. In our case, the image is stored in GHCR and we can pull the image by using command:
   ```sh
   docker pull ghcr.io/zarizk7/abide-demo:master
   ```
   > Optionally, we may check if the image is already stored locally with `docker images` command. GHCR images may sometimes not appear on the list. A workaround to check is to use `docker inspect ghcr.io/zarizk7/abide-demo:master`

2. Next, we need to ensure that the image is properly deployable. It uses a Python script as the entrypoint to execute the code within the container. We can first check the arguments available in the image with:
   ```sh
   docker run ghcr.io/zarizk7/abide-demo:master -h
   ```
   We can observe that there are various arguments available when running the container, with `--input-dir` and `--output-dir` being the "required" ones that must be specified.

3. Finally, we can run the container by executing:
   ```sh
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

Congratulations! We have sucessfully to train and evaluate an Autism classication model with a Docker container.

## Apptainer

While Docker is the most widely used containerization platform, high performance computing (HPC) clusters in many cases won't have it installed for security purposes due to requiring root privileges to set up and deploy its containers. There are multiple alternatives to Docker that doesn't require root privileges like **Apptainer** or **Podman**. For this instruction, we will use Apptainer and assume that the users have logged in to their clusters and deployed a worker node in an interactive session (e.g., `srun`) to run the container. Like Docker's instruction, we omitted the instructions for building a container image locally.

1. While Apptainer has a different file format for its images, we can conveniently pull the container image from registries like Docker Hub or GHCR with command:
   ```sh
   apptainer pull $IMAGE.sif docker://ghcr.io/zarizk7/abide-demo:master
   ```
    > `IMAGE` can also be a directory along with the specified filename.

    Once the image has been pulled and built, we will find the image in a `*.sif` file format in the working/specified directory.

2. Similarly to Docker, we can check the image's entrypoint available flag and argument, calling:
   ```sh
   apptainer run $IMAGE.sif -h
   ```
    > By default, Apptainer sets the container’s working directory to the current directory on the host. However, this may not match the working directory defined in a Docker-based image, which typically defaults to the root directory `/`. If the Docker container’s entrypoint expects a specific working directory and uses relative paths, this mismatch can cause errors when running the image with Apptainer. This issue usually does not occur when the entrypoint uses absolute paths.
    
    > To avoid this, we should explicitly set the working directory in Apptainer using `--pwd $ORIGINAL_WORKDIR`, or use `--pwd /` if the original working directory isn’t specified (e.g. `apptainer run --pwd / $IMAGE.sif ...`).
    
    > See [Apptainer issue #2573](https://github.com/apptainer/apptainer/issues/2573) for details.

3. To run deploy a container in a computing cluster in Apptainer is slightly different to Docker's with:
   ```sh
   apptainer run \
        -B $INPUT_SOURCE_DIRECTORY:$INPUT_MOUNT_DIRECTORY:ro \
        -B $OUTPUT_SOURCE_DIRECTORY:$OUTPUT_MOUNT_DIRECTORY \
        $IMAGE.sif \
        --input-dir $INPUT_MOUNT_DIRECTORY \
        --output-dir $OUTPUT_MOUNT_DIRECTORY \
        --random-state 0
   ```
   Unlike Docker, which uses `-v` to mount directories, Apptainer uses `-B` (or `--bind`) to achieve the same functionality.

   > On some clusters, certain directories may already be pre-mounted from the host operating system into the container environment. In such cases, we may not need to explicitly mount those directories using the `-B` flag.

4. The Apptainer container will run similarly as Docker's and is expected to produce the same output.

Congratulations! We have sucessfully to train and evaluate an Autism classication model with an Apptainer container.

# References

[1] *Kunda, Mwiza, Shuo Zhou, Gaolang Gong, and Haiping Lu*. **Improving Multi-Site Autism Classification via Site-Dependence Minimization and Second-Order Functional Connectivity**. IEEE Transactions on Medical Imaging 42, no. 1 (January 2023): 55–65. https://doi.org/10.1109/TMI.2022.3203899.

[2] *Nielsen, Jared A., Brandon A. Zielinski, P. Thomas Fletcher, Andrew L. Alexander, Nicholas Lange, Erin D. Bigler, Janet E. Lainhart, and Jeffrey S. Anderson*. **Multisite Functional Connectivity MRI Classification of Autism: ABIDE Results**. Frontiers in Human Neuroscience 7 (25 September 2013): 599. https://doi.org/10.3389/fnhum.2013.00599.

[3] *Lu, Haiping, Xianyuan Liu, Shuo Zhou, Robert Turner, Peizhen Bai, Raivo E. Koot, Mustafa Chasmai, Lawrence Schobs, and Hao Xu*. **PyKale**. In Proceedings of the 31st ACM International Conference on Information & Knowledge Management. New York, NY, USA: ACM, 2022. https://doi.org/10.1145/3511808.3557676.

[4] *Nilearn contributors, Ahmad Chamma, Aina Frau-Pascual, Alex Rothberg, Alexandre Abadie, Alexandre Abraham, Alexandre Gramfort, et al*. **Nilearn**. Zenodo, 2025. https://doi.org/10.5281/ZENODO.8397156.