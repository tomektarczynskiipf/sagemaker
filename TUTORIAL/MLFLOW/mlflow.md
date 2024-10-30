# MLflow

## Basic information

**MLflow** is an open-source platform designed to simplify the machine learning lifecycle. It provides a comprehensive set of tools and frameworks to manage and track the end-to-end ML development process, including experimentation, reproducibility, deployment, and collaboration. 

MLflow consists of several key components:

1. **MLflow Tracking** - MLflow Tracking allows users to track and log experiments, parameters, metrics, and artifacts associated with their ML projects. It provides a centralized repository for storing experiment data and enables easy comparison and reproducibility of different runs. By integrating with popular ML libraries and frameworks, MLflow Tracking allows users to log and track experiments across various environments.

2. **MLflow Model Registry** - MLflow Model Registry provides a centralized repository for managing and organizing ML models across their lifecycle. The Model Registry allows users to register, version, and track models, providing governance, collaboration, and control over the deployment and lifecycle management of models.

3. **MLflow Projects** - MLflow Projects provides a standardized format for packaging and sharing ML code, making it easier to reproduce and deploy ML projects. With MLflow Projects, users can define project dependencies, specify entry points, and encapsulate the entire ML workflow into a portable format. This allows for seamless collaboration and reproducibility across different platforms and environments.

4. **MLflow Models** -  MLflow Models enables users to package trained ML models in a format that can be easily deployed and used in different production environments. It provides a consistent way to serialize and load models, regardless of the underlying framework or library used for training. MLflow Models supports various deployment options, including serving models via REST APIs, batch inference, and integration with popular serving platforms.

5. **MLflow Pipeline** - MLflow Pipeline is a component introduced in MLflow 1.14.0 that enables users to define and execute complex ML workflows. It allows users to define multi-step pipelines that encompass data preparation, model training, evaluation, and deployment. MLflow Pipeline provides a unified interface for orchestrating and managing these workflows, making it easier to build scalable and reproducible ML pipelines.



---

## MLflow Full Hierarchy

### 1. **Experiment**
   - A collection of related runs, typically representing a goal, dataset, or analysis task within the project. Experiments are grouped by names and IDs and help organize different attempts at achieving a specific objective within a project.
   - **Contains:** Multiple runs.

### 2. **Run**
   - A single execution of code or task within an experiment. Each run logs parameters, metrics, and artifacts, allowing for detailed tracking and analysis of individual executions.
   - **Contains:** Optional nested runs (child runs).
   - **Parent Run**: The main or primary run in a workflow, which may contain several nested runs (sub-tasks).

### 3. **Nested Run**
   - A run that exists within the context of a parent run, representing a sub-task or step in a complex workflow.
   - **Example Use Cases for Nested Runs**:
     - **Pipeline Step**: Sub-steps in a model training pipeline (e.g., data preprocessing, model training, and evaluation).
     - **Hyperparameter Tuning**: Different configurations within a main hyperparameter tuning task.
     - **Evaluation Metrics Calculation**: Separate metrics or stages within a single task.
   - **Contains**: Sub-runs if there are multiple levels of nesting.
   - **Nesting Levels**: Nested runs can further contain additional levels of nested runs, creating a hierarchy of tasks within tasks.

### Hierarchy in our workflow:
  - Experiment is an equivalent of our project, i.e.: 04_churn_pol
  - Option 1:
    - Every fit of a model is in separate run
    - Each run contains only one fit of a model
    - We don't used nested runs
  - Option 2:
    - Multiple similar fits of a model are grouped into a single run, i.e.: hyperparameter_tuning
    - Run can contain multiple nested runs.
  - It is up to us to decide how we are going to use it. It is like with files in a folder, if we are going to have more than 1000 files in a folder then it is going to be hard to find the one which we are looking for.

---

## MLflow costs

### Tracking Server Compute

| Tracking Server Size | Price Per Hour(USD) |
| -------| ----------- |
| Small  | 0.648       | 
| Medium | 1.122       |
| Large  | 2.0481      |

### Tracking Server Storage

| Storage Type             | Price Per GB-month |
|--------------------------|--------------------|
| Tracking Server Storage  | 0.11               |




